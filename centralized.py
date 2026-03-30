import os, time, random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import glob
from torch.utils.data import TensorDataset, random_split, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import roc_auc_score, average_precision_score, PrecisionRecallDisplay, RocCurveDisplay
from torch.distributed import init_process_group, destroy_process_group
from resnet import ResNet1d

########## set device (torchrun parallellization)
def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

######### set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
##########

def _load_snap(model, gpu_id, snapshot_path="snapshot.pt"):
    loc = f'cuda:{gpu_id}'
    snapshot = torch.load(snapshot_path, map_location=loc)
    model.module.load_state_dict(snapshot["MODEL_STATE"])
    print(f"=== Loaded an existing snapshot -- device: {gpu_id} === at {snapshot_path}")
    return snapshot["EPOCHS_RUN"]

def _save_snap(model, epoch, snapshot_path="snapshot.pt"):
    snapshot = {
        "MODEL_STATE": model.module.state_dict(),
        "EPOCHS_RUN": epoch
    }
    torch.save(snapshot, snapshot_path)
    print(f"Saved snapshot at Epoch: {epoch} === at {snapshot_path}")

# =========================================================================#
# ========================== Training Functions ===========================#
# =========================================================================#
def train_loop(epoch, chunk, dataloader, model, optimizer, loss_function, device, snapshot_path="snapshot.pt"):
    if os.path.exists(snapshot_path):
        print("Loading snapshot...")
        epochs_run = _load_snap(model, device, snapshot_path)

    dataloader.sampler.set_epoch(epoch)
    # model to training mode (important to correctly handle dropout or batchnorm layers)
    model.train()

    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    # progress bar def
    train_pbar = tqdm(dataloader, desc="Training Epoch {epoch:2d} Chunk {chunk}".format(epoch=epoch, chunk=chunk), leave=True)

    sigmoid = torch.nn.Sigmoid().to(device)
    # training loop
    for i, (traces, diagnoses) in enumerate(train_pbar):
        traces, diagnoses = traces.to(device), diagnoses.to(device)
        
        # data to device (CPU or GPU if available)
        for j, (x,y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            curr_loss = loss_function(pred, y)
            curr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Update accumulated values
        total_loss += curr_loss.detach().cpu().numpy()
        n_entries += len(traces)

        # Update progress bar
        train_pbar.set_postfix({'loss': total_loss / n_entries})
    train_pbar.close()
    return total_loss / n_entries

def eval_loop(epoch, chunk, dataloader, model, loss_function, device):
    # model to evaluation mode (important to correctly handle dropout or batchnorm layers)
    model.eval()
    y_trues, y_preds = [], []
    
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    avg_precisions = []  # avg precision (pr-auc)
    roc_aucs = [] # roc_auc

    # progress bar def
    eval_pbar = tqdm(dataloader, desc="Evaluation Epoch {epoch:2d} Chunk {chunk:2d}".format(epoch=epoch, chunk=chunk), leave=True)
    sigmoid = torch.nn.Sigmoid().to(device)
    # evaluation loop
    for traces_cpu, diagnoses_cpu in eval_pbar:
        # data to device (CPU or GPU if available)
        traces, diagnoses = traces_cpu.to(device), diagnoses_cpu.to(device)
        with torch.no_grad():
            for x,y in dataloader:
                xt, yt = x.to(device), y.to(device)
                pred = model(xt)
                curr_loss = loss_function(pred, yt)

                if len(np.unique(yt.cpu())) == 2: 
                    # if both positive and negative truth values are present, compute the avg. precision
                    avg_precisions.append(average_precision_score(yt.cpu(), sigmoid(pred).cpu()))
                    roc_aucs.append(roc_auc_score(yt.cpu(), sigmoid(pred).cpu()))
                    if len(y_trues) > 0 and len(y_preds) > 0:
                        y_trues = torch.cat((y_trues, torch.flatten(yt.cpu())))
                        y_preds = torch.cat((y_preds, torch.flatten(sigmoid(pred).cpu())))
                    else:
                        y_trues = torch.flatten(yt.cpu())
                        y_preds = torch.flatten(sigmoid(pred).cpu())
                
            # Update accumulated values
            total_loss += curr_loss.detach().cpu().numpy()
            n_entries += len(traces)
            # Update progress bar
            eval_pbar.set_postfix({
                'loss': total_loss / n_entries, 
                'avg_prec': np.mean(avg_precisions), 
                'roc_auc': np.mean(roc_aucs)
            })
    eval_pbar.close()
    return y_trues, y_preds, total_loss / n_entries, np.mean(avg_precisions), np.mean(roc_aucs)
##########

# =========================================================================#
# ========================= Training PROCEDURE ============================#
# =========================================================================#

def main(num_rounds, num_chunks, id_=int(random.uniform(127962, 236777))):
    ddp_setup()
    gpu_id = int(os.environ["LOCAL_RANK"])
    # =============== Define model ============================================#
    tqdm.write("Define model...")
    model = ResNet1d(input_dim=(12, 4096),
                         blocks_dim=list(zip([64, 128, 196, 256, 320], # net_filter_size
                             [4096, 1024, 256, 64, 16])), # net_sequence_length
                         n_classes=1,
                         kernel_size=17,
                         dropout_rate=0.4) 
    # model = CustomCNN()
    tqdm.write("Done!\n")
    
    learning_rate = 1e-3
    weight_decay = 0.1
    num_epochs = num_rounds # 10
    batch_size = 256

    pos_weight = torch.tensor([48], device=gpu_id) # mean ratio of neg. samples / pos. samples in all chunks of code15 to tackle class imbalance (only around 2% are positives)
    
    # =============== Define loss function ====================================#
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # =============== Define optimizer ========================================#
    tqdm.write("Define optimiser...")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    tqdm.write("Done!\n")
    
    # =============== Define lr scheduler =====================================#
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)
   
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])

    train_files_list = sorted(glob.glob("data/code15-12l/*.hdf5"))

    # =============== Build data loaders ======================================#
    tqdm.write("Building data loaders...")
    
    vloaders = []
    for i, filepath in enumerate(sorted(glob.glob("data/code15-12l/*.hdf5"))):
        # build data loaders
        if filepath.replace("data/code15-12l/", "") in ["exams_part0.hdf5", "exams_part1.hdf5", "exams_part2.hdf5", "exams_part3.hdf5"]:
            path_to_h5_train, path_to_csv_train = filepath, 'data/code15-12l/exams.csv' # path_to_records = 'data/codesubset/RECORDS.txt'
        
            # load traces
            f = h5py.File(path_to_h5_train, 'r')
            traces = torch.tensor(np.array(f['tracings'], dtype=np.float32), dtype=torch.float32)[:-1,:,:]
            
            # load labels
            ids_traces = np.array(f['exam_id'])
            df = pd.read_csv(path_to_csv_train)
            #df = df.drop_duplicates(subset=["patient_id"])
            f.close()
            df.set_index('exam_id', inplace=True)
            df = df.reindex(ids_traces).dropna(subset=["AF"]) # make sure the order is the same
            labels = torch.tensor(np.array(df['AF'], dtype=np.float16), dtype=torch.float16, device=gpu_id).reshape(-1,1)
            #traces = traces[:labels.shape[0],:,:]
            print("\nat", i, ">> number of pos. examples >>", len(df[df['AF']==1])) 
            print(">> weight >>", len(df[df['AF']==0])/len(df[df['AF']==1]))

            # load dataset
            dataset = TensorDataset(traces, labels)
            #len_dataset = len(dataset)
            #n_classes = len(torch.unique(labels))
        
            train_files_list.remove(filepath)
            vloaders.append(dataset)
            print("at", filepath, " >> put in validation!")

    vset = torch.utils.data.ConcatDataset(vloaders)
    vloader = DataLoader(vset, batch_size=256, shuffle=False, sampler=DistributedSampler(vset))

    # =============== Train model =============================================#
    tqdm.write("Training...")
    best_loss = np.inf

    # allocation
    avgpreclist, rocauclist = [], []
    train_loss_all, valid_loss_all = [], []
    lrs = []
    counter = 0

    size = len(train_files_list) if num_chunks == -1 else num_chunks

    # loop over epochs
    for epoch in tqdm(range(1, num_epochs + 1)):
        for i, filepath in enumerate(train_files_list[:size]):
            if i > size-4:
                break
            if i % 4 == 0:
                datasets = []
                gap = 4 if len(train_files_list[:size])-4 >= i else len(train_files_list[:size])-i
                for j in range(gap):
                    # build data loaders
                    path_to_h5_train, path_to_csv_train = train_files_list[i+j], 'data/code15-12l/exams.csv' # path_to_records = 'data/codesubset/RECORDS.txt'
                
                    # load traces
                    f = h5py.File(path_to_h5_train, 'r')
                    traces = torch.tensor(np.array(f['tracings'], dtype=np.float32), dtype=torch.float32)[:-1,:,:]
                    
                    # load labels
                    ids_traces = np.array(f['exam_id'])
                    df = pd.read_csv(path_to_csv_train)
                    #df = df.drop_duplicates(subset=['patient_id'])
                    #traces = traces[:labels.shape[0],:,:]
                    f.close()

                    df = df.set_index('exam_id')
                    df = df.reindex(ids_traces).dropna(subset=["AF"]) # make sure the order is the same
                    labels = torch.tensor(np.array(df['AF'], dtype=np.float16), dtype=torch.float16, device=gpu_id).reshape(-1,1)
                    print("\nat", i, ">> number of pos. examples >>", len(df[df['AF']==1])) 
                    print(">> weight >>", len(df[df['AF']==0])/len(df[df['AF']==1]))
                    datasets.append(TensorDataset(traces, labels))
                
                # load dataset
                dataset = ConcatDataset(datasets)
                tloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(dataset, shuffle=True))
            
                # training loop
                train_loss = train_loop(epoch, filepath.replace("data/code15-12l/", ""), tloader, model, optimizer, loss_function, device=gpu_id, snapshot_path=f"{id_}-snapshot.pt")

                # validation loop
                yt, ypred, valid_loss, avg_precisions, avg_roc_aucs = eval_loop(
                    epoch, i, vloader, 
                    model, loss_function, 
                    device=gpu_id
                )

                # collect losses
                train_loss_all.append(train_loss)
                valid_loss_all.append(valid_loss)
                
                # collect validation metrics
                avgpreclist.append(avg_precisions)
                rocauclist.append(avg_roc_aucs)

                # save checkpoints between epochs
                if gpu_id == 0:
                    _save_snap(model, epoch, snapshot_path=f"{id_}-snapshot.pt")
                    pd.DataFrame({
                        "epoch": np.arange(counter+1), 
                        "train_loss": train_loss_all, 
                        "valid_loss": valid_loss_all, 
                        "mAP": avgpreclist, 
                        "ROC/AUC": rocauclist
                    }).to_csv(f"{id_}-results-partwise-lr{learning_rate}-ep{num_epochs}-exams{size-4}.csv", index=False)
        
                    # =============== PLOTTING  =============================================#
                    PrecisionRecallDisplay.from_predictions(yt, ypred) 
                    #precision, recall, thresholds = precision_recall_curve(yt, ypred)
                    #ax.fill_between() ----> uncertainties
                    plt.savefig(f"{id_}-pr_curve-partwise.png", dpi=300)
                    plt.close()
                    
                    fig2 = plt.figure(figsize=(8,6), dpi=300)
                    ax2 = fig2.add_subplot()
                    ax2.set_title("Train-Validation Loss Curves - CODE-15% Centralized Training")
                    ax2.set_xlabel("Iterations")
                    ax2.set_ylabel("BCE Loss")
                    ax2.plot(np.arange(counter+1), train_loss_all, color="blue", label="Train")
                    ax2.plot(np.arange(counter+1), valid_loss_all, color="orange", label="Validation")
                    ax2.legend(loc="best")
                    fig2.tight_layout()
                    fig2.savefig(f"{id_}-losses-partwise-centralizedcode15.png")
                    plt.close()
        
                    RocCurveDisplay.from_predictions(yt, ypred)
                    plt.savefig(f"{id_}-roc_curve-partwise.png", dpi=300)
                    plt.close()
                
                    fig = plt.figure(figsize=(8,6), dpi=300)
                    ax = fig.add_subplot()
                    ax.set_title("PR-AUC and ROC-AUC - CODE-15% Centralized Training")
                    ax.set_xlabel("Iterations")
                    ax.set_ylabel("Average Precision")
                    ax.plot(np.arange(counter+1), avgpreclist, color="blue", label="PR-AUC")
                    ax.plot(np.arange(counter+1), rocauclist, color="orange", label="ROC/AUC")
                    ax.legend(loc="best")
                    fig.tight_layout()
                    fig.savefig(f"{id_}-pr+roc_aucs-partwise-centralizedcode15.png") 
                    plt.close()
            
                    # save best model: here we save the model only for the lowest validation loss
                    if valid_loss < best_loss:
                        # Save model parameters
                        torch.save({'model': model.state_dict()}, f'{id_}-resnetmodel-centralizedcode15-partwise.pth')
                        # Update best validation loss
                        best_loss = valid_loss        

                counter += 1

                # statement
                model_save_state = "best model -> saved" if valid_loss < best_loss else ""
                
                # Update learning rate with lr-scheduler
                if lr_scheduler:
                    print("lr >>>>", optimizer.param_groups[0]['lr'])
                    lrs.append(optimizer.param_groups[0]['lr'])
                    
                    # Print message
                    tqdm.write('Epoch {epoch:2d}: \t'
                            'Train Loss {train_loss:.6f} \t'
                            'Valid Loss {valid_loss:.6f} \t'
                            '{model_save}'
                            .format(epoch=epoch,
                                    train_loss=train_loss,
                                    valid_loss=valid_loss,
                                    model_save=model_save_state))

        if lr_scheduler and gpu_id == 0:
            fig2 = plt.figure(figsize=(8,6), dpi=300)
            ax2 = fig2.add_subplot()
            ax2.set_title("LR Scheduling - Centralized CODE-15%")
            ax2.set_xlabel("Iterations")
            ax2.set_ylabel("Learning Rate")
            ax2.plot(np.arange(counter), lrs, color="blue")
            ax2.grid(True)
            fig2.tight_layout()
            fig2.savefig(f"{id_}-lrscheduling-centralizedcode15.png")
            plt.close()
            lr_scheduler.step(avgpreclist[-1])
        #if lrs[-1] < 1e-8:
        #    sys.exit(0)
                    
    destroy_process_group()
    # =======================================================================#

if __name__ == "__main__":
    main(num_rounds=8, num_chunks=-1, id_=int(random.uniform(1209310, 2230240)))
