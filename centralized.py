import os, time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import glob
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import roc_auc_score, average_precision_score, PrecisionRecallDisplay, RocCurveDisplay
from torch.distributed import init_process_group, destroy_process_group

########## set device (torchrun parallellization)
def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

######### set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
##########

# =========================================================================#
########## Model
# =========================================================================#
def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding

def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample

class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y


class ResNet1d(nn.Module):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8):
        super(ResNet1d, self).__init__()
        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = x.transpose(2,1)
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully conected layer
        x = self.lin(x)
        return x

def _load_snap(model, gpu_id, snapshot_path):
    loc = f'cuda:{gpu_id}'
    snapshot = torch.load(snapshot_path, map_location=loc)
    model.module.load_state_dict(snapshot["MODEL_STATE"])
    print(f"=== Loaded an existing snapshot -- device: {gpu_id} ===")
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

    # model to training mode (important to correctly handle dropout or batchnorm layers)
    model.train()

    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    # progress bar def
    train_pbar = tqdm(dataloader, desc="Training Epoch {epoch:2d} Chunk {chunk:2d}".format(epoch=epoch, chunk=chunk), leave=True)

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

def main(num_rounds, num_chunks):
    ddp_setup()
    gpu_id = int(os.environ["LOCAL_RANK"])
    # =============== Define model ============================================#
    tqdm.write("Define model...")
    model = ResNet1d(input_dim=(12, 4096),
                         blocks_dim=list(zip([64, 128, 196, 256], # net_filter_size
                             [4096, 1024, 256, 64])), # net_sequence_length
                         n_classes=1,
                         kernel_size=17,
                         dropout_rate=0.4) 
    # model = CustomCNN()
    tqdm.write("Done!\n")
    
    learning_rate = 1e-5
    weight_decay = 1e-3
    num_epochs = num_rounds # 10
    batch_size = 256

    pos_weight = torch.tensor([48], device=gpu_id) # mean ratio of neg. samples / pos. samples in all chunks of code15 to tackle class imbalance (only around 2% are positives)
    
    # =============== Define loss function ====================================#
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # =============== Define optimizer ========================================#
    tqdm.write("Define optimiser...")
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    tqdm.write("Done!\n")
    
    # =============== Define lr scheduler =====================================#
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
   
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])

    # =============== Build data loaders ======================================#
    tqdm.write("Building data loaders...")
    
    tloaders = []
    vloaders = []
    
    size = len(glob.glob("data/code15-12l/*.hdf5")) if num_chunks == -1 else num_chunks

    for i, filepath in enumerate(sorted(glob.glob("data/code15-12l/*.hdf5"))[:size]):
        path_to_h5_train, path_to_csv_train = filepath, 'data/code15-12l/exams.csv' # path_to_records = 'data/codesubset/RECORDS.txt'
        
        # load traces
        f = h5py.File(path_to_h5_train, 'r')
        traces = torch.tensor(np.array(f['tracings'], dtype=np.float32), dtype=torch.float32)[:-1,:,:]
        
        # load labels
        ids_traces = np.array(f['exam_id'])
        df = pd.read_csv(path_to_csv_train)
        f.close()
        df.set_index('exam_id', inplace=True)
        df = df.reindex(ids_traces).dropna(subset=["AF"]) # make sure the order is the same
        labels = torch.tensor(np.array(df['AF'], dtype=np.float16), dtype=torch.float16, device=gpu_id).reshape(-1,1)
        #print("\nat", i, ">> number of pos. examples >>", len(df[df['AF']==1])) #print(">> weight >>", len(df[df['AF']==0])/len(df[df['AF']==1]))

        # load dataset
        dataset = TensorDataset(traces, labels)
        len_dataset = len(dataset)
        n_classes = len(torch.unique(labels))
        del traces, labels
    
        # build data loaders
        if filepath.replace("data/code15-12l/", "") in ["exams_part0.hdf5", "exams_part1.hdf5", "exams_part2.hdf5", "exams_part3.hdf5"]:
            vloaders.append(dataset)
            print("at", filepath, " >> put in validation!")
        else:
            tloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(dataset)))
            print("at", filepath, " >> put in training!")

    vloader = DataLoader(torch.utils.data.ConcatDataset(vloaders), batch_size=batch_size, shuffle=False, sampler=DistributedSampler(dataset))
    
    tqdm.write("Done!\n")

    # =============== Train model =============================================#
    tqdm.write("Training...")
    best_loss = np.inf

    # allocation
    avgpreclist, rocauclist = [], []
    y_trues, y_preds = [], []
    train_loss_all, valid_loss_all = [], []

    counter = 0
    
    # loop over epochs
    for epoch in tqdm(range(1, num_epochs + 1)):
        for i in range(len(tloaders)):
            # training loop
            train_loss = train_loop(epoch, i, tloaders[i], model, optimizer, loss_function, device=gpu_id)

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

            # collect classifications and truth labels
            if len(y_trues) > 0 and gpu_id == 0:
                y_trues = torch.cat((y_trues, yt))
                y_preds = torch.cat((y_preds, ypred))
            else:
                y_trues = yt
                y_preds = ypred

            pd.DataFrame({
                    "epoch": np.arange(counter+1), 
                    "train_loss": train_loss_all, 
                    "valid_loss": valid_loss_all, 
                    "mAP": avgpreclist, 
                    "ROC/AUC": rocauclist
            }).to_csv(f"results-lr{learning_rate}-epochs{num_epochs}-exams{list(range(num_chunks))}.csv", index=False)

            # save checkpoints between epochs
            if gpu_id == 0:
                _save_snap(model, epoch)
 
            # =============== PLOTTING  =============================================#
            PrecisionRecallDisplay.from_predictions(y_trues, y_preds) 
            #ax.fill_between() ----> uncertainties
            plt.savefig("pr_curve.png", dpi=300)
            plt.close()
            
            fig2 = plt.figure(figsize=(8,6), dpi=300)
            ax2 = fig2.add_subplot()
            ax2.set_title("Train-Validation Loss Curves - CODE-15% Centralized Training")
            ax2.set_xlabel("Iterations")
            ax2.set_ylabel("BCE Loss")
            ax2.plot(np.arange(counter+1)/len(tloaders), train_loss_all, color="blue", label="Train")
            ax2.plot(np.arange(counter+1)/len(tloaders), valid_loss_all, color="orange", label="Validation")
            ax2.legend(loc="best")
            fig2.tight_layout()
            fig2.savefig("losses-centralizedcode15.png")

            RocCurveDisplay.from_predictions(y_trues, y_preds)
            plt.savefig("roc_curve.png", dpi=300)
            plt.close()
        
            fig = plt.figure(figsize=(8,6), dpi=300)
            ax = fig.add_subplot()
            ax.set_title("PR-AUC and ROC-AUC - CODE-15% Centralized Training")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Average Precision")
            ax.plot(np.arange(counter+1)/len(tloaders), avgpreclist, color="blue", label="PR-AUC")
            ax.plot(np.arange(counter+1)/len(tloaders), rocauclist, color="orange", label="ROC/AUC")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig("pr+roc_aucs-centralizedcode15.png")

            plt.close()
            counter += 1

        # save best model: here we save the model only for the lowest validation loss
        if valid_loss < best_loss and gpu_id == 0:
            # Save model parameters
            torch.save({'model': model.state_dict()}, 'resnetmodel-centralizedcode15.pth')
            # Update best validation loss
            best_loss = valid_loss
            # statement
            model_save_state = "Best model -> saved"
        else:
            model_save_state = ""
        
        # Update learning rate with lr-scheduler
        if lr_scheduler:
            print("scheduler updated lr ...")
            lr_scheduler.step()

        # Print message
        tqdm.write('Epoch {epoch:2d}: \t'
                    'Train Loss {train_loss:.6f} \t'
                    'Valid Loss {valid_loss:.6f} \t'
                    '{model_save}'
                    .format(epoch=epoch,
                            train_loss=train_loss,
                            valid_loss=valid_loss,
                            model_save=model_save_state))
    destroy_process_group()
    # =======================================================================#

if __name__ == "__main__":
    main(num_rounds=8, num_chunks=15)
