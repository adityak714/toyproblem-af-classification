"""Implement the neural network models and training functions."""
from tqdm import tqdm
from typing import List, Tuple
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch, h5py, glob, time, argparse, os
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.nn.parameter import Parameter
from torch.optim import SGD, Optimizer, Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split, ConcatDataset
from flwr.common import Scalar, Context
from typing import Callable, Dict, List, OrderedDict, Union, Optional, Tuple
from pytorchexample.resnet import ResNet1d
from pytorchexample.dataloader import BatchDataloader

fds = None  # Cache FederatedDataset

class ScaffoldOptimizer(SGD):
    """Implements SGD optimizer step function as defined in the SCAFFOLD paper."""

    def __init__(self, params, lr, momentum=0, weight_decay=0):
        # Base optimizer set to standard SGD as derived mathematically in SCAFFOLD
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def step_custom(self, server_cv, client_cv):
        """Implement the custom step function for SCAFFOLD."""
        # Standard gradient step first
        self.step()
        # Corrected: y_i = y_i - eta * (c - c_i)
        for group in self.param_groups:
            for par, s_cv, c_cv in zip(group["params"], server_cv, client_cv):
                if par.requires_grad:
                    # Subtract the control variate difference using the group learning rate
                    par.data.add_(s_cv.to(par.device) - c_cv.to(par.device), alpha=-group["lr"])

"""
class ScaffoldOptimizer(Adam):
    "Implements SGD optimizer step function as defined in the SCAFFOLD paper."

    def __init__(self, grads, step_size, momentum, weight_decay):
        super().__init__(
            grads, lr=step_size, 
            #momentum=momentum, 
            weight_decay=weight_decay
        )

    def step_custom(self, server_cv, client_cv):
        "Implement the custom step function fo SCAFFOLD."
        # y_i = y_i - \eta * (g_i + c - c_i)  -->
        # y_i = y_i - \eta*(g_i + \mu*b_{t}) - \eta*(c - c_i)
        self.step()
        for group in self.param_groups:
            for par, s_cv, c_cv in zip(group["params"], server_cv, client_cv):
                print("parameter vs. s_cv - c_cv", par.grad.norm(), (s_cv.to(par.device) - c_cv.to(par.device)).norm())
                par.data.add_(s_cv.to(par.device) - c_cv.to(par.device), alpha=-group["lr"])
"""

def load_centralized_dataset():
    """Load entire test set (selected to be exams_part0, exams_part1, 2 and 3) and return the dataloader.""" 
    vloaders = []
    device = torch.device("cuda:0")
    
    """
    path_to_h5_train, path_to_csv_train = '../data/VDS_ECG_Dataset.h5', '../data/code15-12l/exams.csv' 
    
    f = h5py.File(path_to_h5_train, 'r')
    traces = f['tracings']
    ids_traces = f['exam_id'][:]
    print("traces successfully converted to tensors ...", type(ids_traces))

    # load labels
    df = pd.read_csv(path_to_csv_train) # "exams.csv"
    df = df.set_index('exam_id')
    df = df.drop_duplicates(subset=["patient_id"], keep='last')[df.trace_file.isin(["exams_part0.hdf5","exams_part1.hdf5","exams_part2.hdf5","exams_part3.hdf5"])]
    #df = df.reindex(ids_traces).dropna(subset=["AF"]) # make sure the order is the same

    vds_map = pd.DataFrame({
        'exam_id': ids_traces,
        'vds_idx': np.arange(len(ids_traces))
    })
    # It maintains the order of the VDS because 'vds_map' is the left table.
    aligned_data = pd.merge(vds_map, df, on='exam_id', how='inner')
    labels = np.array(aligned_data[['AF', 'age']], dtype=np.float32).reshape(-1,2)
    
    global_labels = np.zeros((len(ids_traces), 2), dtype=np.float32)
    vds_indices = aligned_data['vds_idx'].values
    global_labels[vds_indices] = labels
    
    partition_mask = np.zeros(len(ids_traces), dtype=bool)
    samples = aligned_data['vds_idx'].values
    partition_mask[samples] = True
    assert np.all(ids_traces[vds_indices] == aligned_data['exam_id'].values)

    return BatchDataloader(traces, global_labels, bs=128, mask=partition_mask)
    """

    for i, filepath in enumerate(sorted(glob.glob("../data/code15-12l/*.hdf5"))):
        # build data loaders
        if filepath.replace("../data/code15-12l/", "") in [
                "exams_part0.hdf5", "exams_part1.hdf5"#, "exams_part2.hdf5", "exams_part3.hdf5"
            ]:
            path_to_h5_train, path_to_csv_train = filepath, '../data/code15-12l/exams.csv'
            # load traces
            f = h5py.File(path_to_h5_train, 'r')
            traces = torch.tensor(np.array(f['tracings'], dtype=np.float32), dtype=torch.float32, 
                                  #device=device
                                 )[:-1,:,:]
            # load labels
            ids_traces = np.array(f['exam_id'])
            df = pd.read_csv(path_to_csv_train)
            f.close()
            df = df.set_index('exam_id')
            df = df.drop_duplicates(subset=["patient_id"], keep='last')
            df = df.reindex(ids_traces).dropna(subset=["AF"]) # make sure the order is the same
            labels = torch.tensor(
                np.array(df['AF'], dtype=np.float32), 
                dtype=torch.float32,
                #device=device
            ).reshape(-1,1)
            traces = torch.index_select(traces, 0, torch.tensor(np.isin(ids_traces, np.array(df.index)).nonzero()[0], dtype=torch.int32)) #[:labels.shape[0],:,:]
            
            # load dataset
            dataset = TensorDataset(traces, labels)
            vloaders.append(dataset)
            print("at", filepath, " >> put in validation!")

    vset = ConcatDataset(vloaders)
    return DataLoader(vset, 
        batch_size=256, 
        shuffle=False, 
        #sampler=DistributedSampler(vset)
    )

# def load_data(partition_id: int, num_partitions: int, batch_size: int):
def load_datasets(partition_id: int, num_partitions: int, batch_size: int, partitioning: str = "iid", val: float = 1.0, device: torch.device = torch.device("cpu"), seed: Optional[int] = 42) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    np.random.seed(seed)
    trainloaders, testloader = [], []
    train_list = sorted(glob.glob("../data/code15-12l/*.hdf5"))

    trains = {
        "features": [],
        "labels": []
    }

    """
    path_to_h5_train, path_to_csv_train = '../data/VDS_ECG_Dataset.h5', '../data/code15-12l/exams.csv' 

    # load traces
    f = h5py.File(path_to_h5_train, 'r')
    traces = f['tracings']
    ids_traces = f['exam_id'][:]
    print("traces successfully converted to tensors ...", type(ids_traces))

    # load labels
    df = pd.read_csv(path_to_csv_train) # "exams.csv"
    df = df.set_index('exam_id')
    df = df.drop_duplicates(subset=["patient_id"], keep='last')[~df.trace_file.isin(["exams_part0.hdf5","exams_part1.hdf5","exams_part2.hdf5","exams_part3.hdf5"])]
    #df = df.reindex(ids_traces).dropna(subset=["AF"]) # make sure the order is the same

    vds_map = pd.DataFrame({
        'exam_id': ids_traces,
        'vds_idx': np.arange(len(ids_traces))
    })
    # It maintains the order of the VDS because 'vds_map' is the left table.
    aligned_data = pd.merge(vds_map, df, on='exam_id', how='inner')
    labels = np.array(aligned_data[['AF', 'age']], dtype=np.float32).reshape(-1,2)
    
    global_labels = np.zeros((len(ids_traces), 2), dtype=np.float32)
    vds_indices = aligned_data['vds_idx'].values
    global_labels[vds_indices] = labels
    
    """
    if os.path.exists("new_dataset.pt"):
        loaded_tensors = torch.load('new_dataset.pt')
        trainset = TensorDataset(*loaded_tensors)
    else:
        for i, filepath in enumerate(train_list):
            if filepath.replace("../data/code15-12l/", "") not in [
                    "exams_part0.hdf5", "exams_part1.hdf5", #"exams_part2.hdf5", "exams_part3.hdf5"
                ]:
                path_to_h5_train, path_to_csv_train = filepath, '../data/code15-12l/exams.csv' 

                # load traces
                f = h5py.File(path_to_h5_train, 'r')
                traces = torch.tensor(np.array(f['tracings'][()], dtype=np.float32), dtype=torch.float32)[:-1,:,:]
                ids_traces = np.array(f['exam_id'])
                #print("traces successfully converted to tensors ...")

                # load labels
                df = pd.read_csv(path_to_csv_train) # "exams.csv"
                df = df.set_index('exam_id')
                df = df.drop_duplicates(subset=["patient_id"], keep='last')
                df = df.reindex(ids_traces).dropna(subset=["AF"]) # make sure the order is the same
        
                labels = np.array(df[['AF', 'age']], dtype=np.float32).reshape(-1,2)
                traces = torch.index_select(traces, 0, torch.tensor(np.isin(ids_traces, np.array(df.index)).nonzero()[0], dtype=torch.int32))  # [:labels.shape[0],:,:]    
    
                if len(trains["features"]) > 0:
                    trains["features"] = np.vstack((trains["features"], traces.detach().cpu()))
                    trains["labels"] = np.vstack((trains["labels"], labels))
                    print("VSTACK DONE >>", len(trains["features"]), len(trains["labels"]))
                else:
                    trains["features"] = traces.detach().cpu()
                    trains["labels"] = labels
    
        trainset = TensorDataset(
            torch.tensor(trains["features"], dtype=torch.float32),
            torch.tensor(trains["labels"], dtype=torch.float32)
        )
 
        torch.save(trainset.tensors, 'new_dataset.pt')

    # partition the data -- modified impl originally from: https://flower.ai/docs/baselines/niid_bench.html
    if partitioning == "dirichlet":
        alpha = val
        #min_required_samples_per_client = 1500
        prng = np.random.default_rng(seed)
        # get the targets
        tmp_t = [item[1].cpu() for item in trainset]
        # tmp_t = labels
    
        if isinstance(tmp_t, list):
            tmp_t = np.array(tmp_t)
        if isinstance(tmp_t, torch.Tensor):
            tmp_t = tmp_t.numpy()
        
        targets = tmp_t[:,1].flatten()
        bins = pd.cut(targets, [10, 20, 30, 40, 50, 60, 70, 80, 90]) # 100])
        
        print(bins.value_counts())
        print("..................")
    
        classes = list(set(bins))
        if np.nan in classes:
            classes.remove(np.nan)
        classes = sorted(classes)
        num_classes = len(classes)
        total_samples = len(targets)
    
        ### GOAL: adjusting client sizes to be as equal as possible
        # the smallest age bin has around 12,000 values, hence we want at least that many samples uniformly per client
        # ---> smallest age bin * num_partitions = 12,000 (use a small safety margin, so multiply this by 0.7 or equiv.)
        # NEW: customized per label cutoffs to minimize loss of data points for training
        target = int(160000/num_partitions)
        per_label_cutoff = [11500, 29700, 29700, 29700, 29700, 29700, 29700, 14500] # basically same as target based on the distribution ########## int(0.4*total_samples/num_classes)
        print("per_label_cutoff", per_label_cutoff)
        
        all_classes = [[] for _ in range(num_classes)]
        for i, class_ in enumerate(classes):
            for j, label in enumerate(targets):
                if label in class_:
                    all_classes[i].append(j)
            prng.shuffle(all_classes[i])
        
        idx_clients = np.ones((num_partitions, num_classes), dtype=np.int32)
        for i in range(num_classes):
            idx_clients[:,i] = idx_clients[:,i] * target
            idx_clients[:,i] = idx_clients[:,i] * prng.dirichlet(np.repeat(alpha, num_partitions))
    
        for i in range(num_partitions):
            n = idx_clients[i].size
            if idx_clients[i].sum() > target:
                print("size of partition: (will be decreased to)", target, "-- ", idx_clients[i].sum())
                surplus = int(idx_clients[i].sum()) - target
                while surplus > 0:
                    pos = np.where(idx_clients[i] > 0)[0]
                    if pos.size == 0:
                        break
                    k = min(surplus, pos.size)
                    idx_clients[i][pos[:k]] -= 1
                    surplus -= k
            elif idx_clients[i].sum() < target:
                print("size of partition: (will be increased to)", target, "-- ", idx_clients[i].sum())
                deficit = target - int(idx_clients[i].sum())
                while deficit > 0:
                    pos = np.where((idx_clients[i] > 0))[0]
                    if pos.size == 0:
                        break
                    k = min(deficit, pos.size)
                    idx_clients[i][pos[:k]] += 1 # if idx_clients[i][pos[j]] + 1 <= per_label_cutoff else 0
                    deficit -= k # if idx_clients[i][pos[j]] <= per_label_cutoff else 0
    
        for i in range(num_classes):
            while idx_clients[:, i].sum() > per_label_cutoff[i]:
                pos = np.where(idx_clients[:,i] > 1)[0]
                if pos.size == 0:
                    break
                idx_clients[pos, i] -= 1
            while idx_clients[:, i].sum() < per_label_cutoff[i]:
                pos = np.where((idx_clients[:,i] > 0) & (idx_clients[:,i] < per_label_cutoff[i]))[0]
                if pos.size == 0:
                    break
                idx_clients[pos, i] += 1
    
        for i in range(num_partitions):
            for j in range(num_classes):
                if idx_clients[i,j] > 0 and np.sum(idx_clients[i,:]) < target:
                    idx_clients[i,j] += 1
                elif idx_clients[i,j] > 0 and np.sum(idx_clients[i,:]) > target:
                    idx_clients[i,j] -= 1

        for i in range(num_partitions):
            n = idx_clients[i].size
            if idx_clients[i].sum() > target:
                print("size of partition: (will be decreased to)", target, "-- ", idx_clients[i].sum())
                surplus = int(idx_clients[i].sum()) - target
                while surplus > 0:
                    pos = np.where(idx_clients[i] > 0)[0]
                    if pos.size == 0:
                        break
                    k = min(surplus, pos.size)
                    idx_clients[i][pos[:k]] -= 1
                    surplus -= k
        
        proportions = np.cumsum(idx_clients, axis=0, dtype=np.int32)    
        clients_sets = [np.split(curr_class, proportions[:,i])[:-1] for i, curr_class in enumerate(all_classes)] 
    
        partitioned_sets = [list(i) for i in zip(*clients_sets)] 
        print(len(partitioned_sets)) 
        for i, idxs in enumerate(partitioned_sets):
            partitioned_sets[i] = [x for xs in idxs for x in xs]
            print("client", i, "has", len(partitioned_sets[i]), "samples")
    
        # the proportions are used to do the cumsum along the clients' axis, so that we can split the data of each class according to the proportions to allocate each client.
       
        """
        splits = [train_test_split(idxs, test_size=0.2, random_state=seed) for idxs in partitioned_sets]
        set_per_client = []
        for idxs in splits:
            train_val = []
            # train
            partition_mask = np.zeros(len(ids_traces), dtype=bool)
            samples = aligned_data.iloc[idxs[0]]['vds_idx'].values
            partition_mask[samples] = True
            train_val.append(BatchDataloader(traces, global_labels, bs=128, mask=partition_mask))
            # val
            partition_mask = np.zeros(len(ids_traces), dtype=bool)
            samples = aligned_data.iloc[idxs[1]]['vds_idx'].values
            partition_mask[samples] = True
            train_val.append(BatchDataloader(traces, global_labels, bs=128, mask=partition_mask))
            set_per_client.append(train_val)
          
        for i, partition in enumerate(set_per_client):
            # positives, total = 0, 0 
            labels = partition[0].tensors[1]
            mask = partition[0].mask
            positives = int(labels[mask, 0].sum())
            total = int(mask.sum())
            print(f"Partition {i}: positives={positives}, total={total}")

        return set_per_client[partition_id][0], set_per_client[partition_id][1]

        """
        set_per_client = [Subset(trainset, idxs) for idxs in partitioned_sets]
        splits = [random_split(client_set, [0.8, 0.2]) for client_set in set_per_client]
        trainsets_per_client = [set_[0] for set_ in splits]
        testsets_per_client = [set_[1] for set_ in splits]

        ages = [[] for i in range(num_partitions)]
        for i in range(num_partitions):
            for x,y in set_per_client[i]:
                ages[i].append(y[1].cpu().numpy())
            ages[i] = np.array(ages[i]).flatten()
        import pickle
        with open(f'clients{num_partitions}-dirichl{alpha}.pkl', 'wb') as f:
            pickle.dump(ages, f)

        positives = 0
        for i in range(num_partitions):
            for x, y in set_per_client[i]:
                positives += np.sum(y[0].cpu().numpy())
        print("# of positive AF for Dirichlet alpha=", alpha, "--", positives)

        trainsets_per_client = [set_[0] for set_ in splits]
        testsets_per_client = [set_[1] for set_ in splits] 

        return DataLoader(
            trainsets_per_client[partition_id], 
            batch_size=batch_size, 
            shuffle=False
        ), DataLoader(
            testsets_per_client[partition_id], 
            batch_size=batch_size, 
            shuffle=False
        )
    else: # if any other partitioning strategy given that is not implemented here >>
        raise NotImplementedError

    return trainloaders, testloader


# def train(net, trainloader, epochs, lr, device):
def train_fedavg(config) -> None:
    # pylint: disable=too-many-arguments
    """Train the network on the training set using FedAvg."""
    net = config["net"] # ray.train.torch.prepare_model(net)
    partition_id = config["partition_id"]
    num_partitions = config["num_partitions"]
    batch_size = config["batch_size"]
    partitioning = config["partitioning"]
    val = config["val"]
    epochs = config["epochs"]
    learning_rate = config["lr"]
    proximal_mu = config["proxmu"] # will be zero if strategy == "fedavg", handled in client_app.py
    momentum = 0
    weight_decay = 0.01

    #trainloader = ray.train.torch.prepare_data_loader(trainloader)
    print("****** CUDA DEVICES:", torch.cuda.device_count())
    device = torch.device(f"cuda:{partition_id % torch.cuda.device_count()}")
    pos_weight = torch.tensor([65], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay) #,weight_decay=weight_decay)

    count = 0
    trainloader, valloader = load_datasets(partition_id, num_partitions, batch_size, partitioning=partitioning, val=val, device=device)
    for _ in enumerate(trainloader):
        count += 1

    loss = 0
    global_params = [val.detach().clone() for val in net.parameters()] if proximal_mu > 0.0 else None # will be empty if strategy == "fedavg"
    
    for i in range(epochs):
        loss, model = _train_one_epoch(net, device, trainloader, criterion, optimizer, i, proximal_mu=proximal_mu, global_params=global_params)

    return loss, model, count

def _train_one_epoch(
    net,
    rank, # torch.device
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    proximal_mu: float = 0.0, # if this is zero, or not passed, then it is simply FedAvg
    global_params = None
) -> nn.Module:
    """Train the network on the training set for one epoch."""
    net.to(rank)
    tqdm.write("Training model...")

    total_loss, n_entries = 0, 0
    train_pbar = tqdm(initial=0, total=len(trainloader), desc="Training Epoch {epoch:2d} - Loss: {total_loss:.5f}".format(epoch=epoch, total_loss=total_loss), leave=True)
    # trainloader.sampler.set_epoch(epoch)
    if global_params is not None:
        print("#### USING FEDPROX with mu=", proximal_mu, " length of global_params", len(global_params))

    net.train()
    for x, y in trainloader:
        x, y = x.to(rank), y[:,0].reshape(-1,1).to(rank)
        pred = net(x)
        ####### DEPENDING ON IF FEDAVG OR FEDPROX #######
        if global_params is not None: ### FEDPROX
            proximal_term = 0.0
            for local_weights, global_weights in zip(net.parameters(), global_params, strict=True):
                proximal_term += torch.square((local_weights - global_weights.to(local_weights.device)).norm(2))
                
            curr_loss = criterion(pred, y) + (proximal_mu / 2) * proximal_term
        else:                         ### FEDAVG
            curr_loss = criterion(pred, y)
        
        curr_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += curr_loss.detach().cpu().numpy()
        n_entries += 1
        #n_entries += len(traces)

        train_pbar.desc = "Training Epoch {epoch:2d} - Loss: {total_loss:.5f}".format(epoch=epoch, total_loss=total_loss/n_entries)
        train_pbar.update(1)
    train_pbar.close()

    return float(total_loss/n_entries), net

def test(net, testloader, device) -> Tuple[float, float]: # == rank >> before: torch.device, # world_size=torch.cuda.device_count(): int
    """Evaluate the network on the test set.""" 
    net.to(device)

    net.eval()
    pos_weight = torch.tensor([65], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # nn.CrossEntropyLoss(reduction="sum")
    sigmoid = nn.Sigmoid().to(device)

    avg_precisions = []  # avg precision (pr-auc)
    loss = 0.0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in testloader:
            #assert not isinstance(data, str), "FAULTY DATALOADER ... Check your data loading."
            data, target = data.to(device), target[:,0].reshape(-1,1).to(device)
            output = net(data)
            loss += criterion(output, target).item()
            
            all_targets.append(target.cpu())
            all_probs.append(sigmoid(output).cpu())

    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()
            
    loss = loss / len(testloader)
    ap = average_precision_score(all_targets, all_probs)
    return loss, ap

def train_scaffold(config, server_cv: torch.Tensor, client_cv: torch.Tensor) -> None:
    # pylint: disable=too-many-arguments
    """Train the network on the training set using FedAvg."""
    net = config["net"]
    partition_id = config["partition_id"]
    trainloader = config["trainloader"]
    valloader = config["valloader"]
    epochs = config["epochs"]
    learning_rate = config["lr"]
    batch_size = config["batch_size"]
    momentum = 0
    weight_decay = 0.01

    # pylint: disable=too-many-arguments
    print("****** CUDA DEVICES:", torch.cuda.device_count())
    device = torch.device(f"cuda:{partition_id % torch.cuda.device_count()}")
    pos_weight = torch.tensor([65], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = ScaffoldOptimizer(
        net.parameters(), learning_rate, momentum, weight_decay
    )
    
    count = 0
    for _ in enumerate(trainloader):
        count += 1
    print("****** CLIENT: ", partition_id, "BATCH SIZE:", batch_size, "DATA SIZE:", count)

    loss = 0
    for i in range(epochs):
        loss, net = _train_one_epoch_scaffold(
            i, net, trainloader, device, criterion, optimizer, server_cv, client_cv
        )
    return loss, net, count

def _train_one_epoch_scaffold(
    epoch: int,
    net: nn.Module,
    trainloader: DataLoader,
    device,
    criterion: nn.Module,
    optimizer: ScaffoldOptimizer,
    server_cv: torch.Tensor,
    client_cv: torch.Tensor,
) -> nn.Module:
    # pylint: disable=too-many-arguments
    """Train the network on the training set for one epoch."""
    print("In train_one_epoch_scaffold ...")
    net.to(device)
    tqdm.write("Training model...")
    total_loss, n_entries = 0, 0
    train_pbar = tqdm(initial=0, total=len(trainloader), desc="Training Epoch {epoch:2d} - Loss: {total_loss:.5f}".format(epoch=epoch, total_loss=total_loss), leave=True)

    """
    net.train()
    for traces, diagnoses in train_pbar:
        traces, diagnoses = traces.to(device), diagnoses.to(device)
        for x, y in trainloader:
            x, y = x.to(device), y[:,0].reshape(-1,1).to(device)
            optimizer.zero_grad()
            pred = net(x)
            curr_loss = criterion(pred, y)
            curr_loss.backward()
            optimizer.step_custom(server_cv, client_cv)
            
        total_loss += curr_loss.detach().cpu().numpy()
        n_entries += len(x)
        
        train_pbar.set_postfix({'loss': total_loss/n_entries})
    train_pbar.close()
    """

    net.train()
    for x, y in trainloader:
        x, y = x.to(device), y[:,0].reshape(-1,1).to(device)
        optimizer.zero_grad()
        pred = net(x)
        curr_loss = criterion(pred, y)
        curr_loss.backward()
        optimizer.step_custom(server_cv, client_cv)
        
        total_loss += curr_loss.detach().cpu().numpy()
        n_entries += 1
        #n_entries += len(x)

        train_pbar.desc = "Training Epoch {epoch:2d} - Loss: {total_loss:.5f}".format(epoch=epoch, total_loss=total_loss/n_entries)
        train_pbar.update(1)
    train_pbar.close()

    return float(total_loss/n_entries), net
