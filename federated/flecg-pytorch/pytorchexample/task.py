"""Implement the neural network models and training functions."""
from tqdm import tqdm
from typing import List, Tuple
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import torch, h5py, glob, time
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.nn.parameter import Parameter
from torch.optim import SGD, Optimizer, Adam
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split, ConcatDataset
from flwr.common import Scalar, Context
from typing import Callable, Dict, List, OrderedDict, Union, Optional, Tuple

from pytorchexample.resnet import ResNet1d

fds = None  # Cache FederatedDataset

def load_centralized_dataset():
    """Load entire test set (selected to be exams_part0, exams_part1, 2 and 3) and return the dataloader."""
    vloaders = []
    for i, filepath in enumerate(sorted(glob.glob("../../data/code15-12l/*.hdf5"))):
        # build data loaders
        if filepath.replace("../../data/code15-12l/", "") in ["exams_part0.hdf5", "exams_part1.hdf5", "exams_part2.hdf5", "exams_part3.hdf5"]:
            path_to_h5_train, path_to_csv_train = filepath, '../../data/code15-12l/exams.csv'
            # load traces
            f = h5py.File(path_to_h5_train, 'r')
            traces = torch.tensor(np.array(f['tracings'], dtype=np.float32), dtype=torch.float32)[:-1,:,:]
            # load labels
            ids_traces = np.array(f['exam_id'])
            df = pd.read_csv(path_to_csv_train)
            #df = df.drop_duplicates(subset=["patient_id"])
            f.close()
            df = df.set_index('exam_id')
            df = df.reindex(ids_traces).dropna(subset=["AF"]) # make sure the order is the same
            labels = torch.tensor(
                np.array(df['AF'], dtype=np.float16), 
                dtype=torch.float16
                #, device=gpu_id
            ).reshape(-1,1)
            print("\nat", i, ">> number of pos. examples >>", len(df[df['AF']==1]), "->> weight >>", len(df[df['AF']==0])/len(df[df['AF']==1])) 
            # load dataset
            dataset = TensorDataset(traces, labels)
            vloaders.append(dataset)
            print("at", filepath, " >> put in validation!")

    vset = torch.utils.data.ConcatDataset(vloaders)
    return DataLoader(vset, 
        batch_size=256, 
        shuffle=False, 
        #sampler=DistributedSampler(vset)
    )

# def load_data(partition_id: int, num_partitions: int, batch_size: int):
def load_datasets(partition_id: int, num_partitions: int, batch_size: int, partitioning: str = "iid", seed: Optional[int] = 42) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    trainloaders, testloader = [], []

    print("Loading hdf5 ...")
    trains = {
        "features": [],
        "labels": []
    }
    for i, filepath in enumerate(glob.glob("../../data/code15-12l/*.hdf5")[:2]):
        prefix = filepath.replace("data/code15-12l/", "").replace(".hdf5", "")
        path_to_h5_train, path_to_csv_train = filepath, '../../data/code15-12l/exams.csv' 
        print("path_to_h5_train:", path_to_h5_train, "path_to_csv", path_to_csv_train)

        # load traces
        f = h5py.File(path_to_h5_train, 'r')
        traces = np.array(f['tracings'][()], dtype=np.float16)[:-1,:,:]
        print("traces successfully converted to tensors ...")

        # load labels
        df = pd.read_csv(path_to_csv_train)
        df.set_index('exam_id', inplace=True)
        df = df.reindex(np.array(f['exam_id'])).dropna(subset=["AF"]) # make sure the order is the same
        labels = np.array(df[['AF', 'age']], dtype=np.float32).reshape(-1,2) 
        
        if len(trains["features"]) > 0:
            trains["features"] = np.vstack((trains["features"], traces))
            trains["labels"] = np.vstack((trains["labels"], labels))
            print("VSTACK DONE >>", len(trains["features"]), len(trains["labels"]))
        else:
            trains["features"] = traces
            trains["labels"] = labels

    trainset = TensorDataset(torch.tensor(trains["features"], dtype=torch.float32), torch.tensor(trains["labels"], dtype=torch.float32)) # TODO: This will be very memory intensive.
    trains, testset = random_split(trainset, [0.8, 0.2])
    #print("Dataset prepared with_format('torch') >>", trains)

    # partition the data -- courtesy: https://flower.ai/docs/baselines/niid_bench.html
    if partitioning == "dirichlet":
        alpha = 0.5
        min_required_samples_per_client = 10
        min_samples = 0
        prng = np.random.default_rng(seed)

        # get the targets
        tmp_t = [y for x,y in trains.dataset] # rem_trainset.dataset.targets
        if isinstance(tmp_t, list):
            tmp_t = np.array(tmp_t)
        if isinstance(tmp_t, torch.Tensor):
            tmp_t = tmp_t.numpy()
        print("PARTITIONING NOW >>", partitioning, tmp_t)
        targets = tmp_t[:,1].flatten()
        print("FLATTENED", targets)
        num_classes = len(set(targets))
        total_samples = len(targets)

        while min_samples < min_required_samples_per_client:
            idx_clients: List[List] = [[] for _ in range(num_partitions)]
            for k in range(num_classes):
                idx_k = np.where(targets == k)[0]
                prng.shuffle(idx_k)
                proportions = prng.dirichlet(np.repeat(alpha, num_partitions))
                proportions = np.array(
                    [
                        p * (len(idx_j) < total_samples / num_partitions)
                        for p, idx_j in zip(proportions, idx_clients)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_k_split = np.split(idx_k, proportions)
                idx_clients = [
                    idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
                ]
                min_samples = min([len(idx_j) for idx_j in idx_clients])

        trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
        return DataLoader(trainsets_per_client[partition_id], batch_size=batch_size, shuffle=True, num_workers=4), DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    elif partitioning == "iid":  # sim_iid_non_iid -> dirichl_iid_non_iid
        similarity = 0.6
        trainsets_per_client = []

        # for s% similarity sample iid data per client
        s_fraction = int(similarity * len(trains))
        prng = np.random.default_rng(seed)
        iid_idxs = prng.choice(len(trains), s_fraction, replace=False)
        rem_idxs = np.setdiff1d(np.arange(len(trains)), iid_idxs)
        iid_trainset = Subset(trains, iid_idxs) # = trainset[iid_idxs] s*idxs
        rem_trainset = Subset(trains, rem_idxs) # = trainset[rem_idxs] (1-s)*idxs
        # s*idxs + (1-s)*idxs = len(trainset)

        # sample iid data per client from iid_trainset
        all_ids = np.arange(len(iid_trainset))
        splits = np.array_split(all_ids, num_partitions)
        for i in range(num_partitions):
            c_ids = splits[i]
            d_ids = iid_trainset.indices[c_ids]
            trainsets_per_client.append(Subset(iid_trainset.dataset, d_ids))

        if similarity == 1.0:
            return DataLoader(trainsets_per_client[partition_id], batch_size=batch_size, shuffle=True, num_workers=4), DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

        tmp_t = [y for x,y in rem_trainset.dataset] # rem_trainset.dataset.targets
        if isinstance(tmp_t, list):
            tmp_t = np.array(tmp_t)
        if isinstance(tmp_t, torch.Tensor):
            tmp_t = tmp_t.numpy()

        targets = tmp_t[rem_trainset.indices,1].flatten() # shape: 1d (N,)
        bins = pd.cut(targets, [10, 20, 30, 40, 50, 60, 70, 80])
        remaining_classes = list(set(bins))
        remaining_classes.remove(np.nan)
        num_remaining_classes = len(remaining_classes) # should be 7 ... 10-20, 20-30, ... 70-80 (7 age groups)

        client_classes: List[List] = [[] for _ in range(num_partitions)]
        times = np.zeros(num_remaining_classes)

        for i in range(num_partitions): # partition = client
            client_classes[i] = [remaining_classes[i % num_remaining_classes]]
            times[i % num_remaining_classes] += 1
            j = 1
            while j < 2:
                index = prng.choice(num_remaining_classes)
                class_t = remaining_classes[index]
                if class_t not in client_classes[i]:
                    client_classes[i].append(class_t)
                    times[index] += 1
                    j += 1

        rem_trainsets_per_client: List[List] = [[] for _ in range(num_partitions)]

        for i in range(num_remaining_classes): # data partition not the same as 
            class_t = remaining_classes[i]
            idx_k = [] # this is supposed to be indices, so taking i through enumerate(targets)
            for j, label in enumerate(targets):
                if label in class_t:
                    idx_k.append(j)
            print(times)
            prng.shuffle(idx_k)
            idx_k_split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(num_partitions):
                if class_t in client_classes[j]:
                    act_idx = rem_trainset.indices[idx_k_split[ids]]             #############################################
                    rem_trainsets_per_client[j].append(
                        Subset(rem_trainset.dataset, act_idx)
                    )
                    ids += 1

        for i in range(num_partitions):
            trainsets_per_client[i] = ConcatDataset(
                [trainsets_per_client[i]] + rem_trainsets_per_client[i]
            )

        return DataLoader(trainsets_per_client[partition_id], batch_size=batch_size, shuffle=True, num_workers=4), DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    else: # if any other partitioning strategy given that is not implemented here >>
        raise NotImplementedError

    return trainloaders, testloader


# def train(net, trainloader, epochs, lr, device):
def train_fedavg(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    learning_rate: float,
    momentum=0,
    weight_decay=0,
    device=torch.device("cuda:0"),
) -> None:
    # pylint: disable=too-many-arguments
    """Train the network on the training set using FedAvg."""
    count = 0
    for x, y in trainloader:
        count += 1
    print(count, epochs, learning_rate)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(net.parameters(), lr=learning_rate) #,weight_decay=weight_decay)

    loss = 0
    for i in range(epochs):
        loss, net = _train_one_epoch(net, device, trainloader, criterion, optimizer, i)

    return loss, net

def _train_one_epoch(
    net,
    rank, # torch.device
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    epoch: int
) -> nn.Module:
    """Train the network on the training set for one epoch."""
    net.to(rank)
    tqdm.write("Training model...")
    train_pbar = tqdm(trainloader, desc="Training Epoch {epoch:2d}".format(epoch=epoch), leave=True)
    total_loss, n_entries = 0, 0
    
    net.train()
    for traces, diagnoses in train_pbar:
        traces, diagnoses = traces.to(rank), diagnoses.to(rank)
        for x, y in trainloader:
            assert not isinstance(x, str), "FAULTY DATALOADER ... Check your data loading."
            x, y = x.to(rank), y.to(rank)
            pred = net(x)
            curr_loss = criterion(pred, y)
            curr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += curr_loss.detach().cpu().numpy()
        n_entries += len(traces)
        
        train_pbar.set_postfix({'loss': total_loss / n_entries})
    train_pbar.close()

    return float(total_loss/n_entries), net

# def test(net, testloader, device):
def test(net, testloader: DataLoader, device,) -> Tuple[float, float]: # == rank >> before: torch.device, # world_size=torch.cuda.device_count(): int
    """Evaluate the network on the test set.""" 
    net.to(device)
    net.eval()
    criterion = nn.BCEWithLogitsLoss() # nn.CrossEntropyLoss(reduction="sum")
    sigmoid = nn.Sigmoid().to(device)

    avg_precisions = []  # avg precision (pr-auc)
    loss = 0.0

    with torch.no_grad():
        for data, target in testloader:
            assert not isinstance(data, str), "FAULTY DATALOADER ... Check your data loading."
            data, target = data.to(device), target[:,0].reshape(-1,1).to(device)
            output = net(data)
            loss += criterion(output, target).item()
            if len(np.unique(target.cpu())) == 2: 
                # if both positive and negative truth values are present, 
                # compute the avg. precision
                avg_precisions.append(
                    average_precision_score(target.cpu(), sigmoid(output).cpu())
                )

    loss = loss / len(testloader)
    ap = np.mean(avg_precisions)
    return loss, ap
