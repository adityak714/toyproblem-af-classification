#import os, glob
#import numpy as np
#import torch
#from torchvision import datasets, transforms
#from typing import Callable, Dict, List, OrderedDict, Union, Optional, Tuple
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

def generate_distributed_datasets(k: int, alpha: float, save_dir: str) -> None:

    np.random.seed(42)
    torch.manual_seed(42)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    full_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    os.makedirs(save_dir, exist_ok=True)
    labels = full_dataset.targets.numpy()
    num_classes = len(full_dataset.classes)
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    client_indices = [[] for _ in range(k)]
    
    for class_idx in range(num_classes):
        proportions = np.random.dirichlet(np.repeat(alpha, k))
        indices = class_indices[class_idx]
        np.random.shuffle(indices)
        splits = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        class_splits = np.split(indices, splits)
        
        for client_idx in range(k):
            client_indices[client_idx].extend(class_splits[client_idx])
    
    for client_idx in range(k):
        torch.save({
            'indices': client_indices[client_idx],
            'targets': labels[client_indices[client_idx]]
        }, os.path.join(save_dir, f'client_{client_idx}.pt'))


"""
def load_client_data(cid: int, data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    torch.manual_seed(42)
    full_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transforms.ToTensor()
    )
    client_data = torch.load(os.path.join(data_dir, f'client_{cid}.pt'), weights_only=False)
    client_dataset = Subset(full_dataset, client_data['indices'])
    
    train_size = int(0.8 * len(client_dataset))
    val_size = len(client_dataset) - train_size
    train_dataset, val_dataset = random_split(
        client_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset.dataset.transform = transform
    val_dataset.dataset.transform = transform
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size)
    )
"""

def load_datasets(cid: int, batch_size: int, 
                  num_partitions: int = 10, 
                  partitioning: str = "dirichlet", 
                  val: float = 1.0, 
                  device: torch.device = torch.device("cpu"), 
                  seed: Optional[int] = 42
                 ) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainloaders, testloader = [], []
    train_list = sorted(glob.glob("../data/code15-12l/*.hdf5"))

    trains = {
        "features": [],
        "labels": []
    }

    if os.path.exists("../../new_dataset.pt"):
        print("Pre-loading training dataset ...")
        loaded_tensors = torch.load("../../new_dataset.pt")
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
            trainsets_per_client[cid], 
            batch_size=batch_size, 
            shuffle=False
        ), DataLoader(
            testsets_per_client[cid], 
            batch_size=batch_size, 
            shuffle=False
        )
    else: # if any other partitioning strategy given that is not implemented here >>
        raise NotImplementedError

    return trainloaders, testloader