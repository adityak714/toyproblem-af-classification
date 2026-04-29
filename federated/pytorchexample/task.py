"""Implement the neural network models and training functions."""
from tqdm import tqdm
from typing import List, Tuple
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import torch, h5py, glob, time, argparse, os
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.nn.parameter import Parameter
from torch.optim import SGD, Optimizer, AdamW
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split, ConcatDataset
from flwr.common import Scalar, Context
from typing import Callable, Dict, List, OrderedDict, Union, Optional, Tuple
from pytorchexample.resnet import ResNet1d
import ray.train.torch
import ray.train

fds = None  # Cache FederatedDataset

class ScaffoldOptimizer(SGD):
    """Implements SGD optimizer step function as defined in the SCAFFOLD paper."""

    def __init__(self, grads, step_size, momentum, weight_decay):
        super().__init__(
            grads, lr=step_size, momentum=momentum, weight_decay=weight_decay
        )

    def step_custom(self, server_cv, client_cv):
        """Implement the custom step function fo SCAFFOLD."""
        # y_i = y_i - \eta * (g_i + c - c_i)  -->
        # y_i = y_i - \eta*(g_i + \mu*b_{t}) - \eta*(c - c_i)
        self.step()
        for group in self.param_groups:
            for par, s_cv, c_cv in zip(group["params"], server_cv, client_cv):
                par.data.add_(s_cv - c_cv, alpha=-group["lr"])

def load_centralized_dataset():
    """Load entire test set (selected to be exams_part0, exams_part1, 2 and 3) and return the dataloader."""
    vloaders = []
    device = torch.device("cuda:0")
    
    for i, filepath in enumerate(sorted(glob.glob("../data/code15-12l/*.hdf5"))):
        # build data loaders
        if filepath.replace("../data/code15-12l/", "") in ["exams_part0.hdf5", "exams_part1.hdf5", "exams_part2.hdf5", "exams_part3.hdf5"]:
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
    trainloaders, testloader = [], []
    train_list = sorted(glob.glob("../data/code15-12l/*.hdf5"))

    trains = {
        "features": [],
        "labels": []
    }
    #device = torch.device("cuda")
    #print(os.getcwd())
    for file_ in train_list:
        if file_.replace("../data/code15-12l/", "") in ["exams_part0.hdf5", "exams_part1.hdf5", "exams_part2.hdf5", "exams_part3.hdf5"]:
            print("*** file removed from train_list", file_.replace("../data/code15-12l/", ""))
            train_list.remove(file_)
    
    #if fds is not None:
    #    return DataLoader(
    #        fds[0][partition_id], 
    #        batch_size=batch_size, 
    #        shuffle=False
    #    ), DataLoader(
    #        fds[1][partition_id], 
    #        batch_size=batch_size, 
    #        shuffle=False
    #    )

    for i, filepath in enumerate(train_list):
        path_to_h5_train, path_to_csv_train = filepath, '../data/code15-12l/exams.csv' 
        #print("path_to_h5_train:", path_to_h5_train, "path_to_csv", path_to_csv_train)

        # load traces
        f = h5py.File(path_to_h5_train, 'r')
        traces = torch.tensor(np.array(f['tracings'][()], dtype=np.float32), dtype=torch.float32)[:-1,:,:]
        ids_traces = np.array(f['exam_id'])
        print("traces successfully converted to tensors ...")

        # load labels
        df = pd.read_csv(path_to_csv_train)
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

    # partition the data -- courtesy: https://flower.ai/docs/baselines/niid_bench.html
    if partitioning == "dirichlet":
        alpha = val
        min_required_samples_per_client = 1500

        prng = np.random.default_rng(seed)

        # get the targets
        tmp_t = [item[1].cpu() for item in trainset] # rem_trainset.dataset.targets

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

        upper_size = int((len(tmp_t))/num_partitions)
        
        per_label_cutoff = 12000 # int(0.4*total_samples/num_classes)
        print("per_label_cutoff", per_label_cutoff)
        
        all_classes = [[] for _ in range(num_classes)]
        for i, class_ in enumerate(classes):
            for j, label in enumerate(targets):
                if label in class_:
                    all_classes[i].append(j)
            prng.shuffle(all_classes[i])






        
        idx_clients = np.ones((num_partitions, num_classes), dtype=np.int32)
        for i in range(num_partitions):
            idx_clients[i] = idx_clients[i]*upper_size
            idx_clients[i] = idx_clients[i]*prng.dirichlet(np.repeat(alpha, num_classes))
            deficit = upper_size - np.sum(idx_clients[i])
            idx_clients[i, j % num_classes] += int(deficit/num_classes)
        
        for j in range(num_classes):
            if np.sum(idx_clients[:,j]) > per_label_cutoff:
                deficit = int(np.sum(idx_clients[:,j]) - per_label_cutoff) 
                reduc_factor = (per_label_cutoff/np.sum(idx_clients[:,j]))
                idx_clients[:,j] = idx_clients[:,j]*(reduc_factor)

        max_lim = np.max(np.sum(idx_clients, axis=1))
        for i in range(num_partitions):
            if np.sum(idx_clients[i,:]) < max_lim:
                deficit = max_lim - np.sum(idx_clients[i,:])
                #for j in range(deficit):
                idx_clients[i, j % num_classes] += int(deficit/num_classes)
        
        #print(counter, "datasets made")
        
        # regulating
        #print("NEW >>>>>>")
        #idx_clients = [np.array(subarr, dtype=np.int32) for subarr in idx_clients]
        #print(np.sum(idx_clients, axis=1))
        #print(np.sum(idx_clients, axis=0))

        target = int(62000/num_partitions) 
        for i in range(num_partitions):
            if idx_clients[i].sum() > target:
                print("size of set for partition: (will be reduced to)", target, "-- ", idx_clients[i].sum())
                surplus = int(idx_clients[i].sum()) - target
                while surplus > 0:
                    pos = np.where(idx_clients[i] > 0)[0]
                    if pos.size == 0:
                        break
                    k = min(surplus, pos.size)
                    idx_clients[i][pos[:k]] -= 1
                    surplus -= k
        
        import sys
        print(np.array(idx_clients, dtype=np.int32))
        proportions = np.cumsum(idx_clients, axis=0, dtype=np.int32)
        print(np.sum(idx_clients, axis=1))
        print("PROPORTIONS\n", proportions)
        clients_sets = [np.split(curr_class, proportions[:,i])[:-1] for i, curr_class in enumerate(all_classes)] 
        # initially in untransposed form --> (num_classes, num_partitions)

        #partitioned_sets = [[] for _ in range(num_partitions)]
        #for item in clients_sets:
        #    print("in each class collection there is", len(item))
        #    for i in range(len(item)):
        #        partitioned_sets[i].append(item[i])
        
        #print(len(partitioned_sets[0]))
        partitioned_sets = [list(i) for i in zip(*clients_sets)] 
        print(len(partitioned_sets)) 
        for i, idxs in enumerate(partitioned_sets):
            partitioned_sets[i] = [x for xs in idxs for x in xs]
            #partitioned_sets[i] = partitioned_sets[i][:4000]
            print(len(partitioned_sets[i]))

        print("clients have following >>>", len(clients_sets), [len(clients_sets[i]) for i in range(len(clients_sets))])
        
        set_per_client = [Subset(trainset, idxs) for idxs in partitioned_sets]
        splits = [random_split(client_set, [0.8, 0.2]) for client_set in set_per_client]
        trainsets_per_client = [set_[0] for set_ in splits]
        testsets_per_client = [set_[1] for set_ in splits] 
        #fds = (trainsets_per_client, testsets_per_client)

        ages = [[] for i in range(num_partitions)]
        for i in range(num_partitions):
            for x,y in set_per_client[i]:
                ages[i].append(y[1].cpu().numpy())
            ages[i] = np.array(ages[i]).flatten()

        import pickle
        with open(f'clients{num_partitions}-dirichl{alpha}.pkl', 'wb') as f:
            pickle.dump(ages, f)

        return DataLoader(
            trainsets_per_client[partition_id], 
            batch_size=batch_size, 
            shuffle=False
        ), DataLoader(
            testsets_per_client[partition_id], 
            batch_size=batch_size, 
            shuffle=False
        )
    elif partitioning == "iid":  # sim_iid_non_iid -> dirichl_iid_non_iid
        similarity = val
        if similarity > 1.0 or similarity < 0.1:
            raise ValueError("Invalid value", val, "for partitioning =", partitioning)

        trainsets_per_client = []

        # for s% similarity sample iid data per client
        s_fraction = int(similarity * len(trains))
        prng = np.random.default_rng(seed)
        iid_idxs = prng.choice(len(trains), s_fraction, replace=False)
        rem_idxs = np.setdiff1d(np.arange(len(trains)), iid_idxs)
        iid_trainset = Subset(trains, iid_idxs) # = trainset[iid_idxs] s*idxs
        rem_trainset = Subset(trains, rem_idxs) # = trainset[rem_idxs] (1-s)*idxs
        # s*idxs + (1-s)*idxs = len(trainset)

        all_ids = np.arange(len(iid_trainset))
        splits = np.array_split(all_ids, num_partitions)
        for i in range(num_partitions):
            c_ids = splits[i]
            d_ids = iid_trainset.indices[c_ids]
            trainsets_per_client.append(Subset(iid_trainset.dataset, d_ids))

        if similarity == 1.0:
            ages = [[] for i in range(num_partitions)]
            for i in range(num_partitions):
                for x,y in trainsets_per_client[i]:
                    ages[i].append(y[1].numpy())
                ages[i] = np.array(ages[i]).flatten()

            import pickle
            with open(f'clients{num_partitions}-sim{similarity}.pkl', 'wb') as f:
                pickle.dump(ages, f)

            return DataLoader(
                trainsets_per_client[partition_id], 
                batch_size=batch_size, 
                shuffle=False
            ), DataLoader(
                testset, 
                batch_size=batch_size, 
                shuffle=False
            )

        tmp_t = [item[1].cpu() for item in rem_trainset] # rem_trainset.dataset.targets
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
            prng.shuffle(idx_k)
            idx_k_split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(num_partitions):
                if class_t in client_classes[j]:
                    act_idx = rem_trainset.indices[idx_k_split[ids]]
                    rem_trainsets_per_client[j].append(
                        Subset(rem_trainset.dataset, act_idx)
                    )
                    ids += 1

        for i in range(num_partitions):
            trainsets_per_client[i] = ConcatDataset([trainsets_per_client[i]] + rem_trainsets_per_client[i])

        ages = [[] for i in range(num_partitions)]
        for i in range(num_partitions):
            for x,y in trainsets_per_client[i]:
                ages[i].append(y[1].numpy())
            ages[i] = np.array(ages[i]).flatten()
        import pickle
        with open(f'clients{num_partitions}-sim{similarity}.pkl', 'wb') as f:
            pickle.dump(ages, f)

        #sys.exit(0)
        
        return DataLoader(
            trainsets_per_client[partition_id], 
            batch_size=batch_size, 
            shuffle=False
        ), DataLoader(
            testset, 
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
    momentum = 0
    weight_decay = 0.01

    #trainloader = ray.train.torch.prepare_data_loader(trainloader)
    print("****** CUDA DEVICES:", torch.cuda.device_count())
    device = torch.device(f"cuda:{partition_id % torch.cuda.device_count()}")
    pos_weight = torch.tensor([61], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay) #,weight_decay=weight_decay)

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model.to(device)
    # model = DDP(model, device_ids=[device])
    count = 0
    trainloader, valloader = load_datasets(partition_id, num_partitions, batch_size, partitioning=partitioning, val=val, device=device)
    for _ in enumerate(trainloader):
        count += 1
    print("****** CLIENT: ", partition_id, "BATCH SIZE:", batch_size, "DATA SIZE:", count)

    loss = 0
    for i in range(epochs):
        #if ray.train.get_context().get_world_size() > 1:
        #    trainloader.sampler.set_epoch(i)
        loss, model = _train_one_epoch(net, device, trainloader, criterion, optimizer, i)
        #with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        #    torch.save(model.module.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
        #    ray.train.report(
        #        {
        #            "loss": loss, 
        #            "epoch": i, 
        #            "partition_id": partition_id,
        #            "trainloader_size": len(trainloader)
        #        },
        #        checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
        #    )
        #if ray.train.get_context().get_world_rank() == 0:
        #    print(metrics)

    # -- END MULTIGPU PROCESS --
    # destroy_process_group()
    return loss, model, count

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
    # trainloader.sampler.set_epoch(epoch)

    net.train()
    for traces, diagnoses in train_pbar:
        traces, diagnoses = traces.to(rank), diagnoses.to(rank)
        for x, y in trainloader:
            x, y = x.to(rank), y[:,0].reshape(-1,1).to(rank)
            pred = net(x)
            #print(pred.get_device(), y.get_device())
            curr_loss = criterion(pred, y)
            curr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += curr_loss.detach().cpu().numpy()
        n_entries += len(traces)
        
        train_pbar.set_postfix({'loss': total_loss/n_entries})
    train_pbar.close()

    return float(total_loss/n_entries), net

def test(net, testloader: DataLoader, device) -> Tuple[float, float]: # == rank >> before: torch.device, # world_size=torch.cuda.device_count(): int
    """Evaluate the network on the test set.""" 
    net.to(device)

    net.eval()
    criterion = nn.BCEWithLogitsLoss() # nn.CrossEntropyLoss(reduction="sum")
    sigmoid = nn.Sigmoid().to(device)

    avg_precisions = []  # avg precision (pr-auc)
    loss = 0.0

    with torch.no_grad():
        for data, target in testloader:
            #assert not isinstance(data, str), "FAULTY DATALOADER ... Check your data loading."
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

def train_scaffold(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    server_cv: torch.Tensor,
    client_cv: torch.Tensor,
) -> None:
    # pylint: disable=too-many-arguments
    """Train the network on the training set using SCAFFOLD.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The training set dataloader object.
    device : torch.device
        The device on which to train the network.
    epochs : int
        The number of epochs to train the network.
    learning_rate : float
        The learning rate.
    momentum : float
        The momentum for SGD optimizer.
    weight_decay : float
        The weight decay for SGD optimizer.
    server_cv : torch.Tensor
        The server's control variate.
    client_cv : torch.Tensor
        The client's control variate.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = ScaffoldOptimizer(
        net.parameters(), learning_rate, momentum, weight_decay
    )
    net.train()
    for _ in range(epochs):
        net = _train_one_epoch_scaffold(
            net, trainloader, device, criterion, optimizer, server_cv, client_cv
        )

def _train_one_epoch_scaffold(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: ScaffoldOptimizer,
    server_cv: torch.Tensor,
    client_cv: torch.Tensor,
) -> nn.Module:
    # pylint: disable=too-many-arguments
    """Train the network on the training set for one epoch."""
    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step_custom(server_cv, client_cv)
    return net
