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

###################################################################
###################################################################
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
    # list(zip([64, 128, 196, 256, 320], [4096, 1024, 256, 64, 16]))
    def __init__(self, n_classes, 
                 input_dim=(12, 4096), 
                 blocks_dim=list(zip([64, 128, 320], [4096, 1024, 16])), 
                 kernel_size=17, 
                 dropout_rate=0.4):
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

###################################################################
###################################################################

fds = None  # Cache FederatedDataset

def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    filepath = "../../data/code15-12l/exams_part3.hdf5"
    path_to_h5_train, path_to_csv_train = filepath, '../../data/code15-12l/exams.csv' # path_to_records = 'data/codesubset/RECORDS.txt'
    print("path_to_h5_train:", path_to_h5_train, "path_to_csv", path_to_csv_train)
    f = h5py.File(path_to_h5_train, 'r')
    traces = torch.tensor(f['tracings'][()], dtype=torch.float32)[:-1,:,:]
    df = pd.read_csv(path_to_csv_train)
    df.set_index('exam_id', inplace=True)
    df = df.reindex(np.array(f['exam_id'])).dropna(subset=["AF"]) # make sure the order is the same
    labels = torch.tensor(np.array(df['AF'], dtype=np.float32), dtype=torch.float32).reshape(-1,1)
    dataset = TensorDataset(traces, labels)
    return DataLoader(dataset, batch_size=128)

# def load_data(partition_id: int, num_partitions: int, batch_size: int):
def load_datasets(partition_id: int, num_partitions: int, batch_size: int, partitioning: str = "iid", seed: Optional[int] = 42) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    trainloaders, testloader = [], []

    print("Loading hdf5 ...")
    trains = {
        "features": [],
        "labels": []
    }
    for i, filepath in enumerate(glob.glob("../../data/code15-12l/*.hdf5")[:5]):
        prefix = filepath.replace("data/code15-12l/", "").replace(".hdf5", "")
        path_to_h5_train, path_to_csv_train = filepath, '../../data/code15-12l/exams.csv' # path_to_records = 'data/codesubset/RECORDS.txt'
        print("path_to_h5_train:", path_to_h5_train, "path_to_csv", path_to_csv_train)
        # load traces
        f = h5py.File(path_to_h5_train, 'r')
        traces = np.array(f['tracings'][()], dtype=np.float32)[:-1,:,:]
        print("traces successfully converted to tensors ...")
        # load labels
        df = pd.read_csv(path_to_csv_train)
        df.set_index('exam_id', inplace=True)
        df = df.reindex(np.array(f['exam_id'])).dropna(subset=["AF"]) # make sure the order is the same
        labels = np.array(df['AF'], dtype=np.float32).reshape(-1,1)
        
        if len(trains["features"]) > 0:
            trains["features"] = np.vstack((trains["features"], traces))
            trains["labels"] = np.vstack((trains["labels"], labels))
            print("VSTACK DONE >>", len(trains["features"]), len(trains["labels"]))
        else:
            trains["features"] = traces
            trains["labels"] = labels

    trainset = TensorDataset(torch.tensor(trains["features"], dtype=torch.float32), torch.tensor(trains["labels"], dtype=torch.float32))
    trains, testset = random_split(trainset, [0.8, 0.2])
    print("Dataset prepared with_format('torch') >>", trains)

    # partition the data
    # courtesy: https://flower.ai/docs/baselines/niid_bench.html
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
        targets = tmp_t.flatten()
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
    elif partitioning == "label_quantity":
        """
        labels_per_client = 2
        if "labels_per_client" in config:
            labels_per_client = config.labels_per_client
        datasets, testset = partition_data_label_quantity(
            num_clients,
            labels_per_client=labels_per_client,
            seed=seed,
            dataset_name=config.name,
        )
        """
    # courtesy: https://flower.ai/docs/baselines/niid_bench.html
    elif partitioning == "iid": 
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
        targets = tmp_t[rem_trainset.indices].flatten()
        num_remaining_classes = len(set(targets))
        remaining_classes = list(set(targets))
        client_classes: List[List] = [[] for _ in range(num_partitions)]
        times = [0 for _ in range(num_remaining_classes)]

        for i in range(num_partitions):
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

        for i in range(num_remaining_classes):
            class_t = remaining_classes[i]
            idx_k = np.where(targets == i)[0]
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
            trainsets_per_client[i] = ConcatDataset(
                [trainsets_per_client[i]] + rem_trainsets_per_client[i]
            )

        return DataLoader(trainsets_per_client[partition_id], batch_size=batch_size, shuffle=True, num_workers=4), DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
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
    #optimizer = SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
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
    loss = 0.0
    sigmoid = nn.Sigmoid().to(device)
    avg_precisions = []  # avg precision (pr-auc)

    with torch.no_grad():
        for data, target in testloader:
            assert not isinstance(data, str), "FAULTY DATALOADER ... Check your data loading."
            data, target = data.to(device), target.to(device)
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
