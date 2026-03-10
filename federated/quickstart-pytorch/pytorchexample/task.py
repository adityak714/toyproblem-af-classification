"""Implement the neural network models and training functions."""
from tqdm.notebook import trange, tqdm
from typing import List, Tuple
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import torch, h5py
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from torch.nn.parameter import Parameter
from torch.optim import SGD, Optimizer, Adam
from torch.utils.data import DataLoader, TensorDataset
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

"""
def load_data(partition_id: int, num_partitions: int, batch_size: int):
    # Load partition CIFAR10 data.
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader
"""

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
def load_datasets(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    partitioning: str = "iid",
    seed: Optional[int] = 42,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:

    trainloaders, testloader = [], []

    print("Loading hdf5 ...")
    datasets = []
    for i, filepath in enumerate(glob.glob("../data/code15-12l/*.hdf5")[:4]):
        trains = {
            "features": [],
            "labels": []
        }
        prefix = filepath.replace("data/code15-12l/", "").replace(".hdf5", "")
        path_to_h5_train, path_to_csv_train = filepath, '../data/code15-12l/exams.csv' # path_to_records = 'data/codesubset/RECORDS.txt'
        print("path_to_h5_train:", path_to_h5_train, "path_to_csv", path_to_csv_train)
        # load traces
        f = h5py.File(path_to_h5_train, 'r')
        traces = np.array(f['tracings'][()], dtype=torch.float32)[:-1,:,:]
        print("traces successfully converted to tensors ...")
        # load labels
        df = pd.read_csv(path_to_csv_train)
        df.set_index('exam_id', inplace=True)
        df = df.reindex(np.array(f['exam_id'])).dropna(subset=["AF"]) # make sure the order is the same
        labels = np.array(df['AF'], dtype=np.float32).reshape(-1,1)
        
        print("reindexing of the csv for the given chunk of code15% successful ... now splitting train-test ...")
        trains["features"] = traces
        trains["labels"] = labels    
        datasets.append(pd.DataFrame(trains))

    df = pd.concat(datasets)
    dataset = Dataset.from_pandas(df)

    # partition the data
    if partitioning == "dirichlet":
        """
        alpha = 0.5
        if "alpha" in config:
            alpha = config.alpha
        datasets, testset = partition_data_dirichlet(
            num_clients,
            alpha=alpha,
            seed=seed,
            dataset_name=config.name,
        )
        """
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
    # both this and below call the same function! only difference is similarity value for non-IID.
    elif partitioning == "iid": 
        global fds
        if fds is None:
            fds = IidPartitioner(num_partitions=num_partitions)
            fds.dataset = dataset

        partition = fds.load_partition(partition_id)
        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    elif partitioning == "iid_noniid":
        """
        similarity = 0.5
        if "similarity" in config:
            similarity = config.similarity
        datasets, testset = partition_data(
            num_clients,
            similarity=similarity,
            seed=seed,
            dataset_name=config.name,
        )
        """
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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(net.parameters(), lr=learning_rate) #,weight_decay=weight_decay)
    #optimizer = SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    for i in range(epochs):
        net = _train_one_epoch(net, rank, trainloader, criterion, optimizer, i)

def _train_one_epoch(
    net, #: nn.Module >> changed to DDP for parallellization
    rank, # torch.device
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    epoch: int
) -> nn.Module:
    """Train the network on the training set for one epoch."""
    net.to(rank)
    tqdm.write("Training model...")
    train_pbar = tqdm(trainloader, desc="Training Epoch {epoch:2d}".format(epoch=1), leave=True)
    total_loss, n_entries = 0, 0
     
    net.train()
    for traces, diagnoses in train_pbar:
        traces, diagnoses = traces.to(rank), diagnoses.to(rank)
        for x,y in trainloader:
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

    return net

# def test(net, testloader, device):
def test(
        net, testloader: DataLoader, 
        device, # == rank >> before: torch.device
        # world_size=torch.cuda.device_count(): int
) -> Tuple[float, float]:
    """Evaluate the network on the test set.""" 
    net.to(device)
    net.eval()
    criterion = nn.BCEWithLogitsLoss() # nn.CrossEntropyLoss(reduction="sum")
    loss = 0.0
    sigmoid = nn.Sigmoid().to(device)
    avg_precisions = []  # avg precision (pr-auc)

    with torch.no_grad():
        for data, target in testloader:
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