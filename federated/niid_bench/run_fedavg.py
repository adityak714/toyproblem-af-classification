import os, pickle, time
import flwr as fl
import hydra
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from flwr.common import Scalar, Context
from typing import Callable, Dict, List, OrderedDict, Union, Optional, Tuple
from flwr.common.typing import NDArrays
from hydra.utils import instantiate
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm

#from niid_bench.server_fednova import FedNovaServer
from niid_bench.dataset_preparation import (
    partition_data,
    partition_data_dirichlet,
    partition_data_label_quantity,
)
from torch.utils.data import DataLoader, random_split
#from niid_bench.server_scaffold import ScaffoldServer#, gen_evaluate_fn
from niid_bench.strategy import FedNovaStrategy, ScaffoldStrategy

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import SGD, Optimizer, Adam
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(
            backend="nccl", # nvidia comm library (for distr. comm. across CUDA GPUs)
            rank=rank,
            world_size=world_size
    )

# pylint: disable=too-many-instance-attributes
class FlowerClientFedAvg(fl.client.NumPyClient):
    """Flower client implementing FedAvg."""
    # pylint: disable=too-many-arguments
    def __init__(
        self, net, #: DistributedDataParallel (DDP)
        trainloader: DataLoader, valloader: DataLoader,
        # rank: int, world_size: int, 
        device: torch.device,
        num_epochs: int, learning_rate: float,
        momentum: float, weight_decay: float,
    ) -> None:
        #self.world_size = world_size
        #self.rank = rank
        self.device = device
        #print("\nclient initialized >>>> WORLD_SIZE ==", self.world_size, ">>>> RANK ==", self.rank, ">>>>\n")
        self.net = net
        #self.net = DDP(self.net, device_ids=[rank])
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v).detach().clone() for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for FedAvg."""
        self.set_parameters(parameters)
        train_fedavg(
            self.net,
            self.device,
            self.trainloader,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
        )
        final_p_np = self.get_parameters({})
        return final_p_np, len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        loss, acc = test(
                self.net, 
                self.valloader, 
                self.device
        )
        return float(loss), len(self.valloader.dataset), {"ap": float(acc)}

def train_fedavg(
    net: nn.Module,
    rank,
    trainloader: DataLoader,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
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

####################################
# pylint: disable=too-many-arguments
def gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    num_epochs: int,
    learning_rate: float,
    # rank: int,
    model: DictConfig,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
) -> Callable[[str], FlowerClientFedAvg]:  
    # pylint: disable=too-many-arguments
    """Generate the client function that creates the FedAvg flower clients."""
    def client_fn(cid: str) -> FlowerClientFedAvg:
        """Create a Flower client representing a single organization."""
        net = instantiate(model) 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClientFedAvg(
            net, # load model (net)
            trainloader,
            valloader,
            device,
            # rank, world_size, # (rank variable is dynamically inserted, 
                                # by mp.spawn called by main.py) 
            num_epochs,
            learning_rate,
            momentum,
            weight_decay,
        ).to_client()
        
    return client_fn

####################
def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    # rank: int,
    model: DictConfig,
    world_size=torch.cuda.device_count()
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """Generate the function for centralized evaluation."""

    def evaluate(server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        # print("\ngen_evaluate >>>> RANK==", rank, ">>>>\n")
        net = instantiate(model)#.to(rank)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays) 
        state_dict = OrderedDict({k: torch.from_numpy(v).detach().clone() for k, v in params_dict}) 
        net.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(net, testloader, device) 
        return loss, {"ap": accuracy}

    return evaluate
###################

# pylint: disable=too-many-locals, too-many-branches
def load_datasets(
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    seed: Optional[int] = 42,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """Create the dataloaders to be fed into the model.

    Parameters
    ----------
    config: DictConfig
        Parameterises the dataset partitioning process
    num_clients : int
        The number of clients that hold a part of the data
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoaders for training, validation, and testing.
    """
    print(f"Dataset partitioning config: {config}")
    partitioning = ""
    if "partitioning" in config:
        partitioning = config.partitioning
    # partition the data
    if partitioning == "dirichlet":
        alpha = 0.5
        if "alpha" in config:
            alpha = config.alpha
        datasets, testset = partition_data_dirichlet(
            num_clients,
            alpha=alpha,
            seed=seed,
            dataset_name=config.name,
        )
    elif partitioning == "label_quantity":
        labels_per_client = 2
        if "labels_per_client" in config:
            labels_per_client = config.labels_per_client
        datasets, testset = partition_data_label_quantity(
            num_clients,
            labels_per_client=labels_per_client,
            seed=seed,
            dataset_name=config.name,
        )
    # both this and below call the same function! only difference is similarity value for non-IID.
    elif partitioning == "iid": 
        datasets, testset = partition_data(
            num_clients,
            similarity=1.0,
            seed=seed,
            dataset_name=config.name,
        )
    elif partitioning == "iid_noniid":
        similarity = 0.5
        if "similarity" in config:
            similarity = config.similarity
        datasets, testset = partition_data(
            num_clients,
            similarity=similarity,
            seed=seed,
            dataset_name=config.name,
        )

    batch_size = -1
    if "batch_size" in config:
        batch_size = config.batch_size
    elif "batch_size_ratio" in config:
        batch_size_ratio = config.batch_size_ratio
    else:
        raise ValueError

    # split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio)) if val_ratio > 0 else 0
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        if batch_size == -1:
            batch_size = int(len(ds_train) * batch_size_ratio)
        trainloaders.append(
            DataLoader(
                ds_train, 
                batch_size=batch_size, 
                shuffle=False, 
                pin_memory=True, 
                sampler=DistributedSampler(ds_train)
            )
        )
        print(len(ds_train))
        valloaders.append(
            DataLoader(
                ds_val, 
                batch_size=batch_size, 
                shuffle=False, 
                pin_memory=True, 
                sampler=DistributedSampler(ds_val)
            )
        )
        testloader = DataLoader(
            testset, 
            batch_size=len(testset), 
            shuffle=False, 
            pin_memory=True, 
            sampler=DistributedSampler(testset)
        )
        print(">>>> HOW MANY TRAINING SPLITS", len(trainloaders))
    return trainloaders, valloaders, testloader

def spawner(rank: int, world_size: int, cfg):
    ddp_setup(rank, world_size)
    #############################################################################
    print("\n>>>> RANK ==", rank, ">>>>\n")
    # 1. Prepare your dataset
    trainloaders, valloaders, testloader = load_datasets(
        config=cfg.dataset,
        num_clients=cfg.num_clients,
        val_ratio=cfg.dataset.val_split,
    )
    #############################################################################
    destroy_process_group()

    print(len(trainloaders), len(trainloaders[0]), "batch_size -->>", [len(x) for x, y in trainloaders[0]])

    # 2. Define your clients
    client_fn = None
    # pylint: disable=protected-access
    if cfg.client_fn._target_ == "niid_bench.client_scaffold.gen_client_fn":
        save_path = HydraConfig.get().runtime.output_dir
        client_cv_dir = os.path.join(save_path, "client_cvs")
        print("Local cvs for scaffold clients are saved to: ", client_cv_dir) 
        client_fn = call(
            cfg.client_fn,
            trainloaders,
            valloaders,
            # TODO: rank=rank,
            model=cfg.model,
            client_cv_dir=client_cv_dir,
        )
    else:
        client_fn = call(
            cfg.client_fn,
            trainloaders,
            valloaders,
            model=cfg.model,
        )

    device = torch.device("cuda:0") if cfg.server_device == "cuda" else "cpu"

    # 3. Set Evaluation Function
    evaluate_fn = gen_evaluate_fn(
            testloader, 
            device=device, # rank=rank,
            model=cfg.model
    )

    # 4. Define your strategy
    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=evaluate_fn,
    )

    # 5. Define your server
    server = Server(strategy=strategy, client_manager=SimpleClientManager())
    if isinstance(strategy, FedNovaStrategy):
        server = FedNovaServer(strategy=strategy, client_manager=SimpleClientManager())
    elif isinstance(strategy, ScaffoldStrategy):
        server = ScaffoldServer(
            strategy=strategy, model=cfg.model, client_manager=SimpleClientManager()
        )

    # 6. Start Simulation
    history = fl.simulation.start_simulation(
        server=server,
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
        ray_init_args = {"address": "auto"}
    )

    print(history)
    # destroy_process_group()
    return history

@hydra.main(config_path="conf", config_name="fedavg_base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    if "mnist" in cfg.dataset_name:
        cfg.model.input_dim = 256
        # pylint: disable=protected-access
        cfg.model._target_ = "niid_bench.models.CNNMnist"
    print(OmegaConf.to_yaml(cfg))

    world_size = torch.cuda.device_count()
    #############################################################################
    mp.spawn(spawner, args=(world_size, cfg), nprocs=world_size)
    #############################################################################

    save_path = HydraConfig.get().runtime.output_dir
    print(save_path)

    # 7. Save your results
    with open(os.path.join(save_path, "history.pkl"), "wb") as f_ptr:
        pickle.dump(history, f_ptr)

if __name__ == "__main__":
    main()
