"""Defines the client class and support functions for FedAvg."""

from typing import Callable, Dict, List, OrderedDict

import flwr as fl
import torch, os
from flwr.common import Scalar, Context
from typing import Dict, List, Union, Optional, Tuple
from flwr.common.typing import NDArrays
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from niid_bench.models import test, train_fedavg

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "10.21.30.152"
    os.environ["MASTER_PORT"] ) "12355"
    init_process_group(backend="nccl", # nvidia comm library (backend for distr. comm. across CUDA GPUs
            rank=rank, world_size=world_size)


# pylint: disable=too-many-instance-attributes
class FlowerClientFedAvg(fl.client.NumPyClient):
    """Flower client implementing FedAvg."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net, #: DistributedDataParallel (DDP)
        trainloader: DataLoader,
        valloader: DataLoader,
        #device: torch.device,
        rank: int,
        world_size: int
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        #self.device = device
        self.rank = rank
        self.world_size = world_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.module.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.module.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v).detach().clone() for k, v in params_dict})
        self.net.module.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for FedAvg."""
        ddp_setup(self.rank, self.world_size) # move INSIDE train_fedavg, not here.
        self.set_parameters(parameters)
        # TODO: mp.spawn is BEST to go here mp.spawn(train_fedavg, args=(world_size,...), nprocs=world_size)
        train_fedavg(
            self.net,
            self.trainloader,
            #self.device,
            self.rank,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
        )
        final_p_np = self.get_parameters({})
        destroy_process_group() # move INSIDE train_fedavg, not here.
        return final_p_np, len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        loss, acc = test(self.net, self.valloader, self.rank)
        return float(loss), len(self.valloader.dataset), {"ap": float(acc)}

# pylint: disable=too-many-arguments
def gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    rank: int,
    world_size: int,
    num_epochs: int,
    learning_rate: float,
    model: DictConfig,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
) -> Callable[[str], FlowerClientFedAvg]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the FedAvg flower clients.

    Parameters
    ----------
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    momentum : float
        The momentum for SGD optimizer of clients
    weight_decay : float
        The weight decay for SGD optimizer of clients

    Returns
    -------
    Callable[[str], FlowerClientFedAvg]
        The client function that creates the FedAvg flower clients
    """

    def client_fn(cid: str) -> FlowerClientFedAvg:
        """Create a Flower client representing a single organization."""
        # Load model
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        #world_size = torch.cuda.device_count()

        net = DDP(instantiate(model), device_ids=[self.rank])
        print(torch.cuda.device_count())

        #if torch.cuda.device_count() > 1:
        #    net = torch.nn.DataParallel(net)#, device_ids=list(range(torch.cuda.device_count())))
        #net.to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClientFedAvg(
            net,
            trainloader,
            valloader,
            #device,
            self.rank, # analogous to gpu_id # CHANGE
            self.world_size, # world_size (rank variable is dynamically inserted, this is mp.spawn called by main.py)
            num_epochs,
            learning_rate,
            momentum,
            weight_decay,
        ).to_client()

    return client_fn

#################### brought from server_scaffold.py
def gen_evaluate_fn(
    testloader: DataLoader,
    #device: torch.device,
    rank: int,
    world_size: int,
    model: DictConfig,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
               Optional[Tuple[float, Dict[str, Scalar]]] ]
    The centralized evaluation function.
    """

    def evaluate(server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        #net = instantiate(model)
        net = DDP(instantiate(model), device_ids=[self.rank])
        params_dict = zip(net.module.state_dict().keys(), parameters_ndarrays) 
        state_dict = OrderedDict({k: torch.from_numpy(v).detach().clone() for k, v in params_dict}) 
        net.module.load_state_dict(state_dict, strict=True)
        #net.to(device)

        # TODO: call mp.spawn for this also.? mp.spawn(test, args=(world_size,...))
        loss, accuracy = test(net, testloader) 
        # INSIDE test(), do the ddp_setup, and destroy_process() things 
        # test() must have rank in its signature!
        return loss, {"ap": accuracy}

    return evaluate
