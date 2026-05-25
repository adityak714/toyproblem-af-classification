"""Defines the client class and support functions for SCAFFOLD."""

import os, time
from typing import Callable, Dict, List, OrderedDict

import flwr as fl
import torch
from flwr.common import Context, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pytorchexample.resnet import ResNet1d
from pytorchexample.task import test, train_scaffold, load_datasets

# pylint: disable=too-many-instance-attributes
class FlowerClientScaffold(fl.client.NumPyClient):
    """Flower client implementing scaffold."""

    # pylint: disable=too-many-arguments
    def __init__(
        self, cid: int, net: torch.nn.Module, num_partitions: int, batch_size: int,
        val, device: torch.device, num_epochs: int, learning_rate: float,
        momentum: float, weight_decay: float, save_dir: str = "",
    ) -> None:
        self.cid = cid
        self.net = net
        self.num_partitions = num_partitions
        self.batch_size = batch_size
        self.val = val
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.trainloader = None
        self.valloader = None
        
        self.dir = save_dir if save_dir != "" else "client_cvs"
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
    
    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters and buffers using state_dict."""
        return [val.cpu().numpy() for val in self.net.state_dict().values()]

    def set_parameters(self, parameters):
        """Safely set local model parameters and buffers."""
        state_dict = self.net.state_dict()
        # Enforce that all state_dict values (including BatchNorm buffers) are updated
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v, dtype=state_dict[k].dtype)
        
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for SCAFFOLD."""
        # Unpack absolute weights and server control variates
        num_state_dict = len(self.net.state_dict())
        server_w = parameters[:num_state_dict]
        server_cv_np = parameters[num_state_dict:]
        
        # Load the global weights/buffers into the model
        self.set_parameters(server_w)

        # Initialize or load client_cv ONLY for trainable parameters
        cv_path = f"{self.dir}/client_cv_{self.cid}.pt"
        if os.path.exists(cv_path):
            self.client_cv = torch.load(cv_path, map_location=self.device)
        else:
            self.client_cv = [torch.zeros_like(p, device=self.device) for p in self.net.parameters() if p.requires_grad]
        
        # Convert server CV arrays to tensors
        server_cv = [torch.tensor(cv, device=self.device) for cv in server_cv_np]

        # Backup initial trainable weights (x) for CV update calculation
        x_trainable = [p.clone().detach() for p in self.net.parameters() if p.requires_grad]

        trainloader, valloader = load_datasets(
            self.cid, self.num_partitions, self.batch_size,
            partitioning="dirichlet", val=self.val, device=self.device
        )

        print(f"[client {self.cid}] starting train_scaffold with batches {len(trainloader)}")
        start_time = time.time()
        train_loss, net, count = train_scaffold({
                "net": self.net,
                "partition_id": self.cid,
                "trainloader": trainloader,
                "valloader": valloader,
                "epochs": self.num_epochs, 
                "lr": self.learning_rate, 
                "batch_size": self.batch_size
            },
            server_cv,
            self.client_cv,
        )
        end_time = time.time()
        training_time = end_time - start_time
        print("training done!")
        
        # Fetch updated trainable weights (y_i)
        y_trainable = [p.clone().detach() for p in self.net.parameters() if p.requires_grad]

        c_i_n = []
        server_update_c = []
        K = self.num_epochs * len(trainloader) # Total update steps

        # Update client control variate: c_i^+ = c_i - c + 1/(K * eta) * (x - y_i)
        for c_i, c_s, x_j, y_j in zip(self.client_cv, server_cv, x_trainable, y_trainable):
            c_i_plus = c_i - c_s + (1.0 / (K * self.learning_rate)) * (x_j - y_j)
            c_i_n.append(c_i_plus)
            # Server expects the difference Delta c_i = c_i^+ - c_i
            server_update_c.append((c_i_plus - c_i).cpu().numpy())
            
        self.client_cv = c_i_n
        torch.save(self.client_cv, cv_path)

        # Combined updates payload: [New Absolute Weights/Buffers] + [Delta Client CVs]
        combined_updates = self.get_parameters(config={}) + server_update_c

        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
            "training_time": training_time,
            "local-epochs": self.num_epochs,
            "partition_id": self.cid
        }

        with open(f'{self.dir}/clients{self.num_partitions}-partitioningdirichlet{self.val}-loceps{self.num_epochs}.txt', "a") as logger:
            logger.write(f"{str(metrics)}\n")
        
        return combined_updates, len(trainloader.dataset), {}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        
        trainloader, valloader = load_datasets(
            self.cid, 
            self.num_partitions, 
            self.batch_size,
            partitioning="dirichlet",
            val=self.val, device=self.device
        )
        self.trainloader = trainloader
        self.valloader = valloader

        loss, acc = test(self.net, self.valloader, self.device)
        metrics = {
            "eval_loss": float(loss),
            "eval_acc": float(acc),
            "num-examples": len(self.valloader.dataset)
        }
        with open(f'{self.dir}/clients{self.num_partitions}-partitioningdirichlet{self.val}-loceps{self.num_epochs}.txt', "a") as logger:
            logger.write(f"{str(metrics)}\n")
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}


# pylint: disable=too-many-arguments
def gen_client_fn(
    net,
    num_partitions,
    batch_size,
    val, # alpha --- dirichlet-iid-non-iid partitioning
    client_cv_dir: str,
    num_epochs: int,
    learning_rate: float,
    momentum: float = 0,
    weight_decay: float = 0.0,
    ) -> Callable[[Context], fl.client.Client]:
    # -> Callable[[str], FlowerClientScaffold]:  
    # pylint: disable=too-many-arguments
    
    def client_fn(context: Context):
        """Create a Flower client representing a single organization."""
        # Load model
        cid = int(context.node_config.get("partition-id", context.node_id))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        model = ResNet1d(n_classes=1)
        model.load_state_dict(net.to_torch_state_dict())
        model.to(device)
        
        return FlowerClientScaffold(
            cid,
            model,
            num_partitions,
            batch_size,
            val,
            device,
            num_epochs,
            learning_rate,
            momentum,
            weight_decay,
            save_dir=client_cv_dir,
        ).to_client()

    return client_fn
