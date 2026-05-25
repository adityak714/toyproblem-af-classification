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
        self,
        cid: int,
        net: torch.nn.Module,
        num_partitions: int,
        batch_size: int,
        val,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        save_dir: str = "",
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
        
        # initialize client control variate with 0 and shape of the network parameters
        self.client_cv = []
        
        # load_datasets
        self.trainloader = None
        self.valloader = None
        
        for param in self.net.parameters():
            self.client_cv.append(torch.zeros(param.shape))
        # save cv to directory
        if save_dir == "":
            save_dir = "client_cvs"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
    
    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.detach().cpu().numpy() for _, val in self.net.named_parameters()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        #params_dict = zip(list(self.net.state_dict().keys()), parameters)
        #state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        #params_list = list(self.net.parameters())
        #if len(parameters) != len(params_list):
        #    raise ValueError(f"Expected {len(params_list)} but got {len(parameters)}")
        #with torch.no_grad():
        #    for i, param in enumerate(params_list):
        #        param.copy_(torch.as_tensor(parameters[i]).to(param.device))
         
        state_dict = self.net.state_dict()
        params_list = [k for k, _ in self.net.named_parameters()]

        for k, v in zip(params_list, parameters):
            state_dict[k] =  torch.tensor(v, dtype=state_dict[k].dtype)
        
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for SCAFFOLD."""
        # the first half are model parameters and the second are the server_cv
        server_cv = parameters[len(parameters) // 2 :]
        parameters = parameters[: len(parameters) // 2]
        self.set_parameters(parameters)
        self.client_cv = []
        #for param in self.net.parameters():
        #    self.client_cv.append(param.clone().detach())
        # load client control variate
        if os.path.exists(f"{self.dir}/client_cv_{self.cid}.pt"):
            self.client_cv = torch.load(f"{self.dir}/client_cv_{self.cid}.pt")
        else:
            self.client_cv = [torch.zeros_like(param) for param in self.net.parameters()]
        # convert the server control variate to a list of tensors
        server_cv = [torch.Tensor(cv) for cv in server_cv]

        #print(self.cid)
        trainloader, valloader = load_datasets(
            self.cid, 
            self.num_partitions, 
            self.batch_size,
            partitioning="dirichlet",
            val=self.val, device=self.device
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
        
        x = parameters
        y_i = self.get_parameters(config={})
        c_i_n = []
        server_update_x = []
        server_update_c = []

        # update client control variate c_i_1 = c_i - c + 1/eta*K (x - y_i)
        for c_i_j, c_j, x_j, y_i_j in zip(self.client_cv, server_cv, x, y_i):
            #print(c_i_j.device, c_j.device)
            c_i_n.append(
                c_i_j.cpu()
                - c_j.cpu()
                + (1.0 / (self.learning_rate * self.num_epochs * len(trainloader)))
                * (x_j - y_i_j)
            )
            # y_i - x, c_i_n - c_i for the server
            server_update_x.append((y_i_j - x_j))
            server_update_c.append((c_i_n[-1] - c_i_j.cpu()).cpu().numpy())
        self.client_cv = c_i_n
        torch.save(self.client_cv, f"{self.dir}/client_cv_{self.cid}.pt")

        combined_updates = server_update_x + server_update_c

        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
            "training_time": training_time,
            "local-epochs": self.num_epochs,
            "partition_id": self.cid
        }
        #metric_record = MetricRecord(metrics)
        #content = RecordDict({"arrays": model_record, "metrics": metric_record})

        with open(f'{self.dir}/clients{self.num_partitions}-partitioningdirichlet{self.val}-loceps{self.num_epochs}.txt', "a") as logger:
            logger.write(f"{str(metrics)}\n")
        
        return (
            combined_updates,
            len(trainloader.dataset),
            {},
        )

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
