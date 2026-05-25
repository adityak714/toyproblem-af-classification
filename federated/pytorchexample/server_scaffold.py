"""Server class for SCAFFOLD."""

from logging import DEBUG, INFO
from typing import Callable, Dict, List, Optional, OrderedDict, Tuple, Union

import torch, numpy as np
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns, NDArrays
from flwr.server import Server
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import fit_clients
from flwr.server.strategy import Strategy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pytorchexample.task import test
from pytorchexample.resnet import ResNet1d

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]

class ScaffoldStrategy(FedAvg):
    """Implement custom strategy for SCAFFOLD based on FedAvg class."""

    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate everything seamlessly. W_i gets averaged into W_agg, Delta C_i into Delta C_agg
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_arrays = aggregate(weights_results)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        # Return the unified array block. The server will unpack it.
        return ndarrays_to_parameters(aggregated_arrays), metrics_aggregated

class ScaffoldServer(Server):
    """Implement server for SCAFFOLD."""

    def __init__(self, strategy: Strategy, model, client_manager=None, global_lr: float = 1.0):
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.model_params = ResNet1d(n_classes=1)
        self.server_cv: List[torch.Tensor] = []
        self.global_lr = global_lr 

    def _get_initial_parameters(self, server_round: int, timeout: Optional[float]) -> Parameters:
        """Get initial parameters and strictly scope CVs to trainable variables."""
        parameters = self.strategy.initialize_parameters(client_manager=self._client_manager)
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
        else:
            log(INFO, "Requesting initial parameters from one random client")
            random_client = self._client_manager.sample(1)[0]
            ins = GetParametersIns(config={})
            get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout, group_id=server_round)
            parameters = get_parameters_res.parameters

        # CVs ONLY track gradient-requiring parameters, unlike the main weights/buffers
        self.server_cv = [
            torch.zeros_like(p) for p in self.model_params.parameters() if p.requires_grad
        ]
        return parameters

    def fit_round(self, server_round: int, timeout: Optional[float]):
        """Perform a single round of federated averaging."""
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=update_parameters_with_cv(self.parameters, self.server_cv),
            client_manager=self._client_manager,
        )

        if not client_instructions:
            return None

        results, failures = fit_clients(
            client_instructions=client_instructions, max_workers=self.max_workers,
            timeout=timeout, group_id=server_round,
        )

        aggregated_result = self.strategy.aggregate_fit(server_round, results, failures)
        if aggregated_result[0] is None:
            return None
        
        aggregated_arrays = parameters_to_ndarrays(aggregated_result[0])

        # Slice the payload into W (Weights/Buffers) and C (Control Variates)
        num_state_dict = len(self.model_params.state_dict())
        aggregated_w = aggregated_arrays[:num_state_dict]
        aggregated_dc = aggregated_arrays[num_state_dict:]

        # Global model update: x_new = x_old + global_lr * (aggregated_w - x_old)
        curr_w = parameters_to_ndarrays(self.parameters)
        updated_w = [x + self.global_lr * (agg_w - x) for x, agg_w in zip(curr_w, aggregated_w)]
        parameters_updated = ndarrays_to_parameters(updated_w)
        self.parameters = parameters_updated

        # Update Server CV: c = c + (|S| / N) * aggregated_dc
        total_clients = len(self._client_manager.all())
        cv_multiplier = len(results) / total_clients
        self.server_cv = [
            cv + cv_multiplier * torch.from_numpy(dc)
            for cv, dc in zip(self.server_cv, aggregated_dc)
        ]

        metrics_aggregated = aggregated_result[1]
        return parameters_updated, metrics_aggregated, (results, failures)

"""
class ScaffoldServer(Server):
    #"Implement server for SCAFFOLD."

    def __init__(
        self,
        strategy: Strategy,
        model,
        client_manager: Optional[ClientManager] = None,
        global_lr: float = 1.0
    ):
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.model_params = ResNet1d(n_classes=1)
        self.server_cv: List[torch.Tensor] = []
        self.global_lr = global_lr # new - for modifying a global learning rate

    def _get_initial_parameters(self, server_round: int, timeout: Optional[float]) -> Parameters:
        #"Get initial parameters from one of the available clients."
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            self.server_cv = [
                torch.zeros_like(torch.from_numpy(t))
                for t in parameters_to_ndarrays(parameters)
            ]
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(
            ins=ins, timeout=timeout, group_id=server_round
        )
        if get_parameters_res.status.code == Code.OK:
            log(INFO, "Received initial parameters from one random client")
        else:
            log(INFO, "Failed to receive initial parameters from the client. Using empty initial parameters.")

        self.server_cv = [
            torch.zeros_like(torch.from_numpy(t))
            for t in parameters_to_ndarrays(get_parameters_res.parameters)
        ]
        return get_parameters_res.parameters

    # pylint: disable=too-many-locals
    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        "Perform a single round of federated averaging."
        # Get clients and their respective instructions from strateg
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=update_parameters_with_cv(self.parameters, self.server_cv),
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[Optional[Parameters], Dict[str, Scalar]] = (
            self.strategy.aggregate_fit(server_round, results, failures)
        )

        #aggregated_result_arrays_combined = []
        if aggregated_result[0] is None:
            return None
        
        aggregated_result_arrays_combined = parameters_to_ndarrays(
            aggregated_result[0]
        )
        aggregated_parameters = aggregated_result_arrays_combined[
            : len(aggregated_result_arrays_combined) // 2
        ]
        aggregated_cv_update = aggregated_result_arrays_combined[
            len(aggregated_result_arrays_combined) // 2 :
        ]

        # convert server cv into ndarrays
        server_cv_np = [cv.numpy() for cv in self.server_cv]
        # update server cv
        total_clients = len(self._client_manager.all())
        cv_multiplier = len(results) / total_clients
        self.server_cv = [
            torch.from_numpy(cv + cv_multiplier * aggregated_cv_update[i])
            for i, cv in enumerate(server_cv_np)
        ]

        # update parameters x = x + global_lr* aggregated_update
        curr_params = parameters_to_ndarrays(self.parameters)
        updated_params = [
            x + (self.global_lr*aggregated_parameters[i]) for i, x in enumerate(curr_params)
        ]
        parameters_updated = ndarrays_to_parameters(updated_params)

        # metrics
        metrics_aggregated = aggregated_result[1]
        return parameters_updated, metrics_aggregated, (results, failures)
"""

def update_parameters_with_cv(
    parameters: Parameters, s_cv: List[torch.Tensor]
) -> Parameters:
    """Extend the list of parameters with the server control variate."""
    # extend the list of parameters arrays with the cv arrays
    cv_np = [cv.numpy() for cv in s_cv]
    parameters_np = parameters_to_ndarrays(parameters)
    parameters_np.extend(cv_np)
    return ndarrays_to_parameters(parameters_np)
