"""pytorchexample: A Flower / PyTorch app."""

import flwr, torch, os, uuid, json, glob, h5py, pickle
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
from collections import OrderedDict
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from flwr.server.client_manager import SimpleClientManager
from datetime import date
from pytorchexample.resnet import ResNet1d
from pytorchexample.task import load_centralized_dataset, load_datasets, test
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay

###################### SCAFFOLD imports
from pytorchexample.server_scaffold import ScaffoldServer, ScaffoldStrategy
from pytorchexample.client_scaffold import gen_client_fn
######################

today = date.today()
unique_id = str(uuid.uuid4())
stratname = ''

def main() -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    num_partitions: int = 10 #context.run_config["num-partitions"]
    num_rounds: int = 60 #context.run_config["num-server-rounds"]
    fraction_fit: float = 1.0 #context.run_config["fraction-fit"]
    fraction_evaluate: float = 0.5 #context.run_config["fraction-evaluate"]
    lr: float = 0.001 #context.run_config["learning-rate"]
    batch_size: int = 256 #context.run_config["batch-size"]
    stratname = "scaffold" #context.run_config["strategy"]
    val = 0.25 #context.run_config["val"] # alpha value for dirichl-based partitioning
    local_epochs: int = 5 #context.run_config["local-epochs"]
    
    os.environ["CURR_FLWR_SESSION_ID"] = unique_id
    with open(f"tmp-scaffold-d{val}-lr{lr}-frac{fraction_fit}.txt", 'w') as f:
        print("CURR_FLWR_SESSION_ID set as", os.environ["CURR_FLWR_SESSION_ID"])
        filetree = f"./runs/{today}-{unique_id}"
        f.write(filetree)

    if not os.path.isdir(filetree):
        os.makedirs(filetree, exist_ok=True)

    # Load global model
    global_model = ResNet1d(n_classes=1)
    arrays = ArrayRecord(global_model.state_dict())

    client_cv_dir = ""
    strategy = None

    # Initialize FedPROX strategy (before: FEDAVG)
    if stratname == 'fedprox':
        proxmu = context.run_config["proxmu"]
        strategy = FedProx(fraction_evaluate=fraction_evaluate, proximal_mu=proxmu) 
        stratname = f'fedprox{proxmu}'
    elif stratname == 'scaffold':
        client_cv_dir = f"runs/{today}-{unique_id}"
        print("Local cvs for scaffold clients are saved to: ", client_cv_dir)
    else:
        strategy = FedAvg(fraction_evaluate=fraction_evaluate) 

    with open(f'runs/{today}-{unique_id}/{stratname}.txt', 'a'): 
        pass

    if stratname == 'scaffold':
        strategy = ScaffoldStrategy(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            evaluate_fn=scaffold_global_evaluate
        )
        server = ScaffoldServer(
            strategy=strategy, 
            model=arrays, 
            client_manager=SimpleClientManager(),
            global_lr=1.0
        )
        client_fn = gen_client_fn(
            arrays,
            num_partitions,
            batch_size,
            val, # alpha --- dirichlet-iid-non-iid partitioning
            client_cv_dir,
            local_epochs, lr)
        
        history = flwr.simulation.start_simulation(
            server=server,
            client_fn=client_fn,
            num_clients=num_partitions,
            config=flwr.server.ServerConfig(num_rounds=num_rounds),
            client_resources={
                "num_cpus": 16,
                "num_gpus": 1.0,
            },
            strategy=strategy,
            ray_init_args={
                "address": "auto"
            }
        )
        print(history)
        with open(f"runs/{today}-{unique_id}/{stratname}-cl{num_partitions}.pkl", "wb") as f_ptr:
            pickle.dump(history, f_ptr)
 

def scaffold_global_evaluate(server_round: int, parameters, config):
    #params_dict = zip(model.state_dict().keys(), [torch.as_tensor(p) for p in parameters])
    #state_dict = OrderedDict({k: v for k, v in params_dict})
    #model.load_state_dict(state_dict, strict=True)
    model = ResNet1d(n_classes=1)
    with torch.no_grad():
        for param, array in zip(model.parameters(), parameters):
            param.copy_(torch.as_tensor(array))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    model.eval()
    scores = {}
    for i, filepath in enumerate(glob.glob("../data/code-test/annotations/*.csv")):
        if "gold_standard" in filepath:
            f = h5py.File("../data/code-test/ecg_tracings.hdf5", 'r')
            traces = torch.tensor(np.array(f['tracings'], dtype=np.float32), dtype=torch.float32, device=device)
            y_trues = torch.tensor(pd.read_csv(filepath)["AF"], dtype=torch.float16).reshape(-1,1)
            
            test_subset = filepath.replace("../data/code-test/annotations/", "").replace(".csv", "")
            # Run forward pass
            with torch.no_grad():
                y_preds = model(traces).cpu().numpy()
            
            PrecisionRecallDisplay.from_predictions(y_trues, y_preds)
            score = average_precision_score(y_trues, y_preds)
            plt.title(f"PR-Curve on Annotated ECG Dataset - {test_subset}.png")
            plt.savefig(f"runs/{today}-{unique_id}/prCODEtest-{test_subset}.png", dpi=300)
            plt.close()
            
            # add to dictionary storing all holdout scores
            scores[test_subset] = score

    with open(f'runs/{today}-{unique_id}/holdoutmetrics-CODETEST.txt', 'a') as f:
        f.write(f"{scores}\n")

    #torch.save({
    #    "MODEL_STATE": arrays.to_torch_state_dict(),
    #    "COMM_ROUND": server_round
    #}, f'runs/{today}-{unique_id}/fl-globmod-snapshot.pt')

    record = {"comm_round": server_round, "serveragg_avg_prec": test_acc, "serveragg_loss": test_loss}
    with open(f'runs/{today}-{unique_id}/serveragg-metrics.txt', 'a') as f:
        f.write(f"{dict(record)}\n") 
    
    return float(test_loss), {"accuracy": float(test_acc)}

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = ResNet1d(n_classes=1)
    model.load_state_dict(arrays.to_torch_state_dict())
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    ## Testing on gold_standard, cardiologist1 + cardiologist2
    ## HOLDOUT
    model.eval()
    scores = {}
    for i, filepath in enumerate(glob.glob("../data/code-test/annotations/*.csv")):
        if "gold_standard" in filepath:
            f = h5py.File("../data/code-test/ecg_tracings.hdf5", 'r')
            traces = torch.tensor(np.array(f['tracings'], dtype=np.float32), dtype=torch.float32, device=device)
            y_trues = torch.tensor(pd.read_csv(filepath)["AF"], dtype=torch.float16).reshape(-1,1)
            
            test_subset = filepath.replace("../data/code-test/annotations/", "").replace(".csv", "")
            # Run forward pass
            with torch.no_grad():
                y_preds = model(traces).cpu().numpy()
            
            PrecisionRecallDisplay.from_predictions(y_trues, y_preds)
            score = average_precision_score(y_trues, y_preds)
            plt.title(f"PR-Curve on Annotated ECG Dataset - {test_subset}.png")
            plt.savefig(f"runs/{today}-{unique_id}/prCODEtest-{test_subset}.png", dpi=300)
            plt.close()
            
            # add to dictionary storing all holdout scores
            scores[test_subset] = score

    with open(f'runs/{today}-{unique_id}/holdoutmetrics-CODETEST.txt', 'a') as f:
        f.write(f"{scores}\n")

    torch.save({
        "MODEL_STATE": arrays.to_torch_state_dict(),
        "COMM_ROUND": server_round
    }, f'runs/{today}-{unique_id}/fl-globmod-snapshot.pt')

    record = MetricRecord({"comm_round": server_round, "serveragg_avg_prec": test_acc, "serveragg_loss": test_loss})
    with open(f'runs/{today}-{unique_id}/serveragg-metrics.txt', 'a') as f:
        f.write(f"{dict(record)}\n") 

    # Return the evaluation metrics
    return record

if __name__ == "__main__":
    main()
