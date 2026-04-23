"""pytorchexample: A Flower / PyTorch app."""

import torch, os, uuid, json, glob, h5py, numpy as np, pandas as pd
from matplotlib import pyplot as plt
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from datetime import date
from pytorchexample.task import ResNet1d, load_centralized_dataset, load_datasets, test
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay

# Create ServerApp
app = ServerApp()
today = date.today()
unique_id = str(uuid.uuid4())
stratname = ''

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    stratname = context.run_config["strategy"]
    
    os.environ["CURR_FLWR_SESSION_ID"] = unique_id
    with open(f"tmp{context.run_config['run_uid']}.txt", 'w') as f:
        print("CURR_FLWR_SESSION_ID set as", os.environ["CURR_FLWR_SESSION_ID"])
        filetree = f"./runs/{today}-{unique_id}"
        f.write(filetree)

    if not os.path.isdir(filetree):
        os.makedirs(filetree, exist_ok=True)

    # Load global model
    global_model = ResNet1d(n_classes=1)
    arrays = ArrayRecord(global_model.state_dict())

    strategy = None

    # Initialize FedPROX strategy (before: FEDAVG)
    if stratname == 'fedprox':
        proxmu = context.run_config["proxmu"]
        strategy = FedProx(fraction_evaluate=fraction_evaluate, proximal_mu=proxmu) 
        stratname = f'fedprox{proxmu}'
    elif stratname == 'scaffold':
        pass # stratname = 'Scaffold{...}'
    else:
        strategy = FedAvg(fraction_evaluate=fraction_evaluate) 

    with open(f'runs/{today}-{unique_id}/{stratname}.txt', 'a'): 
        pass

    try:
        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=num_rounds,
            evaluate_fn=global_evaluate,
        )
    except KeyboardInterrupt as stopped_session:
        with open(f'runs/{today}-{unique_id}/server-{stratname}-finished_metrics.txt', 'w') as f:
            f.write(str(dict(result.evaluate_metrics_serverapp)))
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, f"runs/{today}-{unique_id}/final_model.pt")
        os.remove(f"tmp{context.run_config['run_uid']}.txt")

    with open(f'runs/{today}-{unique_id}/server-{stratname}-finished_metrics.txt', 'w') as f:
        f.write(str(dict(result.evaluate_metrics_serverapp)))
    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, f"runs/{today}-{unique_id}/final_model.pt") # TODO: may need to save nn.Module instead of state_dict
    os.remove(f"tmp{context.run_config['run_uid']}.txt")

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
