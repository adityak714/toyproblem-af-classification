"""pytorchexample: A Flower / PyTorch app."""

import torch, time, subprocess
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import ResNet1d, load_datasets
from pytorchexample.task import test as test_fn
from pytorchexample.task import train_fedavg as train_fn

# Flower ClientApp
app = ClientApp()
trainloader, valloader = [], []

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Load the necessary args
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    partitioning = context.run_config["partitioning"]
    learning_rate = msg.content["config"]["lr"]
    local_epochs = context.run_config["local-epochs"]
    
    # Load the model and initialize it with the received weights
    #model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model_path = f"output-client{partition_id}.pt"
    partition_model = {"MODEL_STATE": msg.content["arrays"].to_torch_state_dict()}
    torch.save(partition_model, model_path)

    # Call the training function
    start_time = time.time()
    
    with open(f"output-client{partition_id}.txt", "w") as f:
        train_loss = subprocess.run(
            f'torchrun --standalone --nproc_per_node=4 pytorchexample/task.py --model_path {model_path} --num_clients {num_partitions} --partition_id {partition_id} --batch_size {batch_size} --epochs {local_epochs} --learning_rate {learning_rate}'.split(),
            stdout=f,
            text=True
        )
    
    end_time = time.time()
    training_time = end_time - start_time

    # Construct and return reply Message
    model_record = ArrayRecord(torch.load(model_path)["MODEL_STATE"])
    metrics = {
        "train_loss": train_loss.returncode,
        "num-examples": len(trainloader),
        "training_time": training_time
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_datasets(partition_id, num_partitions, batch_size)
    
    # Load the model and initialize it with the received weights
    model = ResNet1d(n_classes=1)
    #model_path = f"output-client{partition_id}.pt"
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict()) # torch.load(model_path)["MODEL_STATE"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(model, valloader, device)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
