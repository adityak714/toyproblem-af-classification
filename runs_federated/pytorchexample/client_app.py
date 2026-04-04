"""pytorchexample: A Flower / PyTorch app."""

import torch, time
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import ResNet1d, load_datasets
from pytorchexample.task import test as test_fn
from pytorchexample.task import train_fedavg as train_fn

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = ResNet1d(n_classes=1)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    partitioning = context.run_config["partitioning"]
    learning_rate = msg.content["config"]["lr"]
    local_epochs = context.run_config["local-epochs"]

    # Call the training function
    start_time = time.time()
    # call load_datasets in train_fn
    # save the model (given as param in train_fn) as a writable torch .pt file, 
    # and the train_fn will be saving and updating the model there.
    # and upon calling train_fn, it will be loading what is existing
    model_path = f"output-client{partition_id}.pt"
    with open(f"output-client{partition_id}.txt", "w") as f:
        train_loss = subprocess.run(
            f'OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=4 task.py --model_path {model_path} --partitioning {partitioning} --num_clients {num_partitions} --partition_id {partition_id} --batch_size {batch_size} --epochs {local_epochs} --learning_rate {learning_rate}'.split(),
            stdout=f,
            text=True
        )
    end_time = time.time()
    training_time = end_time - start_time

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
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

    # Load the model and initialize it with the received weights
    model = ResNet1d(n_classes=1)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_datasets(partition_id, num_partitions, batch_size)

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
