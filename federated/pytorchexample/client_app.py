"""pytorchexample: A Flower / PyTorch app."""

import torch, time
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import ResNet1d, load_datasets
from pytorchexample.task import test as test_fn
from pytorchexample.task import train_fedavg as train_fn
from datetime import date

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Load the model and initialize it with the received weights
    #model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model_path = f"output-client{context.node_config["partition-id"]}.pt"
    partition_model = {"MODEL_STATE": msg.content["arrays"].to_torch_state_dict()}
    torch.save(partition_model, model_path)

    model = ResNet1d(n_classes=1)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Call the training function
    start_time = time.time()
    train_loss, model, train_data_size = train_fn({
        "net": model,
        "partition_id": context.node_config["partition-id"],
        "num_partitions": context.node_config["num-partitions"],
        "partitioning": context.run_config["partitioning"],
        "val": context.run_config["val"],
        "epochs": context.run_config["local-epochs"], 
        "lr": msg.content["config"]["lr"], 
        "batch_size": context.run_config["batch-size"]
    })
    end_time = time.time()

    training_time = end_time - start_time

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": train_data_size,
        "training_time": training_time,
        "local-epochs": context.run_config["local-epochs"],
        "partition_id": context.node_config["partition-id"]
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    today = date.today()

    with open(f'{today}-clients{context.node_config["num-partitions"]}-partitioning{context.run_config["partitioning"]}-commrounds{context.run_config["num-server-rounds"]}-loceps{context.run_config["local-epochs"]}.txt', "a") as logger:
        logger.write(f"{str(dict(metric_record))}\n")

    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    partitioning = context.run_config["partitioning"]
    val = context.run_config["val"]
    trainloader, valloader = load_datasets(partition_id, num_partitions, batch_size, partitioning=partitioning, val=val)
    
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
    today = date.today()

    with open(f'{today}-clients{context.node_config["num-partitions"]}-partitioning{context.run_config["partitioning"]}-commrounds{context.run_config["num-server-rounds"]}-loceps{context.run_config["local-epochs"]}.txt', "a") as logger:
        logger.write(f"{str(dict(metric_record))}\n")

    return Message(content=content, reply_to=msg)
