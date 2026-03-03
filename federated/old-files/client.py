from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from pytorchexample.model import load_data, load_model, train_loop, eval_loop
import flwr as fl, torch

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    local_epochs = 1
    
    """Train the model on local data."""
    # Load the model and initialize it with the received weights
    model = load_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data()#partition_id, num_partitions, batch_size)

    # Call the training function
    for epoch in trange(1, local_epochs + 1):  
        train_loss = train_loop(
            epoch, 
            train_dataloader, net, 
            optimizer, 
            loss_function, 
            #local_lr,
            device)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict()) # weights of the local model
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data()#partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_pr_auc, eval_roc_auc = eval_loop(epoch, valid_dataloader, net, loss_function, device)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_pr_auc,
        "eval_roc_auc": eval_roc_auc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)