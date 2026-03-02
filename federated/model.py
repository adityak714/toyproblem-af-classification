import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, random_split, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import h5py
import pandas as pd
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt

# set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# choose variables
"""
TASK: Adapt the following hyperparameters if necessary
"""
learning_rate = 1e-4
weight_decay = 1e-1
num_epochs = 10
batch_size = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CustomCNN(nn.Module):
    def __init__(self,):
        super(CustomCNN, self).__init__()
        self.kernel_size = 3
        # conv layer
        downsample = self._downsample(4096, 1024)
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=32,
                               kernel_size=self.kernel_size,
                               stride=downsample, padding=self._padding(downsample),
                               bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        downsample_2 = self._downsample(1024, 512)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32,
                               kernel_size=self.kernel_size,
                               stride=downsample_2, padding=self._padding(downsample_2),
                               bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        downsample_3 = self._downsample(512, 256)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32,
                               kernel_size=self.kernel_size,
                               stride=downsample_3,padding=self._padding(downsample_3),
                               bias=False)
        self.bn3 = nn.BatchNorm1d(num_features=32)
        # ReLU
        self.relu = nn.ReLU()
        # linear layer
        self.lin = nn.Linear(in_features=256*32, out_features=1)

    def _padding(self, downsample):
        return max(0, int(np.floor((self.kernel_size - downsample + 1) / 2)))

    def _downsample(self, seq_len_in, seq_len_out):
        return int(seq_len_in // seq_len_out)

    def forward(self, x):
        x= x.transpose(2,1)
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        x = self.bn3(self.conv3(x))
        x = self.relu(x)
        x_flat = x.view(x.size(0), -1) # flatten
        x = self.lin(x_flat)
        return x

def train_loop(epoch, dataloader, model, optimizer, loss_function, device):
    # model to training mode (important to correctly handle dropout or batchnorm layers)
    model.train()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    # progress bar def
    train_pbar = tqdm(dataloader, desc="Training Epoch {epoch:2d}".format(epoch=epoch), leave=True)
    train_correct = []
    sigmoid = torch.nn.Sigmoid().to(device)
    # training loop
    for traces, diagnoses in train_pbar:
        # data to device (CPU or GPU if available)
        traces, diagnoses = traces.to(device), diagnoses.to(device)
        """
        TASK: Insert your code here. This task can be done in 5 lines of code.
        """
        for x,y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            curr_loss = loss_function(pred, y)
            with torch.no_grad():
                train_correct.append((sigmoid(pred).argmax(1) == y).sum().item())
            curr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Update accumulated values
        total_loss += curr_loss.detach().cpu().numpy()
        n_entries += len(traces)
        # Update progress bar
        train_pbar.set_postfix({'loss': total_loss / n_entries})
        
    train_pbar.close()
    return total_loss / n_entries

def eval_loop(epoch, dataloader, model, loss_function, device):
    # model to evaluation mode (important to correctly handle dropout or batchnorm layers)
    model.eval()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    avg_precisions = []  # accumulated predicted probabilities
    roc_s = [] # accumulated true labels
    # progress bar def
    eval_pbar = tqdm(dataloader, desc="Evaluation Epoch {epoch:2d}".format(epoch=epoch), leave=True)
    sigmoid = torch.nn.Sigmoid().to(device)
    # evaluation loop
    for traces_cpu, diagnoses_cpu in eval_pbar:
        # data to device (CPU or GPU if available)
        traces, diagnoses = traces_cpu.to(device), diagnoses_cpu.to(device)
        """
        TASK: Insert your code here. This task can be done in 6 lines of code.
        """
        with torch.no_grad():
            for x,y in dataloader:
                xt, yt = x.to(device), y.to(device)
                pred = model(xt)
                curr_loss = loss_function(pred, yt)
                
                avg_precisions.append(average_precision_score(yt.cpu(), sigmoid(pred).cpu()))
                roc_s.append(roc_auc_score(yt.cpu(), sigmoid(pred).cpu()))
                # valid_true.append(yt.detach().cpu()) # save all of the true labels here instead

            # Update accumulated values
            total_loss += curr_loss.detach().cpu().numpy()
            n_entries += len(traces)
        # Update progress bar
        eval_pbar.set_postfix({
            'loss': total_loss / n_entries, 
            'avg_precision': np.mean(avg_precisions), 
            'avg_roc_auc': np.mean(roc_s)
        })
    eval_pbar.close()
    return total_loss / n_entries, np.mean(avg_precisions), np.mean(roc_s)

def load_data():
    #hdf5_list, csv_list = glob.glob("data/code15-12l/exams_part*.hdf5"), glob.glob("data/code15-12l/exams_part*.hdf5.csv")
    #hdf5_list = hdf5_list.pop("data/code15-12l/exams_part0.hdf5")
    #csv_list = csv_list.pop("data/code15-12l/exams_part0.hdf5.csv")
    dataset = None
    for i in range(1,10):
        # exams_part0 --> holdout
        path_to_h5_train, path_to_csv_train = f'data/code15-12l/exams_part{i}.hdf5', f'data/code15-12l/exams_part{i}.hdf5.csv'
        # load traces
        traces = torch.tensor(h5py.File(path_to_h5_train, 'r')['tracings'][()], dtype=torch.float32)
        df = pd.read_csv(path_to_csv_train)
        # load labels
        labels = torch.tensor(np.array(df['AF']), dtype=torch.float32, device=torch.device("cuda")).reshape(-1,1)
        print(len(traces), len(labels))
        # load dataset
        dataset = TensorDataset(traces, labels) if dataset is None else torch.cat((dataset, TensorDataset(traces, labels)), 0)
        #n_classes = len(torch.unique(labels))
        len_dataset = len(dataset)
        print("size of aggregated dataset >>>", len_dataset)

    dataset_train, dataset_valid = random_split(dataset, lengths=[0.7,0.3])
    
    # build data loaders
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader

def load_holdout():
    # exams_part0 --> holdout
    path_to_h5_train, path_to_csv_train = 'data/code15-12l/exams_part0.hdf5', 'data/code15-12l/exams_part0.hdf5.csv'
    # load traces
    traces = torch.tensor(h5py.File(path_to_h5_train, 'r')['tracings'][()], dtype=torch.float32)
    df = pd.read_csv(path_to_csv_train)
    # load labels
    labels = torch.tensor(np.array(df['AF']), dtype=torch.float32, device=torch.device("cuda")).reshape(-1,1)
    # load dataset
    dataset = TensorDataset(traces, labels)
    len_dataset = len(dataset)
    n_classes = len(torch.unique(labels))
    
    # build data loaders
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def load_model(device=torch.device("cuda")):
    model = CustomCNN().to(device=device)
    return model

if __name__ == "__main__":
    # Set device
    model = load_model()
    train_dataloader, valid_dataloader = load_data()
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = None
    
    best_loss = np.inf
    # allocation
    train_loss_all, valid_loss_all = [], []
    
    # loop over epochs
    for epoch in trange(1, num_epochs + 1):
        # training loop
        train_loss = train_loop(epoch, train_dataloader, model, optimizer, loss_function, device)
        # validation loop
        valid_loss, avg_precision, avg_roc_auc = eval_loop(epoch, valid_dataloader, model, loss_function, device)
    
        # collect losses
        train_loss_all.append(train_loss)
        valid_loss_all.append(valid_loss)
    
        # compute validation metrics for performance evaluation
        ###
        ### TASK: compute validation metrics (e.g. AUROC); Insert your code here
        ### This can be done e.g. in 5 lines of code
        ###
        # y_pred, y_true <-- DO SOMETHING FOR THESE!!
    
        # save best model: 
        # here we save the model only for the lowest validation loss
        if valid_loss < best_loss:
            # Save model parameters
            torch.save({'model': model.state_dict()}, 'model.pth')
            # Update best validation loss
            best_loss = valid_loss
            # statement
            model_save_state = "Best model -> saved"
        else:
            model_save_state = ""
    
        # Print message
        tqdm.write('Epoch {epoch:2d}: \t'
                    'Train Loss {train_loss:.6f} \t'
                    'Valid Loss {valid_loss:.6f} \t'
                    '{model_save}'
                    .format(epoch=epoch,
                            train_loss=train_loss,
                            valid_loss=valid_loss,
                            model_save=model_save_state))
    
        # Update learning rate with lr-scheduler
        if lr_scheduler:
            lr_scheduler.step()
        
    ###
    ### TASK: Here it can make sense to plot your learning curve; Insert your code here
    ###
    plt.plot(np.arange(num_epochs), train_loss_all)
    plt.plot(np.arange(num_epochs), valid_loss_all)
    plt.show()