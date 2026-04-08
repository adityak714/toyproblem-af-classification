---
tags: [ecg, af-classification, resnet]
dataset: [CODE15]
framework: [torch]
---

# FL-ECG - Federated Learning for Automated ECG Classification with the CODE15% Dataset

### Fetch the app

Install Flower (and Ray):

```shell
pip install flwr
pip install -U "ray[default]"
```

Fetch the app:

```shell
flwr new @flwrlabs/quickstart-pytorch
```

This will create a new directory called `quickstart-pytorch` with the following structure:

```shell
quickstart-pytorch
├── pytorchexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `pytorchexample` package.

```bash
pip install -e .
```

Or download from the `requirements-final.txt` file.

```bash
pip install -r requirements-final.txt
```

## Set up + running the project

```bash
cd 1-starter-ecg-model/
python3 -m venv venv
pip install -r ../requirements-final.txt
source venv/bin/activate
```

```bash
# -- dashboard can be viewed as a webpage at localhost:8265 given ray[default] is installed by pip
ray start --include-dashboard --head --temp-dir=$HOME/storage
ray start --address="10.21.30.<>:6379"
ln -s $HOME/.flwr/config.toml 1-starter-ecg-model/federated/config.toml
cat pyproject.toml # project-local .toml file, where contextual config (num. local epochs, batch_size, partitioning_strategy, lr, communication rounds ...) & num_gpu and num_cpu
cat config.toml # here the number of clients is controlled -- symlink to $HOME/.flwr/config.toml, if this is edited, that file also gets edited. 
## Keep num_gpu and num_cpu same with pyproject.toml as a safety step to avoid inconsistent config loading.
```

Finally, run:

```bash
export RAY_ADDRESS="10.21.30.<>:6379" # this must point to the HEAD node of the Ray setup
flwr run # run this in the same dir as the toml files.
```

> **Common errors: the data files not found.**
> Ray may see the root dir to be where the Python virtual environment is. Flower Simulator (when running `flwr run ...`) may see the root dir as where the toml files are. In `1-starter-ecg-model/federated/pytorchexample`, the files for the server, client and the task are present. Modify the os path directory in `task.py` if no files get loaded. Inspect what the working directory is at the place where the data partitioning and loading happens, in `load_datasets()`, by printing `os.getcwd()` to the console.

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!TIP]
> This example runs faster when the `ClientApp`s have access to a GPU. If your system has one, you can make use of it by configuring the `backend.client-resources` component in your Flower Configuration. Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
# Run with the default federation (CPU only)
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"
```

> [!TIP]
> For a more detailed walk-through check our [quickstart PyTorch tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
