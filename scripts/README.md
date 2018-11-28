# Preparing Training Data

## Download the MNIST Handwritten Digit Database

```sh
./download_mnist.sh
```

This will download the MNIST handwritten digit database to the current working
directory.

## Load the Training Data to SharedArray

> Make sure the SharedArray package has been installed.

Run

```sh
python ./load_mnist_to_sa.py ./mnist/ --merge --binary
```

This will load and binarize the MNIST digits and save them to shared memory via
SharedArray package.

# Shell scripts

We provide several shell scripts for easy managing the experiments.

| File               | Description                                   |
|--------------------|-----------------------------------------------|
| `download_data.sh` | Download the training data                    |
| `process_data.sh`  | Save the training data to shared memory       |
| `setup_exp.sh`     | Set up a new experiment with default settings |
| `run_train.sh`     | Train a model                                 |
| `rerun_train.sh`   | Rerun the training                            |

> __Below we assume the working directory is the repository root.__

## Download the training data

```sh
./scripts/download_data.sh
```

This command will download the training data to the default data directory
(`./data/`).

## Save the training data to shared memory

```sh
./scripts/process_data.sh
```

This command will store the training data to shared memory using SharedArray
package.

## Set up a new experiment with default settings

```sh
./scripts/setup_exp.sh "./exp/my_experiment/" "Some notes"
```

This command will create a new experiment directory at the given path
(`"./exp/my_experiment/"`), copy the default configuration and model parameter
files to that folder and save the experiment note (`"Some notes"`) as a text
file in that folder.

## Train a model

```sh
./scripts/run_train.sh "./exp/my_experiment/" "0"
```

This command will look for the configuration and model parameter files in the
given experiment directory (`"./exp/my_experiment/"`) and train a model according
to the configurations and parameters on the specified GPU (`"0"`).

## Rerun the training

```sh
./scripts/rerun_train.sh "./exp/my_experiment/" "0"
```

This command will remove everything in the experiment directory except the
configuration and model parameter files and then rerun the experiment (train,
inference and interpolation) on the specified GPU (`"0"`).
