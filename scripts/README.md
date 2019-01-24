# Shell scripts

We provide several shell scripts for easy managing the experiments.

| File               | Description                                       |
|--------------------|---------------------------------------------------|
| `download_data.sh` | Download the training data                        |
| `process_data.sh`  | Process the data and save it to shared memory     |
| `setup_exp.sh`     | Set up a new experiment with the default settings |
| `run_train.sh`     | Train a model                                     |

> __Below we assume the working directory is the repository root.__

## Download the training data

```sh
./scripts/download_data.sh
```

This command will download the training data to the default data directory
(`./data/`).

## Process the data and save it to shared memory

```sh
./scripts/process_data.sh
```

This command will store the training data to shared memory using SharedArray
package.

## Set up a new experiment with the default settings

```sh
./scripts/setup_exp.sh -n "Some notes" -r 1 "./exp/my_experiment/"
```

This command will create a new experiment directory at the given path
(`"./exp/my_experiment/"`), copy the default configuration file to that folder
and save the experiment note (`"Some notes"`) as a text file in that folder.

## Train a model

```sh
./scripts/run_train.sh -g "0" "./exp/my_experiment/"
```

This command will look for the configuration file in the given experiment
directory (`"./exp/my_experiment/"`) and train a model according to the
configurations on the specified GPU (`"0"`).
