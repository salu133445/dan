# DANTest

Source code for "Towards a Deeper Understanding of Adversarial Losses under a Discriminative Adversarial Network Setting"

## Prerequisites

> __Below we assume the working directory is the repository root.__

### Install dependencies

- Using pipenv (recommended)

  > Make sure `pipenv` is installed. (If not, simply run `pip install --user pipenv`.)

  ```sh
  # Install the dependencies
  pipenv install
  # Activate the virtual environment
  pipenv shell
  ```

- Using pip

  ```sh
  # Install the dependencies
  pip install -r requirements.txt
  ```

### Prepare training data

```sh
# Download the training data
./scripts/download_data.sh
# Store the training data to shared memory
./scripts/process_data.sh
```

You can also download the MNIST handwritten digit database manually
[here](http://yann.lecun.com/exdb/mnist/).

## Scripts

We provide several shell scripts for easy managing the experiments. (See
`scripts/README.md` for a detailed documentation.)

> __Below we assume the working directory is the repository root.__

### Train a new model

1. Run the following command to set up a new experiment with default settings.

   ```sh
   # Set up a new experiment (for one run only)
   ./scripts/setup_exp.sh -r 1 "./exp/my_experiment/"
   ```

2. Modify the configuration files for different experimental settings. The
   configuration file can be found at `./exp/my_experiment/config.yaml`.

3. Train the model by running the following command.

     ```sh
     # Train the model (on GPU 0)
     ./scripts/run_train.sh -c -g 0 "./exp/my_experiment/"
     ```

## Outputs

For each run, there will be three folders created in the experiment folder.

- `logs/`: contain all the logs created
- `model/`: contain the trained model
- `src/`: contain a backup of the source code

Note that the _validation results_ can be found in the `logs/` folder.

## Paper

__Towards a Deeper Understanding of Adversarial Losses under a Discriminative Adversarial Network Setting__<br>
Hao-Wen Dong and Yi-Hsuan Yang<br>
_arXiv preprint arXiv:1901.08753_, 2019.<br>
[[website](https://salu133445.github.io/dan/)]
[[paper](https://salu133445.github.io/dan/pdf/dan-arxiv-paper.pdf)]
[[arxiv](https://arxiv.org/abs/1901.08753)]
[[code](https://github.com/salu133445/dan)]
