# Results

## Experimental Settings

For all the experiments, we used the Adam optimizers and ReLUs as the activation
functions for both the generator and discriminator. Other settings are
summarized in the following table.

| parameter                         | value | &emsp; | parameter       | value |
|:---------------------------------:|:-----:|:------:|:---------------:|:-----:|
| batch size                        | 64    |        | learning rate   | 0.002 |
| number of epochs                  | 20    |        | _β_<sub>1</sub> | 0.5   |
| weight clipping threshlod in WGAN | 0.01  |        | _β_<sub>2</sub> | 0.9   |
| GP coefficient in WGAN-GP         | 10.0  |        | _ε_             | 1e-8  |

## Full Table of Test Accuracies

In the following tables, we report the highest accuracy on the test set that the
model have achieved during the training.

| Model   | G (hidden norm) | G (out norm) | G (out act) | D (norm) | Accuracy |
|:-------:|:---------------:|:------------:|:-----------:|:--------:|:--------:|
| GAN-GP  | BN              | BN           | softmax     | LN       | 0.9581   |
| WGAN-GP | BN              | BN           | softmax     | LN       | 0.9557   |
| GAN-GP  | BN              | BN           | softmax     | X        | 0.9522   |
| WGAN-GP | BN              | X            | softmax     | X        | 0.9489   |
| WGAN-GP | LN              | X            | softmax     | X        | 0.9466   |
| WGAN-GP | BN              | BN           | softmax     | X        | 0.9353   |
| WGAN-GP | LN              | LN           | softmax     | X        | 0.9340   |
| WGAN-GP | BN              | X            | X           | X        | 0.9253   |
| WGAN-GP | BN              | BN           | X           | X        | 0.9252   |
| WGAN-GP | X               | X            | softmax     | X        | 0.9218   |
| WGAN-GP | LN              | LN           | softmax     | LN       | 0.9178   |
| WGAN-GP | LN              | X            | softmax     | LN       | 0.9072   |
| WGAN-GP | X               | X            | X           | X        | 0.8914   |
| WGAN-GP | BN              | BN           | softmax     | BN       | 0.8615   |
| WGAN    | BN              | BN           | softmax     | LN       | 0.8391   |
| WGAN    | BN              | BN           | softmax     | X        | 0.2898   |

_Objectives_&mdash;__GAN-GP__: non-saturating GAN with gradient penalties;
__WGAN__: Wasserstein GAN with weight clipping; __WGAN-GP__: Wasserstein GAN
with gradient penalties. _Normalizations_&mdash;__BN__: batch normalization;
__LN__: layer normalization.

<img src="figs/test_acc.png" alt="test_accuracy" style="width:100%;">

<img src="figs/test_acc_closeup.png" alt="test_accuracy_closeup" style="width:100%;">

## Training Accuracies

We also plotted the accuracy for each training batch along the training process.
As we can see from the following two graphs, the training curve has a similar
trend as the corresponding test curve.

<img src="figs/train_acc.png" alt="train_accuracy" style="width:100%;">

<img src="figs/train_acc_closeup.png" alt="train_accuracy_closeup" style="width:100%;">

## Training Objectives

| Model   | G (hidden norm) | G (out norm) | G (out act) | D (norm) | Accuracy |
|:-------:|:---------------:|:------------:|:-----------:|:--------:|:--------:|
| GAN-GP  | BN              | BN           | softmax     | LN       | 0.9581   |
| WGAN-GP | BN              | BN           | softmax     | LN       | 0.9557   |
| GAN-GP  | BN              | BN           | softmax     | X        | 0.9522   |
| WGAN-GP | BN              | BN           | softmax     | X        | 0.9353   |
| WGAN    | BN              | BN           | softmax     | LN       | 0.8391   |
| WGAN    | BN              | BN           | softmax     | X        | 0.2898   |

![train_objectives](figs/train_objectives.png)

We can see that GAN-GP outperforms WGAN-GP, where both performs better than
WGAN.

## Normalization in Generator

| Model   | G (hidden norm) | G (out norm) | G (out act) | D (norm) | Accuracy |
|:-------:|:---------------:|:------------:|:-----------:|:--------:|:--------:|
| WGAN-GP | BN              | BN           | softmax     | X        | 0.9353   |
| WGAN-GP | BN              | BN           | X           | X        | 0.9252   |
| WGAN-GP | X               | X            | softmax     | X        | 0.9218   |
| WGAN-GP | X               | X            | X           | X        | 0.8914   |
| WGAN-GP | LN              | LN           | softmax     | X        | 0.9340   |

![normalization_in_gen](figs/norm_in_g.png)

We can see that adopting batch normalization in the generator improves the
performance.

## Output Normalization and Activation in Generator

| Model   | G (hidden norm) | G (out norm) | G (out act) | D (norm) | Accuracy |
|:-------:|:---------------:|:------------:|:-----------:|:--------:|:--------:|
| WGAN-GP | BN              | X            | softmax     | X        | 0.9489   |
| WGAN-GP | BN              | BN           | softmax     | X        | 0.9353   |
| WGAN-GP | BN              | X            | X           | X        | 0.9253   |
| WGAN-GP | BN              | BN           | X           | X        | 0.9252   |

![output_normalization_activation_in_gen](figs/out_norm_act_in_g.png)

We can see that, for the output layer of the generator, using no normalization
performances better than using batch normalization and that using softmax
activations achieves better performance than using no activation. Using no
normalization with softmax activation achieves the best performance.

## Normalization in Discriminator

| Model   | G (hidden norm) | G (out norm) | G (out act) | D (norm) | Accuracy |
|:-------:|:---------------:|:------------:|:-----------:|:--------:|:--------:|
| WGAN-GP | BN              | BN           | softmax     | LN       | 0.9557   |
| WGAN-GP | BN              | BN           | softmax     | X        | 0.9353   |
| WGAN-GP | BN              | BN           | softmax     | BN       | 0.8615   |

![normalization_in_dis](figs/norm_in_d.png)

We can see that for discriminator using layer normalization improves the
performance, while using batch normalization results in a reduction in
performance. We can also see from the following graph that adopting layer
normalization speeds up the training, while the performance might drop
significantly after its highest point.
