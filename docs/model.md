# DANTest

## Dicriminative Adversarial Networks (DANs)

Discriminative adversarial networks (DANs) [1] are essentially conditional GANs
(see [here](background)) where both the generator and the discriminator are
discriminative models.

<img src="figs/system.png" alt="system" style="max-width:500px;">

Based on DANs, we propose a new, simple framework, dubbed _DANTest_ for
systematically comparing different adversarial losses.

1. Build several DANs. For each of them, the generator 𝐺 takes as input a real
   sample and outputs a fake label. The discriminator takes as input a real
   sample with either its true label, or a fake label made by 𝐺, and outputs a
   scalar indicating if the "sample–label" pair is real.
2. Train the DANs with different component loss functions, regularization
   approaches or hyperparameters.
3. Predict the labels of test data by the trained models.
4. Compare the performance of different models with standard evaluation metrics
   used in supervised learning.

The DANTest is simple and it is easy to control and extend, which allows us to
easily evaluate new adversarial losses. With the DANTest, we are able to
conduct a extensive comparative study on different adversarial losses (168 in
total) to see how different adversarial losses perform against one another.

Specifically, we consider 10 existing component functions, 2 new component
functions propose in light of our theoretical analysis, along with 14 different
regularization approaches. Moreover, we use the DANTest to empirically study
the effect of the Lipschitz constant, penalty weights momentum terms, and other
hyperparameters. Please refer to our paper for the results.
