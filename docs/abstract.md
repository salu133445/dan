# Abstract

Recent work has proposed various adversarial losses for training generative
adversarial networks. Yet, it remains unclear what certain types of functions
are valid adversarial loss functions, and how these loss functions perform
against one another. In this paper, we aim to gain a deeper understanding of
adversarial losses by decoupling the effects of their component functions and
regularization terms. We first derive some necessary and sufficient conditions
of the component functions such that the adversarial loss is a divergence-like
measure between the data and the model distributions. In order to systematically
compare different adversarial losses, we then propose DANTest—a new, simple
framework based on discriminative adversarial networks. With this framework, we
evaluate an extensive set of adversarial losses by combining different component
functions and regularization approaches. This study leads to some new insights
into the adversarial losses.
