# A Novel Evaluation Metric for Deep Learning-Based Side-Channel Analysis and Its Extended Application to Imbalanced Data

The field of side channel analysis (SCA) has increasingly been using Deep Learning
to enhance the attacks. However, attack evaluation metrics, like Guessing Entropy
(GE) and Success Rate (SR), are computationally inefficient. Furthermore, traditional
Deep Learning metrics, like accuracy and precision, are not suitable for evaluating Deep
Learning models in the context of SCA, as classes are often imbalanced. However,
recently Zhang et al. have proposed a new evaluation metric for SCA, Cross Entropy
Ratio (CER), that provides a good indication of the success of the attack and is viable
to embed in Deep Learning Algorithms. Additionally this metric can be used as a loss
function to train better models when training data is imbalanced. Throughout this
report, a reproduction of the results of the paper introducing CER will be showcased,
and a self-developed metric, the Log-Likelihood Ratio (LLR), will be introduced as
well. These two metrics were compared to Cross Entropy (CROSS) and each other
as loss functions, using several neural network architectures and data-sets. The final
results of this report will showcase that CER, as a loss function, in the context of
SCA, when classes are imbalanced, is better than using regular Cross Entropy. LLR
performs slightly worse than CER in almost every scenario, but is generally better
than Cross Entropy. Therefore, this report shows that the results from Zhang et al.
are reproducible and the CER metric can be used as a evaluation metric to more accurately evaluate Deep Learning models.

## Setup
One should first run the setup.sh bash script after cloning this repository, to make sure the python code can run successfully. It does this by pulling all the datasets needed (so tweak according to your configuration) and then removing any potential result folders.

After that you can run this projects `__main__.py` file. This will execute all the training for all the models in the configuration and store their results in the correct folders.
