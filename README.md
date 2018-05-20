# ams_aggmo_tests
Testing out the combination of learning algorithms [aggregated momentum (AggMo)](https://arxiv.org/abs/1804.00325) and [Amsgrad](https://openreview.net/forum?id=ryQu7f-RZ) in [pytorch](https://pytorch.org/).

Compare 4 learning algorithms: 

1. Stochastic Gradient Descent with Nesterov Momentum (SGD-N) 
2. Amsgrad
3. Amsgrad + AggMo (AmsAggMo) 
4. AggMo 

All learning algorithms trained a 5 layer convlutional neural network to classify the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.


## Step 1 (hyper parameter search)
Searched over learning rates {0.01,0.005,0.001,0.0005,0.0001}

And what I called betamax {0.9,0.99,0.999}

For SGD-N, this was simply the momentum term.

For Amsgrad this was the beta1 term. beta2 was kept fixed at 0.999.

For AmsAggMo and AggMo this was the maximum momentum in the list of aggregated momentum terms. 
e.g. betamax 0.99 gives the aggregated momentum terms {0,0.9,0.99}. 0 term was added as per the suggestion in the AggMo paper.

Most of these combinations were tested. The networks trained for a total of 10 epochs over the entire dataset. After which the test accuracy was determined.

Here are the results:

N/A - Did not perform these experiments. Because of the insight gained from experiments 1 and 2 on stability, I did not waste time training AggMo and SGD-N on 4, 5 and 6.

exp#|     hyper parameters         | SGD-N  | Amsgrad | AmsAggMo | AggMo |
:---|:---------------------------- |:------:|:-------:| :-------:| :---: |
1   |  bmax = 0.999, lr = 0.0001   |12%     |60%      |58%       |61%    |
2   |  bmax = 0.999, lr = 0.0005   |10%     |62%      |**65%**   |10%    |
3   |  bmax = 0.999, lr = 0.001    |N/A     |62%      |62%       |N/A    |
4   |  bmax = 0.999, lr = 0.005    |N/A     |33%      |48%       |N/A    |
5   |  bmax = 0.999, lr = 0.01     |N/A     |10%      |10%       |N/A    |
6   |  bmax = 0.99, lr = 0.0001    |**63%** |59%      |60%       |**64%**|
7   |  bmax = 0.99, lr = 0.0005    |36%     |**64%**  |**64%**   |60%    |
8   |  bmax = 0.99, lr = 0.001     |22%     |62%      |61%       |50%    |
9   |  bmax = 0.99, lr = 0.005     |10%     |38%      |44%       |10%    |
10  |  bmax = 0.99, lr = 0.01      |10%     |10%      |10%       |10%    |
11  |  bmax = 0.9, lr = 0.0001     |57%     |60%      |57%       |49%    |
12  |  bmax = 0.9, lr = 0.0005     |**63%** |**64%**  |63%       |62%    |
13  |  bmax = 0.9, lr = 0.001      |62%     |61%      |62%       |**63%**|
14  |  bmax = 0.9, lr = 0.005      |40%     |39%      |38%       |53%    |
15  |  bmax = 0.9, lr = 0.01       |62%     |61%      |62%       |63%    |

group 1 files are the experiments for step 1.

## Step 2
After finding the two networks with the highest accuracy for all four algorithms, I trained these combinations for 50 epochs over the dataset.

Highest accuracy for each learning algorithm after 50 epochs.

SGD-N | Amsgrad | AmsAggMo | AggMo |
:---: | :-----: | :------: | :---: |
59.6%|60.33%|59.65%|60.18%|60.56%|


group 2 files are the experiments for step 2.


## Remarks
It seems like the networks have begun to over fit by the time we get to 50 epochs. Should probably use regularization.

One thing I noticed was that AmsAggMo was able to stay stable with high damping parameters *and* high learning rates. Interesting stuff.

Unfortunately, because of the way AmsAggMo is implemented now, it takes much longer per-epoch compared to the other four algorithms.
