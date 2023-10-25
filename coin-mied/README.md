# Coin MIED

## Description
This folder contains code for implementing Coin Mollified Interaction Energy Descent (Coin MIED).

## Installation
To install the package, use `pip install -e .` or `conda develop .`

## Requirements
The full list of dependencies are listed in `requirements.txt`.

## Experiments
The results for Coin MIED in the paper can be reproduced as follows.

#### Uniform Target

- In `results/uniform`, run [`uniform_script.py`](https://github.com/louissharrock/constrained-coin-sampling/blob/main/coin-mied/results/uniform/uniform_script.py) to generate all of the results in the paper.


#### Fairness Bayesian Neural Network

- In `results/fairness_bnn`, [`fairness_bnn_script`](https://github.com/louissharrock/constrained-coin-sampling/blob/main/coin-mied/results/fairness_bnn/fairness_bnn_script.sh) contains an example
script used to generate results for Coin MIED and MIED, for a particular
value of the constraint parameter t. The complete set of results for Coin MIED and MIED can be reproduced by
running `fairness_bnn_script` for all values of t in the paper. The dataset for this experiment can be downloaded from the [UCI Machine Learning Repositorry](https://archive.ics.uci.edu/dataset/2/adult). 



## Acknowledgements

Our implementation of Coin MIED is built on top of the official 
implementation of MIED. We gratefully acknowledge the authors
of this paper for their open source code:
* L. Li, Q. Liu, A. Korba, M. Yurochkin and J. Solomon. Sampling with Mollified Interaction Energy Descent. ICLR, 2023. [[Paper](https://arxiv.org/abs/2210.13400)] | [[Code](https://github.com/lingxiaoli94/MIED)].

```
@article{Li2023,
  title={Sampling with Mollified Interaction Energy Descent}, 
  author={Lingxiao Li and Qiang Liu and Anna Korba and Mikhail Yurochkin and Justin Solomon},
  journal={International Conference on Learning Representations},
  year={2023}
}
```







