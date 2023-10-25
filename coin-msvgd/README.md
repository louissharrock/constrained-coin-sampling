# Coin MSVGD

## Description

This folder contains code for implementing Coin MSVGD.

## Requirements
The main software requirements are as follows.
```
python >= 3.8
R >= 4.0.4 
```

The following should also be installed, e.g., via `pip`.
```
rpy2 >= 3.4.4
tensorflow >= 2.4.0
tensorflow_probability >= 0.12
numpy >= 1.19.5
scipy >= 1.6.3
matplotlib >= 3.4.1
pandas >= 1.2.4
seaborn >= 0.11.1
scikit-learn >= 0.24.2
tqdm
absl-py
```

## Experiments
The results for Coin MSVGD, MSVGD, and SVMD in the paper can be 
reproduced as follows.

#### Simplex Targets

* The sparse Dirichlet posterior experiment is contained in [`dirichlet.py`](https://github.com/louissharrock/constrained-coin-sampling/blob/main/coin-msvgd/dirichlet.py)
* The quadratic simplex experiment is contained in [`quadratic.py`](https://github.com/louissharrock/constrained-coin-sampling/blob/main/coin-msvgd/quadratic.py)

#### Post-Selection Inference

* The 2D selection inference experiment is contained in [`selection_inference_2d.py`](https://github.com/louissharrock/constrained-coin-sampling/blob/main/coin-msvgd/selection_inference_2d.py)
* The selection inference coverage experiments is contained in [`selection_inference_coverage.py`](https://github.com/louissharrock/constrained-coin-sampling/blob/main/coin-msvgd/selection_inference_coverage.py)
* The selection inference HIV experiment is contained in [`selection_inference_hiv.py`](https://github.com/louissharrock/constrained-coin-sampling/blob/main/coin-msvgd/selection_inference_hiv.py). The dataset for this experiment can be downloaded from the [Stanford HIV Drug Resistance Database](http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt).

## Acknowledgements

Our implementation of Coin MSVGD is built on top of the official 
implementation of MSVGD. We gratefully acknowledge the authors
of this paper for their open source code:
* J. Shi, C. Liu and L. Mackey. Sampling with Mirrored Stein Operators. ICLR, 2022. [[Paper](https://arxiv.org/abs/2106.12506)] | [[Code](https://github.com/thjashin/mirror-stein-samplers)].

```
@article{shi2021sampling,
  title={Sampling with Mirrored {S}tein Operators}, 
  author={Jiaxin Shi and Chang Liu and Lester Mackey},
  journal={International Conference on Learning Representations},
  year={2022}
}
```


