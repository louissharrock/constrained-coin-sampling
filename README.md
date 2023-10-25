## <p align="center">Learning Rate Free Bayesian Inference in Constrained Domains<br><br>NeurIPS 2023<br></p>

<div align="center">
  <a href="https://louissharrock.github.io/" target="_blank">Louis&nbsp;Sharrock</a> &emsp; <b>&middot;</b> &emsp;
<a href="https://web.stanford.edu/~lmackey/" target="_blank">Lester Mackey</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://chris-nemeth.github.io/" target="_blank">Christopher&nbsp;Nemeth</a> &emsp; </b>
</div>

## Description

This repository contains code to reproduce the results of the numerical experiments contained in [Sharrock et al. (2023)](https://arxiv.org/abs/2305.14943). The code for each of two algorithms can be found in separate directories: 
* The code for Coin MSVGD is contained in [``coin-msvgd``](https://github.com/louissharrock/constrained-coin-sampling/tree/main/coin-msvgd).
* The code for Coin MIED is contained in [``coin-mied``](https://github.com/louissharrock/constrained-coin-sampling/tree/main/coin-mied).

## Citation

If you find the code in this repository useful for your own research, 
please consider citing our paper:

```bib
@InProceedings{Sharrock2023,
  title = 	 {Learning Rate Free Bayesian Inference in Constrained Domains},
  author =       {Sharrock, Louis and Mackey, Lester and Nemeth, Christopher},
  booktitle = 	 {Proceedings of The 37th Conference on Neural Information Processing Systems},
  year =         {2023},
  city =         {New Orleans, LA},
}
```

## Acknowledgements

Our implementations of Coin MSVGD and Coin MIED are based on existing 
implementations of MSVGD and MIED. We gratefully acknowledge the authors
of the following papers for their open source code:
* J. Shi, C. Liu and L. Mackey. Sampling with Mirrored Stein Operators. ICLR, 2022. [[Paper](https://arxiv.org/abs/2106.12506)] | [[Code](https://github.com/thjashin/mirror-stein-samplers)].
* L. Li, Q. Liu, A. Korba, M. Yurochkin and J. Solomon. Sampling with Mollified Interaction Energy Descent. ICLR, 2023. [[Paper](https://arxiv.org/abs/2210.13400)] | [[Code](https://github.com/lingxiaoli94/MIED)].


We did not contribute any of the datasets used in our experiments. Please get in touch if 
there are any conflicts of interest or other issues with hosting these datasets here.
* The HIV dataset is from the [Stanford HIV Drug Resistance Database](http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt).
* The Adult dataset is from the [UCI Machine Learning Repositorry](https://archive.ics.uci.edu/dataset/2/adult). 