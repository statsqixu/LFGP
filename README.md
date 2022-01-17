# Python Package for Latent Factor Modeling with Grouping Penalty (LFGP)

### LFGP can be applied to crowdsourcing, single-cell rna sequencing analysis, etc.

The package includes three main modules:

- `baseLF`: basic latent factor modeling with $L_2$ penalty
- `LF_MCGP`: latent factor modeling with multi-centroid grouping penalty<sup>1</sup>
- `LF_fuse`: latent factor modeling with fusion penalty<sup>2</sup>

The first module `baseLF` does not provide grouping features for latent factors. The last two modules `LF_MCGP` and `LF_fuse` provide grouping features for latent factors, and `LF_MCGP` has lower computation complexity than `LF_fuse`, but `LF_fuse` does not require specification of group numbers.

> ## `baseLF`:
> `mode`: specify data type, `binary` or `continuous`
> `fit`: fit the data with basic latent factor modeling
> > `data`: $n\times 3$ matrix, the first column includes indices for rows of original matrix, the second column includes indices for columns of original matrix, and the third column includes entries for the corresponding row & column in the original matrix.
> > `n_factors`: dimension/rank of latent factors, default=3
> > `epochs`: training epochs, default=100
> > `learning_rate`: learning rate for gradient descent type of algorithm, default=1e-4
> > `verbose`: status for display, 0: no output; 1: output status for each epoch
> > `opt_func`: optimization algorithm, default=`Adam`
> > `weight_decay`: coefficients for $L_2$ penalty, default=1e-2
> > `batch_size`: batch size for training, default=128
> > `device`: cpu or gpu to be used, if gpu is detected, it will be activated automatically

> ## `LF_MCGP`:
> `mode`: specify data type, `binary` or `continuous`
> `fit`: fit the data with basic latent factor modeling
> > `data`: $n\times 3$ matrix, the first column includes indices for rows of original matrix, the second column includes indices for columns of original matrix, and the third column includes entries for the corresponding row & column in the original matrix.
> > `n_factors`: dimension/rank of latent factors, default=3
> > `n_task_group`: number of groups for rows, default=2
> > `n_worker_group`: number of groups for columns, default=2
> > `lambda1`: coefficients for row-wise MCGP
> > `lambda2`: coefficients for column-wise MCGP
> > `epochs`: training epochs, default=100
> > `learning_rate`: learning rate for gradient descent type of algorithm, default=1e-4
> > `verbose`: status for display, 0: no output; 1: output status for each epoch
> > `opt_func`: optimization algorithm, default=`Adam`
> > `weight_decay`: coefficients for $L_2$ penalty, default=1e-2
> > `batch_size`: batch size for training, default=128
> > `device`: cpu or gpu to be used, if gpu is detected, it will be activated automatically

> ## `LF_fuse`:
> `mode`: specify data type, `binary` or `continuous`
> `fit`: fit the data with basic latent factor modeling
> > `data`: $n\times 3$ matrix, the first column includes indices for rows of original matrix, the second column includes indices for columns of original matrix, and the third column includes entries for the corresponding row & column in the original matrix.
> > `n_factors`: dimension/rank of latent factors, default=3
> > `lambda11`: coefficient of row-wise latent factor to centroid penalty
> > `lambda12`: coefficient of row-wise centroid fusion penalty
> > `lambda21`: coefficient of column-wise latent factor to centroid penalty
> > `lambda22`: coefficient of column-wise centroid fusion penalty 
> > `epochs`: training epochs, default=100
> > `learning_rate`: learning rate for gradient descent type of algorithm, default=1e-4
> > `verbose`: status for display, 0: no output; 1: output status for each epoch
> > `opt_func`: optimization algorithm, default=`Adam`
> > `weight_decay`: coefficients for $L_2$ penalty, default=1e-2
> > `batch_size`: batch size for training, default=128
> > `device`: cpu or gpu to be used, if gpu is detected, it will be activated automatically

## References:

1. Q.Xu, Y.Yuan, J.Wang, A.Qu. Crowdsourcing Utilizing Subgroup Structure of Latent Factor Modeling.
2. W.Pan, X.Shen, B.Liu. Cluster Analysis: Unsupervised Learning via Supervised Learning with a Non-convex Penalty. JMLR