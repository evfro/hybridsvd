# Guide to reproduce experiments

If you just want to see all the results, simply open `View_all_results.ipynb` notebook. You can do it directly on Github. All other notebooks (with code) can also be viewed with Github.

## Preparing environment for experimentation
**NOTE:** Only Linux environment is supported. The code has been tested on Ubuntu 14.04.

In order to run the code in this repository you firstly need to install the following software:  
1. Python 2.7 (Python 3 is not supported)  
  1.1 `Pandas`, `Numpy`, `Scipy`, `Matplotlib`, `Jupyter` and `Numba` packages.  
2. [`Suitesparse`](http://faculty.cse.tamu.edu/davis/suitesparse.html) (provides CHOLMOD functionality to compute sparse Cholesky decomposition)
3. [`Scikit-sparse`](https://github.com/scikit-sparse/scikit-sparse) package which conveniently wraps suitesparse (provided in the repository)
4. Special version of [`Polara`](https://github.com/evfro/polara) framework, used to conduct all experiments (provided in the repository)

### Python
The easiest (and recommended) way to get python and all required packages at once is to use the latest [Anaconda distribution](https://www.continuum.io/downloads).
If you use a separate `conda` environment for testing, the following command can be used to ensure that all required dependencies are in place (see [this](http://conda.pydata.org/docs/commands/conda-install.html) for more info):  
```
conda install --file conda_req.txt
```
Alternatively, a new `conda` environment with all required packages can be created by:  
```
conda create -n <your_environment_name> python=2.7 --file conda_req.txt
```
The file `conda_req.txt` can be found in `polara_fixed.zip` archive in this repository.

### Suitesparse
`Suitesparse` can be installed with the following command:

```
sudo apt-get install libsuitesparse-dev
```

### Scikit-sparse
A fixed version of `scikit-sparse` is provided within the repository and can be installed with pip (don't forget to activate your conda environment if you use it):
```
pip install --user scikit-sparse.zip
```

### Polara
This is the most important part as it provides tools to conduct full experiment. `Polara` can also be installed with pip (mind your conda environment):
```
pip install --user --upgrade polara_fixed.zip
```


## Running the code
Use Jupyter Notebooks to play with the code. There are 4 main notebooks, corresponding to 4 experiments: standard and cold-start scenarios for Movielens (ML) and BookCrossing(BX) datasets. The names of the notebooks are self-explaining.

