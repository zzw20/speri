# SPERI: Semi-Parametric Estimator for General Response Differences under Imbalance

**Zhewei Zhang**

This repository contains the code for implementing the SPERI estimator
and reproducing the simulation studies in the accompanying paper.

SPERI is developed for estimating the general response difference when
comparing two separately conducted studies that differ in information disclosure and in whom they enrolled. The
method uses observed covariates to remove the imbalance between the two
study populations and allows the general response difference to be
evaluated using different summaries of the outcome distribution,
including the mean and quantiles.

## Repository Structure

The simulation code is organized into three folders corresponding to the
simulation settings considered in the paper:

-   `low_dimensional/` contains the simulation studies for the
    low-dimensional setting. Results are evaluated for the mean, median,
    0.1-quantile, and 0.9-quantile.
-   `moderate_dimensional/` contains the simulation studies for the
    moderate-dimensional setting. Results are evaluated for the mean,
    median, 0.1-quantile, and 0.9-quantile.
-   `realistic/` contains the simulation studies designed to mimic the
    setting of the PHAROS and PREDICT-HD data analysis. Results are
    evaluated for the mean, median, 0-25 quantile, and 0-75 quantile.

Within each summary folder, the scripts correspond to the
estimators considered in the paper. Depending on the simulation setting,
these include:

-   `simple`: the simple estimator without adjustment for imbalance
    between the two studies.
-   `bruteforce`: the bruteforce estimator that uses the density ratio alone.
-   `outcome_model`: the outcome-model estimator that uses the outcome model alone.
-   `speri_k`: SPERI with kernel-based estimators.
-   `speri_ds`: SPERI with data splitting estimates both nuisance parameters with deep neural networks.

The `.R` scripts are implemented in R, while the `.py` scripts are
implemented in Python.

The overall directory structure is:

``` text
speri/
├── low_dimensional/
│   ├── mean/
│   ├── median/
│   ├── quantile_010/
│   └── quantile_090/
│
├── moderate_dimensional/
│   ├── mean/
│   ├── median/
│   ├── quantile_010/
│   └── quantile_090/
│
├── realistic/
│   ├── mean/
│   ├── median/
│   ├── quantile_025/
│   └── quantile_075/
│
└── LICENSE
```

## Running the Simulations

Each simulation script can be run independently. First select the
simulation setting and the outcome summary of interest, and then run the
script corresponding to the desired estimator.

For example, the scripts in

``` text
low_dimensional/mean/
```

reproduce the low-dimensional simulation for the mean general response
difference. The folder contains:

``` text
simple.R
bruteforce.R
outcome_model.R
speri_k.R
speri_ds.py
```

These scripts implement the simple, bruteforce, outcome-model,
kernel-based SPERI, and data-splitting SPERI estimators, respectively.

Similarly, `low_dimensional/median/` contains the corresponding
simulations when the median is used as the outcome summary, while
`quantile_010/` and `quantile_090/` contain the simulations for the 0.1
and 0.9 quantiles.

### R Scripts

The R simulation scripts can be run from R. For example,

``` r
source("low_dimensional/mean/speri_k.R")
```

runs the kernel-based SPERI estimator for the mean in the
low-dimensional setting.

### Python Scripts

The SPERI estimator with data splitting is implemented in Python. For
example, the low-dimensional mean simulation can be run from the command
line using

``` bash
python low_dimensional/mean/speri_ds.py
```

The data-splitting implementation uses neural networks to estimate the
nuisance parameters. The neural networks are trained separately within
each data-splitting fold so that nuisance parameter estimation and
evaluation are performed on separate observations.

## Low-Dimensional Simulations

The `low_dimensional/` folder contains simulations for a setting with a
low-dimensional covariate. This setting is used to compare the different
estimators when both kernel-based and machine-learning-based nuisance
parameter estimation are feasible.

Four outcome summaries are considered:

``` text
mean/
median/
quantile_010/
quantile_090/
```

## Moderate-Dimensional Simulations

The `moderate_dimensional/` folder contains simulations for the
moderate-dimensional setting.

The same four outcome summaries are considered:

``` text
mean/
median/
quantile_010/
quantile_090/
```

This setting illustrates the data-splitting implementation of SPERI when
the dimension of the covariates increases and kernel-based nuisance
parameter estimation becomes less reliable.

The folders contain scripts for the simple estimator and the SPERI
estimator with data splitting:

``` text
simple.py
speri_ds.py
```

## Realistic Simulations

The `realistic/` folder contains simulations designed to mimic the
setting of the PHAROS and PREDICT-HD data analysis in the accompanying
paper. Because the PHAROS and PREDICT-HD data cannot be released
publicly, the repository does not include the data or code for the
real-data analysis. Instead, we provide this realistic simulation
setting to illustrate the implementation of the proposed methods in a
setting that resembles the real-data application.

Four outcome summaries are considered:

``` text
mean/
median/
quantile_025/
quantile_075/
```

The corresponding folders contain implementations of the estimators
considered in the realistic simulation study:

``` text
simple.R
bruteforce.R
outcome_model.R
speri_ds.py
```

