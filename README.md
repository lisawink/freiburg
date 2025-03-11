# Urban Morphmetrics & Temperature in Freiburg

Caution: code and repo in progress!

## Contents

Core functions in [src/main.py](src/main.py)

Data acquisition and pre-processing in [notebooks/preprocess_data.ipynb](notebooks/preprocess_data.ipynb)

Morphometric parameter calculation in [notebooks/calc_params.ipynb](notebooks/calc_params.ipynb)

Some initial plots of temperature vs parameters [here](https://lisawink.github.io/freiburg-myst/linregress-2d), their temporal dependence in [diurnal plots](https://lisawink.github.io/freiburg-myst/diurnal-plot) and their [scale dependence here](https://lisawink.github.io/freiburg-myst/scale-plot).

Have a look at all the morphometric pararmeter we are analysing [here](https://lisawink.github.io/freiburg-myst/paper) and their [distributions](https://lisawink.github.io/freiburg-myst/param-distribution) as well as a [map](https://lisawink.github.io/freiburg-myst/visualise-data) of weather stations and building footprints and streets.

## Setup

To create the environment, run:

```bash
conda env create --file environment.yml

conda activate py312_freiburg