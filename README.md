# Partial-Label Learning with a Reject Option

This repository contains the code of the paper

> T. Fuchs, F. Kalinke, K. Böhm (2025): "Partial-Label Learning with a Reject Option", Transactions on Machine Learning Research

This document provides (1) an outline of the repository structure and (2) steps to reproduce the experiments including setting up a virtual environment.

## Repository Structure

* The folder `data` contains all datasets used within our work.
  * The subfolder `realworld-datasets` contains commonly used real-world datasets for partial-label learning, which are initially provided by [Min-Ling Zhang](https://palm.seu.edu.cn/zhangml/Resources.htm).
  * The subfolder `ucipp` contains UCI datasets that have been used in our controlled experiments.
  The files are provided by Luis Paulo on [GitHub](https://github.com/lpfgarcia/ucipp).
* The folder `experiments` contains the data to run all experiments. This directory is initially empty. Run `python script_create_data.py` to populate it.
* The folder `partial_label_learning` contains the code of the experiments.
  * The subfolder `related_work` contains all implementations of related work algorithms and our method.
  * `config.py` contains configurations.
  * `data.py` contains utility methods to generate and load data.
  * `pll_classifier_base.py` is the base class for all our implementations.
  * `result.py` contains utility methods to save the experiments' results.
* The folder `plots` contains all the plots that appear in the paper.
* The folder `reference_models` contains source code for supervised models such as the LeNet architecture.
* The folder `results` contains the results of all experiments. This directory is initially empty. Run `python script_runall.py` to populate it.
* The folder `saved_models` contains saved variational auto-encoders for the MNIST datasets to be used by our nearest-neighbor method.

* Additionally, there are the following files in the root directory:
  * `.gitignore`
  * `LICENSE` describes the repository's licensing.
  * `README.md` is this document.
  * `requirements.txt` is a list of all required `pip` packages for reproducibility.
  * `script_create_data.py` is a Python script to create all experimental data.
  * `script_create_plots.py` is a  Python script to create all plots and tables in the paper.
  * `script_run_all.py` runs all experimental configurations in the `experiments` folder.
  * `script_train_vae.py` trains a variational auto-encoder on the MNIST train datasets.

## Setup

Before running scripts to reproduce the experiments, you need to set up an environment with all the necessary dependencies.
Our code is implemented in Python (version 3.11.5; other versions, including lower ones, might also work).

We used `virtualenv` (version 20.24.3; other versions might also work) to create an environment for our experiments.
First, you need to install the correct Python version yourself.
Next, you install `virtualenv` with

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
python -m pip install virtualenv==20.24.3
```

</td>
<td>

``` powershell
python -m pip install virtualenv==20.24.3
```

</td>
</tr>
</table>

To create a virtual environment for this project, you have to clone this repository first.
Thereafter, change the working directory to this repository's root folder.
Run the following commands to create the virtual environment and install all necessary dependencies:

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

</td>
<td>

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

</td>
</tr>
</table>

## Reproducing the Experiments

Make sure that you created the virtual environment as stated above.
The script `script_create_data.py` creates all experimental settings including the artificial noise.
The script `script_run_all.py` runs all the experiments.
Running all experiments takes two to three days on a system with 48 cores and one `NVIDIA GeForce RTX 3090`.

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
python script_create_data.py
python script_run_all.py
```

</td>
<td>

``` powershell
python script_create_data.py
python script_run_all.py
```

</td>
</tr>
</table>

This creates `.parquet.gz` files in `results` containing the results of all experiments.

## Using the Data

The experiments' results are compressed `.parquet` files.
You can easily read any of them with `pandas`.

``` python
import pandas as pd

results = pd.read_parquet("results/xyz.parquet.gz")
```

## Generating Plots

To obtain plots from the data, use the python script `script_create_plots.py`.
Note that this script requires a working installation of LaTeX on your local system.
Use the following snippets to generate all plots in the paper.
Generating all of them takes about half an hour.

<table>
<tr>
<td> Linux + MacOS (bash-like) </td>
<td> Windows (powershell) </td>
</tr>
<tr>
<td>

``` sh
python script_create_plots.py
```

</td>
<td>

``` powershell
python script_create_plots.py
```

</td>
</tr>
</table>
