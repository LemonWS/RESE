# RESE

Code for **Relation-based Equilibrium State Estimation (RESE)** for multi-system time-series forecasting.

This repository contains the statistical RESE implementation, including pairwise relation estimation, equilibrium-state estimation, magnitude forecasting, reconstruction, and multivariate benchmark evaluation.

## Requirements

Python 3.10+ is recommended.

Install the main dependencies:

```bash
pip install numpy pandas scipy scikit-learn statsmodels
```

## Data Format

Input data should be stored as a CSV file.

Example:

```csv
date,system_1,system_2,system_3
2024-01-01,10.2,15.4,8.7
2024-01-02,10.5,15.1,8.9
2024-01-03,10.8,15.7,9.0
```

The date column is optional. All remaining columns are treated as individual time series.

The current implementation requires:

* at least two numeric series;
* chronologically ordered observations;
* no missing or infinite values.

## Basic Usage

Run the standard RESE forecasting pipeline with:

```bash
python demo_v3.py \
    --data data/example.csv \
    --date-col date \
    --input-length 96 \
    --output-length 24 \
    --representation auto
```

`--input-length` specifies the historical lookback window and `--output-length` specifies the forecasting horizon.

Available relation representations are:

```text
auto
log_ratio
asinh
signed_log
additive
```

With `auto`, RESE uses `log_ratio` for non-negative data and `asinh` when negative values are present.

For example:

```bash
python demo_v3.py \
    --data data/example.csv \
    --date-col date \
    --input-length 96 \
    --output-length 96 \
    --representation asinh
```

## Equilibrium Refinement

Optional robust equilibrium refinement can be enabled using:

```bash
python demo_v3.py \
    --data data/example.csv \
    --date-col date \
    --input-length 96 \
    --output-length 24 \
    --refine-equilibrium \
    --refinement-method huber
```

Available methods are:

```text
huber
cauchy
exponential
```

## Multivariate Benchmark

For standard multivariate forecasting experiments, use the selection-locked benchmark:

```bash
python demo_multivariate_benchmark.py \
    --data data/ETTh1.csv \
    --date-col date \
    --dataset ETTh1 \
    --input-length 96 \
    --pred-len 96 \
    --model-space standardized \
    --representation auto \
    --max-test-windows 0 \
    --test-stride 1
```

Supported predefined dataset names are:

```text
ETTh1
ETTh2
ETTm1
ETTm2
generic
```

For a custom dataset:

```bash
python demo_multivariate_benchmark.py \
    --data data/my_dataset.csv \
    --date-col date \
    --dataset generic \
    --input-length 96 \
    --pred-len 96 \
    --max-test-windows 0
```

Setting

```text
--max-test-windows 0
```

evaluates all available test windows.

Predictions and targets can optionally be saved with:

```bash
--save-npz results/rese_results.npz
```

## Main Files

```text
demo_con.py
    Standard RESE forecasting example.

demo_multivariate_benchmark.py
    Selection multivariate benchmark.

pairwise_relation.py
    Statistical pairwise relation estimation.

relation_matrix.py
    Relation-matrix construction.

equilibrium_solver.py
    Global equilibrium-state estimation.

equilibrium_refinement.py
    Optional robust equilibrium refinement.

predictor.py
    Magnitude forecasting and reconstruction.

evaluation.py
    Evaluation utilities.
```
