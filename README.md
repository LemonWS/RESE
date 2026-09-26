# RESE

**Relation-based Equilibrium State Estimation for Multi-System Time Series Forecasting**

RESE is a forecasting framework for **multiple related time series**. Instead of forecasting each system independently, RESE models the relationships among systems and recovers a globally coherent equilibrium state from these pairwise relationships.

The framework separates forecasting into two components:

1. **relative system structure**, estimated from pairwise relationships; and
2. **overall system magnitude**, forecast from the aggregate history.

These components are combined to reconstruct forecasts for all systems.

---

## Overview

Suppose a dataset contains \(N\) related systems observed over time,

$$
\mathbf{y}_t =
[y_{1,t},y_{2,t},\ldots,y_{N,t}]^\top.
$$

RESE first represents the relationship between every pair of systems as

$$
r_{ij,t}
=
T(y_{i,t})-T(y_{j,t}),
$$

where \(T(\cdot)\) is a representation-dependent transformation.

Each pairwise relationship is forecast independently using a set of statistical relation models. The resulting relation forecasts and their validation-based reliability weights form a weighted relation matrix.

RESE then estimates a single globally coherent equilibrium state by solving

$$
\hat{\mathbf u}
=
\arg\min_{\mathbf u}
\sum_{i<j}
w_{ij}
\left(
u_i-u_j-\hat r_{ij}
\right)^2,
$$

subject to the identification constraint

$$
\sum_i u_i=0.
$$

The aggregate magnitude

$$
M_t=\sum_{i=1}^{N} y_{i,t}
$$

is forecast separately. The estimated equilibrium state and magnitude forecast are finally combined to reconstruct system-level forecasts.

---

## RESE Pipeline

```text
Multivariate time-series data
          |
          v
  Relation representation
          |
          v
  Construct system pairs
          |
          v
Pairwise relation estimation
          |
          |  relation forecasts r_ij
          |  reliability weights w_ij
          v
 Weighted relation matrix
          |
          v
 Global equilibrium solver
          |
          v
 Optional robust refinement
          |
          v
 Estimated equilibrium state
          |
          +--------------------+
          |                    |
          |             Aggregate history
          |                    |
          |                    v
          |            Magnitude forecast
          |                    |
          +----------+---------+
                     |
                     v
             Reconstruction
                     |
                     v
          Multi-system forecasts
```

---

## Pairwise Relation Estimation

For every system pair \((i,j)\), RESE constructs a relation series

$$
r_{ij,t}=T(y_{i,t})-T(y_{j,t}).
$$

A collection of statistical models is evaluated using temporal validation.

The current implementation includes:

| Family           | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| Stable           | Robust constant relation estimated from historical observations |
| Trend            | Local linear trend model for smoothly evolving relations        |
| Dynamic          | Low-order ARIMA model for relation dynamics                     |
| Equilibrium      | VECM when cointegration is detected, otherwise VAR              |
| Smooth nonlinear | Cubic spline with ridge regularisation                          |
| Regime           | Two-regime SETAR model                                          |

Candidate models are evaluated using **horizon-matched rolling-origin validation**.

For forecast horizon \(h\),

$$
L_{ij,h}^{(k)}
=
\frac{1}{|\mathcal O|}
\sum_{o\in\mathcal O}
\left(
\hat r_{ij,o+h|o}^{(k)}
-
r_{ij,o+h}
\right)^2,
$$

where \(\mathcal O\) is the set of validation origins.

The selected model is then refitted using the available historical window and produces a complete forecast path

$$
\hat r_{ij,t+1|t},
\ldots,
\hat r_{ij,t+h|t}.
$$

### Relation screening

A persistence relation is used as a null reference. A pair can be rejected when none of the candidate models provides sufficient improvement over this reference.

Rejected or unavailable relations receive zero reliability weight and do not contribute to equilibrium estimation.

---

## Relation Representations

RESE supports several geometries through the common relation form

$$
r_{ij}=T(y_i)-T(y_j).
$$

Available representations are:

| Representation | Intended data                                                      |
| -------------- | ------------------------------------------------------------------ |
| `log_ratio`    | Positive or non-negative systems with multiplicative relationships |
| `asinh`        | Signed data; smooth around zero and compressed for large values    |
| `signed_log`   | Signed data with logarithmic compression                           |
| `additive`     | Direct absolute differences                                        |
| `auto`         | Automatically selects a suitable representation                    |

With

```text
--representation auto
```

the current implementation uses:

```text
all observations >= 0  -> log_ratio
any observation < 0    -> asinh
```

For formal experiments, explicitly specifying the representation is recommended so the experimental condition remains fixed.

---

## Global Equilibrium Estimation

Pairwise forecasts are collected into a relation graph.

Each valid edge contains

$$
(\hat r_{ij},w_{ij}),
$$

where \(\hat r_{ij}\) is the predicted relationship and \(w_{ij}\) is its reliability.

Let \(B\) denote the oriented edge-node incidence matrix, \(W\) the diagonal reliability matrix, and \(\mathbf r\) the vector of pairwise relation forecasts.

The equilibrium problem becomes

$$
\min_{\mathbf u}
\left\|
W^{1/2}(B\mathbf u-\mathbf r)
\right\|_2^2.
$$

Its normal equations involve the weighted graph Laplacian

$$
L=B^\top W B.
$$

The solution converts many local pairwise forecasts into one globally coherent latent equilibrium state.

For the `log_ratio` representation, equilibrium proportions are obtained as

$$
\gamma_i^*
=
\frac{\exp(u_i)}
{\sum_j\exp(u_j)},
$$

so that

$$
\gamma_i^*>0,
\qquad
\sum_i\gamma_i^*=1.
$$

Other representations use their corresponding reconstruction geometry.

---

## Robust Equilibrium Refinement

RESE optionally provides iterative robust refinement.

The pairwise relation forecasts themselves remain fixed. Instead, relation weights are updated according to disagreement between each local relation and the current globally coherent equilibrium.

At iteration \(k\),

$$
\mathbf u^{(k)}
=
\arg\min_{\mathbf u}
\sum_e
w_e^{(k)}
(B_e\mathbf u-r_e)^2,
$$

followed by residual calculation

$$
e_e^{(k)}
=
r_e-B_e\mathbf u^{(k)}.
$$

The reliability weights are then adjusted using a robust consistency function.

Available refinement rules include:

```text
huber
cauchy
exponential
```

This mechanism reduces the influence of pairwise relations that are strongly inconsistent with the global relation structure while preserving their original forecast values.

---

## Magnitude Forecasting

RESE separates the relative equilibrium structure from the overall scale.

The aggregate magnitude is defined as

$$
M_t=\sum_i y_{i,t}.
$$

The current magnitude predictor library includes:

| Predictor  | Description                                          |
| ---------- | ---------------------------------------------------- |
| Naive      | Last-value persistence                               |
| Drift      | Linear extrapolation using average historical change |
| ETS        | Holt additive damped trend                           |
| Auto-ARIMA | Low-order ARIMA selected using AIC                   |

Magnitude models are evaluated using the same temporal forecasting principle as the pairwise relation models.

This produces

$$
\hat M_{t+1|t},\ldots,\hat M_{t+h|t}.
$$

---

## Forecast Reconstruction

The final forecast combines:

* the global equilibrium state; and
* the aggregate magnitude forecast.

For the multiplicative `log_ratio` representation,

$$
\hat y_{i,t+h|t}
=
\hat M_{t+h|t}
\hat\gamma_{i,t+h|t}^{*}.
$$

For signed representations, reconstruction is performed using the corresponding inverse relation geometry implemented by the equilibrium solver.

---

## Installation

Python 3.10+ is recommended.

Clone the repository:

```bash
git clone <repository-url>
cd RESE
```

Install the main dependencies:

```bash
pip install numpy pandas scipy scikit-learn statsmodels
```

A virtual environment is recommended:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Then install the dependencies.

---

## Data Format

The standard input is a CSV file containing one time column and two or more numeric system columns.

Example:

```csv
date,system_1,system_2,system_3,system_4
2024-01-01,10.2,15.4,8.7,20.1
2024-01-02,10.5,15.1,8.9,20.7
2024-01-03,10.8,15.7,9.0,21.0
2024-01-04,11.0,16.0,9.2,21.3
```

All non-date columns are interpreted as systems.

Input data should:

* contain at least two systems;
* contain numeric system values;
* be ordered chronologically;
* contain no `NaN` or infinite values in benchmark experiments.

Missing values should therefore be handled before running RESE.

---

## Quick Start

A basic RESE run can be performed with:

```bash
python demo_con.py \
    --data data/example.csv \
    --date-col date \
    --input-length 96 \
    --output-length 24 \
    --representation auto
```

For positive-valued data, the representation can be fixed explicitly:

```bash
python demo_con.py \
    --data data/example.csv \
    --date-col date \
    --input-length 96 \
    --output-length 24 \
    --representation log_ratio
```

For signed data:

```bash
python demo_con.py \
    --data data/example.csv \
    --date-col date \
    --input-length 96 \
    --output-length 24 \
    --representation asinh
```

---

## Equilibrium Refinement

Robust equilibrium refinement can be enabled with:

```bash
python demo_con.py \
    --data data/example.csv \
    --date-col date \
    --input-length 96 \
    --output-length 24 \
    --representation auto \
    --refine-equilibrium \
    --refinement-method huber \
    --refinement-max-iterations 20
```

Supported methods are:

```text
huber
cauchy
exponential
```

---

## Multivariate Benchmark

The repository provides a selection-locked rolling benchmark for controlled comparison on multivariate forecasting datasets.

Example:

```bash
python demo_multivariate_benchmark_fixed.py \
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

Supported built-in dataset identifiers include:

```text
ETTh1
ETTh2
ETTm1
ETTm2
generic
```

For a custom dataset:

```bash
python demo_multivariate_benchmark_fixed.py \
    --data data/my_dataset.csv \
    --date-col date \
    --dataset generic \
    --input-length 96 \
    --pred-len 96 \
    --max-test-windows 0
```

### Selection-locked protocol

The benchmark is designed so that model-selection decisions are made before test evaluation.

The procedure is:

```text
Training data
      |
      v
Training-only scaling
      |
      v
Training + validation history
      |
      +--> relation-family selection
      +--> relation screening
      +--> reliability estimation
      +--> magnitude-model selection
      |
      v
Freeze model specifications
      |
      v
Test rolling origins
      |
      +--> refit selected specification using past observations
      +--> forecast pairwise relations
      +--> solve global equilibrium
      +--> forecast magnitude
      +--> reconstruct all channels
      |
      v
Evaluation
```

During the test period, candidate-family selection is not rerun.

For complete test evaluation use:

```text
--max-test-windows 0 --test-stride 1
```

---

## Main Command-Line Options

### General RESE demo

```text
--data PATH
--date-col COLUMN
--input-length L
--output-length H
--history-mode {auto,sliding,expanding}
--representation {auto,log_ratio,asinh,signed_log,additive}
--disable-null
--weight-threshold VALUE
--refine-equilibrium
--refinement-method {huber,cauchy,exponential}
--refinement-max-iterations N
```

### Selection-locked benchmark

```text
--data PATH
--date-col COLUMN
--dataset DATASET
--input-length L
--pred-len H
--model-space {standardized,raw}
--representation {auto,log_ratio,asinh,signed_log,additive}
--generic-train-ratio VALUE
--generic-val-ratio VALUE
--test-stride N
--max-test-windows N
--validation-stride N
--max-validation-origins N
--disable-null
--relation-weight-eta VALUE
--weight-threshold VALUE
--refine-equilibrium
--refinement-method {huber,cauchy,exponential}
--refinement-max-iterations N
--save-npz PATH
```

---


## Evaluation

The benchmark evaluates forecasts over

$$
\text{windows}
\times
\text{forecast horizons}
\times
\text{systems}.
$$

For standardized benchmark experiments, the primary MSE is

$$
\mathrm{MSE}
=
\frac{1}{W H N}
\sum_{w=1}^{W}
\sum_{h=1}^{H}
\sum_{i=1}^{N}
\left(
\hat y_{w,h,i}-y_{w,h,i}
\right)^2.
$$

The implementation also records diagnostic information including:

* pairwise relation validation performance;
* selected relation families;
* relation reliability;
* equilibrium consistency;
* magnitude-model selection;
* per-window forecasts;
* runtime information;
* raw-space and standardized-space predictions.

Benchmark predictions and targets can be saved using:

```bash
--save-npz results/rese_result.npz
```

---

## Design Principles

RESE follows several implementation principles.

**Temporal separation.**
Model fitting and validation use historical observations only.

**Relation-first modelling.**
Interactions among systems are explicitly estimated before constructing the global forecast.

**Global consistency.**
Local pairwise forecasts are reconciled through one weighted equilibrium problem.

**Reliability-aware estimation.**
Pairwise relationships contribute according to their validation-derived reliability.

**Geometry-aware forecasting.**
Different data domains can use different relation representations without changing the global solver structure.

**Scale-state separation.**
Overall magnitude and relative system structure are forecast separately before reconstruction.

**Modularity.**
Relation estimation, equilibrium recovery, magnitude forecasting, reconstruction and evaluation are implemented as separate components.

---

## Computational Considerations

A complete relation graph for \(N\) systems contains

$$
K=\frac{N(N-1)}{2}
$$

unique pairwise relations.

The current full-relation implementation therefore has quadratic pair construction with respect to the number of systems.

For example:

```text
N = 10   ->       45 pairs
N = 50   ->    1,225 pairs
N = 100  ->    4,950 pairs
N = 500  ->  124,750 pairs
```

Consequently, full pairwise estimation can become computationally expensive for datasets containing hundreds of channels.

This should be considered when selecting datasets, input lengths, validation origins and forecast horizons.

---

## Reproducibility Notes

For benchmark experiments:

1. keep the temporal train/validation/test split fixed;
2. fit preprocessing statistics using the training period only;
3. use the same input and prediction lengths across compared methods;
4. keep relation representation fixed for formal comparisons;
5. use all eligible test windows for final reported results;
6. avoid using observations after the current forecast origin during fitting or model selection.

The selection-locked benchmark is recommended when the goal is direct comparison under a fixed pre-test model-selection protocol.

---

## Citation

Citation information will be added with the corresponding RESE publication.

If you use this repository in academic work, please cite the associated paper once the bibliographic information becomes available.

---

## License

Please refer to the repository `LICENSE` file for usage and redistribution terms.
