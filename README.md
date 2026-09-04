# Augmented Vector Space Playground

## NASA Ancillary Software Compliance

**Software Type:** Ancillary
**Classification:** Open Source
**Export Control:** EAR99 — No export restrictions apply
**Safety Critical:** No
**Language:** Haskell (GHC 9.x)
**Dependencies:** `base` only (no third-party packages)
**Platform:** Any platform with GHC; tested via [play.haskell.org](https://play.haskell.org)

## Abstract

This software implements an Augmented Vector Space (AVS) representation for univariate time series data. Rather than modelling temporal dependencies with recurrent architectures (e.g. LSTM, GRU), AVS lifts each time step into a fixed-dimensional Euclidean or cosine vector space by concatenating lag features, finite-difference features, and rolling window statistics. Standard vector-space operations — nearest-neighbour retrieval, ordinary least-squares regression, and pairwise distance matrices — are then applied directly to the augmented vectors. The playground is a self-contained, dependency-free Haskell program intended for algorithm exploration, education, and rapid prototyping of time series analysis pipelines.

## Version History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0.0 | 2026-09-04 | (see Point of Contact) | Initial release: augmentation, kNN, OLS |
| 2.0.0 | 2026-09-04 | (see Point of Contact) | Added cosine metric, multi-signal corpus, distance matrix, regression report |

## 1. Background and Motivation

Time series forecasting and similarity search are conventionally addressed with sequence models that maintain hidden state across time steps. These approaches incur substantial complexity in training, hyperparameter selection, and inference. The augmented vector space approach, as discussed in the AI Stack Exchange question "Time Series: LSTM or Augmented Vector Space?" (https://ai.stackexchange.com/questions/5943), proposes that for many practical signals the temporal structure can be captured entirely by three classes of feature. First, lag features: a sliding window of the p most recent observations, forming the primary coordinate axes of the vector space. Second, finite differences: first and second discrete derivatives, encoding local velocity and acceleration. Third, rolling statistics: windowed mean and standard deviation, encoding local amplitude and variability. Once lifted into this space, each time step is a standard vector and the full repertoire of linear algebra and metric geometry applies without recurrence.

## 2. Software Description

### 2.1 Architecture

The program is structured as a single Haskell module (Main) with the following logical layers. Signal generators produce raw sample sequences. Feature extractors consume those sequences under the control of an AugConfig record holding the lag window p, derivative order d, and rolling window w. The extractors produce a list of AugPoint values, each carrying a time index, a feature vector, and a source signal name. Those points are then passed to three analysis components: k-nearest-neighbour search under either L2 or cosine distance, ordinary least-squares regression via coordinate descent, and pairwise distance matrix construction.

### 2.2 Key Data Types

| Type | Description |
|------|-------------|
| `AugConfig` | Augmentation hyperparameters: lag window p, derivative order d, rolling window w |
| `AugPoint` | A single augmented observation: time index, feature vector, source signal name |
| `Metric` | Sum type selecting L2 (Euclidean) or Cosine distance |
| `Vec` | `[Double]` — a dense feature vector |
| `Sample` | `Double` — a single raw time series observation |

### 2.3 Algorithms

#### Augmentation

For a time series x_0, x_1, ..., x_n and config (p, d, w), the augmented vector at time t is the concatenation of p lag values x_t through x_{t-p+1}, d finite differences starting with the first difference of x at t, and the rolling mean and standard deviation of the w-sample window ending at t. Total vector dimension is p + d + 2.

#### Distance Metrics

L2 (Euclidean) distance is the square root of the sum of squared element-wise differences. It is sensitive to amplitude and is appropriate when magnitude carries information. Cosine distance is one minus the normalised dot product of the two vectors. It is invariant to vector scale and is appropriate for shape similarity regardless of amplitude. The implementation guards against zero-norm vectors with a small epsilon floor on the denominator.

#### k-Nearest Neighbours

Brute-force search with complexity O(n times d) over the augmented corpus. Returns the k closest points by the chosen metric, excluding the query point itself.

#### Ordinary Least Squares

Fits a linear predictor y-hat = beta dot v(t) to predict x_{t+1} from v(t). The normal equations are solved iteratively via Gauss-Seidel coordinate descent with 200 iterations as the default. No matrix inversion and no external linear algebra library are required.

## 3. Inputs and Outputs

### 3.1 Inputs

All inputs are specified by editing constants in main.

| Parameter | Variable | Default | Description |
|-----------|----------|---------|-------------|
| Lag window | `lagWindow` | 3 | Number of past observations per vector |
| Derivative order | `derivOrder` | 1 | 0 = none, 1 = first difference, 2 = first and second difference |
| Rolling window | `rollingWin` | 4 | Window size for mean and standard deviation features |
| Signal | `xs` | `sineWave 60` | Input time series |
| kNN query time | `queryT` | 10 | Time index used as kNN query point |
| OLS iterations | `iters` | 200 | Coordinate descent iterations |

Built-in signal generators:

| Function | Description |
|----------|-------------|
| `sineWave n` | Pure sine, amplitude 3, period 20 |
| `stepWave n` | Piecewise constant: 1 then -1 then 2 |
| `noisyWave n` | Sine plus deterministic double-sine interference |

### 3.2 Outputs

All output is written to stdout as formatted plain text. The program produces four sections. The signal norms section shows the L2 norm of each augmented vector as an ASCII bar chart for all three built-in signals side by side. The kNN results section shows the top five neighbours of the query point under both L2 and cosine metrics, with distances and source signal labels. The OLS regression section shows fitted coefficients, RMSE, maximum absolute error, and a ten-row actual versus predicted table. The distance matrices section shows pairwise L2 and cosine distances among six selected query points at t = 5, 10, 15, 20, 25, and 30.

## 4. Usage

### 4.1 Online (no local installation required)

Open https://play.haskell.org, delete the default template, paste the full contents of Main.hs, and press Run or Ctrl+Enter.

### 4.2 Local (GHC required)

Compile with the command `ghc -O Main.hs -o avs` and run with `./avs`. GHC 9.4 or later is recommended. No cabal or stack project file is required as the program uses only base.

### 4.3 Suggested Experiments

Changing lagWindow to 6 increases vector dimensionality and makes the periodic block structure of the distance matrix more pronounced. Swapping sineWave for stepWave in the regression report increases OLS error because a step signal is not well modelled by linear lag features alone. Changing queryT in reportKNN explores different neighbourhoods of the signal manifold. Replacing L2 with Cosine throughout compares shape-based retrieval with amplitude-based retrieval.

## 5. Limitations

Brute-force kNN has complexity O(n squared) for a full distance matrix and is suitable for exploratory use on signals up to approximately ten thousand samples. For production-scale search, index structures such as k-d trees, ball trees, or HNSW would be required. Coordinate descent OLS converges for well-conditioned problems; ill-conditioned feature matrices with highly correlated lags may require more iterations or regularisation. Signal length defaults to 60 samples; longer signals are supported by changing the integer argument to any generator function. The augmentation scheme is defined for scalar time series only; multivariate extension would require concatenating lag vectors across channels. The program produces no output files; all results are printed to stdout.

## 6. Point of Contact

| Field | Value |
|-------|-------|
| Author | — |
| Organisation | — |
| Email | — |
| Distribution | Unlimited (EAR99) |

This README was prepared in accordance with NASA NPR 2210.1C requirements for ancillary software released under an open-source licence. The software has not been subjected to NASA IV&V and is not intended for flight or safety-critical use.

## 7. Licence

This software is released into the public domain under the Unlicense (https://unlicense.org). No warranty is expressed or implied.