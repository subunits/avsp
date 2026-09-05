# Augmented Vector Space Ground Tool

## NASA Ancillary Software Compliance

**Software Type:** Ancillary
**Classification:** Open Source
**Export Control:** EAR99 — No export restrictions apply
**Safety Critical:** No
**Language:** Haskell (GHC 9.x)
**Dependencies:** `base` only (no third-party packages)
**Platform:** Any platform with GHC; tested via [play.haskell.org](https://play.haskell.org)

## Abstract

This software implements an Augmented Vector Space (AVS) representation for univariate time series data. Rather than modelling temporal dependencies with recurrent architectures such as LSTM or GRU, AVS lifts each time step into a fixed-dimensional Euclidean or cosine vector space by concatenating lag features, finite-difference features, and rolling window statistics. Standard vector-space operations — nearest-neighbour retrieval, ordinary least-squares regression, pairwise distance matrices, Shannon entropy analysis, and anomaly detection — are then applied directly to the augmented vectors. The tool is a self-contained, dependency-free Haskell program intended for ground-based signal analysis, algorithm exploration, and rapid prototyping of time series pipelines against telemetry data.

## Version History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0.0 | 2026-09-04 | (see Point of Contact) | Initial release: augmentation, kNN, OLS |
| 2.0.0 | 2026-09-04 | (see Point of Contact) | Added cosine metric, multi-signal corpus, distance matrix, regression report |
| 3.0.0 | 2026-09-05 | (see Point of Contact) | Added CSV ingestion, anomaly detection, CSV export, command-line interface |
| 3.1.0 | 2026-09-05 | (see Point of Contact) | Replaced flat step wave with six high-entropy signal generators; added Shannon entropy report |

## 1. Background and Motivation

Time series forecasting and similarity search are conventionally addressed with sequence models that maintain hidden state across time steps. These approaches incur substantial complexity in training, hyperparameter selection, and inference. The augmented vector space approach proposes that for many practical signals the temporal structure can be captured entirely by three classes of feature. First, lag features: a sliding window of the p most recent observations, forming the primary coordinate axes of the vector space. Second, finite differences: first and second discrete derivatives, encoding local velocity and acceleration. Third, rolling statistics: windowed mean and standard deviation, encoding local amplitude and variability. Once lifted into this space, each time step is a standard vector and the full repertoire of linear algebra and metric geometry applies without recurrence.

Version 3.1.0 adds a suite of high-entropy signal generators — including a logistic map chaotic attractor and a Lorenz system — to ensure that no signal region is dynamically flat, making the tool more representative of real telemetry behaviour.

## 2. Software Description

### 2.1 Architecture

The program is structured as a single Haskell module (Main) with the following logical layers. Signal generators or CSV ingestion produce raw sample sequences. Feature extractors consume those sequences under the control of an AugConfig record holding the lag window p, derivative order d, and rolling window w. The extractors produce a list of AugPoint values, each carrying a time index, a feature vector, and a source signal name. Those points are then passed to six analysis components: k-nearest-neighbour search under either L2 or cosine distance, ordinary least-squares regression via coordinate descent, pairwise distance matrix construction, Shannon entropy per augmented dimension, anomaly detection by mean kNN distance thresholding, and CSV export of all results.

### 2.2 Key Data Types

| Type | Description |
|------|-------------|
| `AugConfig` | Augmentation hyperparameters: lag window p, derivative order d, rolling window w |
| `AugPoint` | A single augmented observation: time index, feature vector, source signal name |
| `Metric` | Sum type selecting L2 (Euclidean) or Cosine distance |
| `Vec` | `[Double]` — a dense feature vector |
| `Sample` | `Double` — a single raw time series observation |

### 2.3 Signal Generators

All generators are deterministic and produce no external I/O. All noise is derived from a linear congruential generator (LCG) seeded at construction time, making every run reproducible without a random seed parameter.

| Function | Type | Description |
|----------|------|-------------|
| `sineWave n` | Periodic | Pure sine, amplitude 3, period 20 |
| `logisticMap n` | Chaotic | Logistic map at r=3.9; deterministic chaos, dense attractor |
| `lorenzWave n` | Chaotic | Euler-integrated Lorenz x-component; two-lobe butterfly structure |
| `sawtoothWave n` | Quasi-periodic | Linear ramp with harmonic interference; continuous non-flat structure |
| `stepWave n` | Piecewise + noise | Step function with Gaussian noise; no flat regions |
| `randomWalk n` | Stochastic | Cumulative LCG noise; non-stationary |

The logistic map and Lorenz generators are particularly useful for testing kNN and anomaly detection because their attractor geometry produces a non-trivial distribution of pairwise distances, unlike periodic or piecewise-constant signals.

### 2.4 Algorithms

#### Augmentation

For a time series x_0, x_1, ..., x_n and config (p, d, w), the augmented vector at time t is the concatenation of p lag values, d finite differences, and the rolling mean and standard deviation of the w-sample window ending at t. Total vector dimension is p + d + 2.

#### Distance Metrics

L2 (Euclidean) distance is the square root of the sum of squared element-wise differences. It is sensitive to amplitude and appropriate when magnitude carries information. Cosine distance is one minus the normalised dot product of the two vectors. It is invariant to vector scale and appropriate for shape similarity regardless of amplitude. The implementation guards against zero-norm vectors with an epsilon floor on the denominator.

#### k-Nearest Neighbours

Brute-force search with complexity O(n times d) over the augmented corpus. Returns the k closest points by the chosen metric, excluding the query point itself.

#### Ordinary Least Squares

Fits a linear predictor y-hat = beta dot v(t) to predict x_{t+1} from v(t). The normal equations are solved iteratively via Gauss-Seidel coordinate descent with 200 iterations as the default. No matrix inversion and no external linear algebra library are required.

#### Shannon Entropy

Each augmented dimension is independently binned into sqrt(n) equal-width bins and the Shannon entropy H = -sum(p_i log2 p_i) is computed in bits. This quantifies how much information each feature dimension carries and guides selection of the lag window and derivative order. Chaotic signals such as the logistic map produce near-maximum entropy; periodic signals produce low entropy concentrated at the attractor.

#### Anomaly Detection

Each augmented point is scored by the mean L2 distance to its k nearest neighbours. A point is flagged as anomalous if its score exceeds a configurable multiple (default 2.0) of the global mean score across all points. This is a parameter-free local outlier approach that requires no training data beyond the input signal itself.

## 3. Inputs and Outputs

### 3.1 Inputs

The tool accepts an optional CSV file path as a command-line argument. If no argument is supplied it falls back to the logistic map. The CSV file must have one sample per row. The last column of each row is used as the signal value. A single-line header row is detected automatically and skipped. Lines beginning with # are treated as comments and ignored.

All augmentation parameters are specified by editing constants in the source:

| Parameter | Variable | Default | Description |
|-----------|----------|---------|-------------|
| Lag window | `lagWindow` | 3 | Number of past observations per vector |
| Derivative order | `derivOrder` | 1 | 0 = none, 1 = first difference, 2 = first and second difference |
| Rolling window | `rollingWin` | 4 | Window size for mean and standard deviation features |
| Anomaly threshold | `anomalyThreshold` | 2.0 | Multiple of global mean kNN distance above which a point is flagged |
| OLS iterations | `iters` | 200 | Coordinate descent iterations |

### 3.2 Outputs

All analysis is printed to stdout as formatted plain text. Three CSV files are written to the working directory:

| File | Contents |
|------|----------|
| `knn_results.csv` | Time index, signal name, and L2 distance for the 5 nearest neighbours of t=10 |
| `anomalies.csv` | Time index, signal name, and mean kNN distance for all flagged anomalous points |
| `dist_matrix_l2.csv` | Full pairwise L2 distance matrix across all augmented points |

## 4. Usage

### 4.1 Online (no local installation required)

Open https://play.haskell.org, delete the default template, paste the full contents of Main.hs, and press Run or Ctrl+Enter. CSV ingestion and file output are not available in the online playground; the tool will use the logistic map and skip file writes.

### 4.2 Local (GHC required)

```bash
ghc -O Main.hs -o avs
./avs
./avs my_telemetry.csv
```

GHC 9.4 or later is recommended. No cabal or stack project file is required as the program uses only base.

### 4.3 CSV Format

```
# Optional comments
time,value
0,1.23
1,2.34
```

Or single-column with no header:

```
1.23
2.34
3.45
```

### 4.4 Suggested Experiments

Running with the logistic map (default) and then the Lorenz wave illustrates how attractor geometry determines the block structure of the distance matrix. Increasing lagWindow to 6 on the Lorenz signal reveals the two-lobe structure more clearly in the lag scatter. Lowering anomalyThreshold to 1.5 increases sensitivity on the step+noise signal. Supplying a real telemetry CSV replaces the synthetic signal with mission data throughout all reports.

## 5. Limitations

Brute-force kNN has complexity O(n squared) for a full distance matrix and is suitable for exploratory use on signals up to approximately ten thousand samples. For production-scale search, index structures such as k-d trees, ball trees, or HNSW would be required. Coordinate descent OLS converges for well-conditioned problems; ill-conditioned feature matrices with highly correlated lags may require more iterations or regularisation. The augmentation scheme is defined for scalar time series only; multivariate extension would require concatenating lag vectors across channels. The anomaly detector has no training phase and no notion of anomaly type; it flags statistical outliers in the vector space and requires human analyst review to determine operational significance. Shannon entropy is estimated by binning and is sensitive to bin count for small samples; the sqrt(n) heuristic is appropriate for n >= 30.

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
