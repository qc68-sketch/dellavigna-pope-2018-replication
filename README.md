# DellaVigna and Pope (2018): Python Replication

**Qianyu Chen** | MA Economics, Duke University (expected 2027)

This repository replicates Table 3 and Table 5 (Panels A and B) of
DellaVigna and Pope (2018), ["What Motivates Effort? Evidence and Expert
Forecasts"](https://doi.org/10.1093/restud/rdx033), *The Review of Economic
Studies* 85(2): 1029–1069. It also includes an audit of the data cleaning
and an extension that predicts effort at the highest piece rate.

## Main findings

- **Data cleaning.** Rebuilding the sample from the raw data gives 9,863
  observations, two more than the paper's 9,861. Both extra records fall
  within the outage interval excluded by the authors but outside my
  inferred interval.

- **Main results.** Table 3 matches apart from minor rounding differences,
  and most of Table 5 agrees closely with the paper. Differences in MD
  standard errors reflect bootstrap draws and implementation choices.
  The discrete-pay predictions for the low-pay treatment are about 12%
  below the published values; most of this gap comes from differences
  between the model in Appendix A and the released simulation code.

- **Extension.** Estimated on the 0-, 1-, and 4-cent treatments, both cost
  functions predict mean effort in the held-out 10-cent group with errors
  below 1.3%. This supports the robustness of the paper's mean-effort
  predictions to a different treatment split: its interpolation result
  also holds when predicting beyond the estimation range.

## Repository structure

```
code/
  01_clean.py                        data cleaning
  02_main_results_replication.py     Tables 3, 5A and 5B
  03_extension.py                    Table E1
report/
  01_data_cleaning_note.pdf
  02_03_replication_and_extension.pdf
results/
  01_data_cleaning/
  02_main_results_replication/
  03_extension/
```

## Reading guide

### 1. Data cleaning

The [data-cleaning note](report/01_data_cleaning_note.pdf) explains each
cleaning step, compares my sample with the authors' procedure and
documents the remaining differences. The code is
[`01_clean.py`](code/01_clean.py). Exclusion counts, a treatment summary
and the run log are in [`results/01_data_cleaning`](results/01_data_cleaning/).

The main replication and the extension use the authors' released cleaned
data, so that the analysis sample is identical to the paper's.

### 2. Main results

Section 1 of the [replication and extension report](report/02_03_replication_and_extension.pdf)
compares the replicated results with the paper.
[`02_main_results_replication.py`](code/02_main_results_replication.py)
produces the files below, which supply Table 3 and the Replication
columns of Tables 5A and 5B.

| Report table | Contents | Output files |
| --- | --- | --- |
| Table 3 | Mean effort and expert forecasts by treatment | `table3.csv` |
| Table 5A | Benchmark parameters and effort predictions | `table5a_parameters.csv`, `table5a_predictions.csv` |
| Table 5B | Worker preferences and parameters implied by expert forecasts | `table5b_workers.csv`, `table5b_experts.csv` |

The tables, run log and expert-level estimates are in
[`results/02_main_results_replication`](results/02_main_results_replication/).

### 3. Extension

Section 2 of the same report presents the extension.
[`03_extension.py`](code/03_extension.py) estimates the models on the 0-,
1- and 4-cent treatments and predicts effort in the held-out 10-cent
treatment. It produces Table E1. The underlying `table_e1_predictions.csv`
and the run log are in [`results/03_extension`](results/03_extension/).

## Running the code

### Data

The data are not included in this repository. Download the authors'
replication package from the [journal article page](https://doi.org/10.1093/restud/rdx033)
and extract it. The scripts use three files in `CodeForPosting/`:

- `input/MTurkData.dta`: raw data, used by `01_clean.py`
- `dtafiles/MTurkCleanedDataShort.dta`: cleaned worker data, used by
  `02_main_results_replication.py` and `03_extension.py`
- `dtafiles/ExpertForecastCleanWide.dta`: expert forecasts, used by
  `02_main_results_replication.py`

### Setup

The code requires Python 3 with NumPy, pandas and SciPy:

```bash
python -m pip install numpy pandas scipy
```

Set the paths at the top of each script: `DATA_FILE` and `OUT` in
`01_clean.py`, and `DATA_FOLDER` and `OUTPUT_FOLDER` in the other two.

### Run

From the top-level folder:

```bash
python code/01_clean.py
python code/02_main_results_replication.py
python code/03_extension.py
```

Each script writes its outputs and a run log to its output folder.
Scripts 02 and 03 read the authors' cleaned data directly, so they do not
depend on script 01. Bootstrap results use the fixed seed 20261026.
