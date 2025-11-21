# Data Requirements and Preparation

## Required Files

Place the following files in the `I3LUNG_PDSS/data` directory before running the pipelines:

- `/data/rwd.csv`
- `/data/genomics.csv`
- `/data/digital_pathology.csv`
- `/data/pyradiomics.csv`
- `/data/fmrad.csv`
- `/data/outcomes.csv`
- `/data/features.json`
- `/data/no_genomics_train.json`
- `/data/no_genomics_test.json`
- `/data/no_genomics_ext_val.json`

Each CSV file must include `Subject`, `SET`, and `CENTER` columns, where `SET` contains values: `TRAIN`, `TEST`, or `EXVAL`.

## How to Create Parquet and Annotation Files

Please run all commands from within the  `I3LUNG_PDSS` directory.

1. **Impute and normalize the data**  
   Run:  
   `python dlif_pipeline/preprocessing/utils/impute_and_normalize.py`  
   This creates processed versions of all modality files with the `_processed` suffix in the `/data` directory.

2. **Create the Parquet files**  
   Run:  
   `python dlif_pipeline/preprocessing/create_parquet.py`  
   This generates the Parquet files in the `/data` directory.

3. **Create the annotation file**  
   Run:  
   `python dlif_pipeline/preprocessing/create_annotations.py`  
   The annotation file will be saved in `/data`.

   - **Note:** The `create_annotations.py` script includes a `use_subfolds` option.  
     To enable intra-center subanalysis, edit the script and set:  
     `ann = create_annotations(use_subfolds=True)`

These steps will produce:
- Two Parquet files (one with `mod2 = pyradiomics`, one with `mod2 = fmrad`)
- An annotation file with outcomes and subanalysis flags

Ensure all files are present before running downstream experiments.