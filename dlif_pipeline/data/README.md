# MIL Pipeline - Data Requirements and Preparation

## Required Files

Place the following files in the `/data` directory before running the pipeline:

- `rwd.csv`
- `genomics.csv`
- `digital_pathology.csv`
- `fmrad.csv`
- `split.json`
- `features.json`
- `Mil2/data/data/no_genomics_train.json`
- `Mil2/data/data/no_genomics_test.json`
- `Mil2/data/data/no_genomics_ext_val.json`

## How to Create Parquet and Annotation Files

1. **Split the data**  
   Run:  
   `python I3LUNG_PDSS/dlif_pipeline/data/scripts/utils/split.py`  
   This creates train, test, and uoc splits for each modality and saves them in the `split` folder.

2. **Impute and normalize the splits**  
   Run:  
   `python I3LUNG_PDSS/dlif_pipeline/data/scripts/utils/impute_and_normalize.py`  
   This creates additional files with the `_processed` suffix in the `split` directory.

3. **Create the Parquet files**  
   Run:  
   `python I3LUNG_PDSS/dlif_pipeline/data/scripts/create_parquet.py`  
   This generates the Parquet files in the `/data` folder.

4. **Create the annotation file**  
   Run:  
   `python I3LUNG_PDSS/dlif_pipeline/data/scripts/create_annotations.py`  
   The annotation file will be saved in `/data`.

   - **Note:** The `create_annotations.py` script includes a `use_subfolds` option.  
     To enable intra-center subanalysis, edit the script and set:  
     `ann = create_annotations(use_subfolds=True)`

These steps will produce:
- Two Parquet files (one with `mod2 = pyrad`, one with `mod2 = fmrad`)
- An annotation file with outcomes and subanalysis flags

Ensure all files are present before running downstream MIL experiments.
