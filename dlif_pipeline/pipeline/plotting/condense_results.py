import pandas as pd
import os
import glob
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / "dlif_pipeline/results"

# Create output directories for consolidated files
CONSOLIDATED_DIR = BASE_DIR / "dlif_pipeline/results"
LINE_DATA_DIR = CONSOLIDATED_DIR / "data/line"
METRICS_DIR = CONSOLIDATED_DIR / "data/metrics"

for dir_path in [CONSOLIDATED_DIR, LINE_DATA_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("="*80)
print("STEP 1: CONSOLIDATING CSV FILES FROM NESTED STRUCTURE")
print("="*80)

def find_csv_files_in_structure(results_dir):
    """
    Traverse the nested directory structure and find all eval_auc_ci.csv and 
    eval_classification_metrics.csv files along with their metadata.
    
    Structure: Cohort/[subanalysis/]outcome/classification_or_survival/method/hypothesis_driven/pyrad_nomip/modality/seed_0/
    """
    csv_data = []
    
    # Walk through all directories
    for root, dirs, files in os.walk(results_dir):
        root_path = Path(root)
        
        # Check if we're in a seed_0 directory with the target files
        if root_path.name == "seed_0" and ("eval_auc_ci.csv" in files or "eval_cindex_ci.csv" in files or "eval_classification_metrics.csv" in files):           # Parse the path to extract metadata
            parts = root_path.parts
            results_idx = parts.index("results")
            print(f"Found: {root_path}, files: {files}")  # debug

            
            # Extract components from path
            path_after_results = parts[results_idx + 1:]
            
            if len(path_after_results) < 8:
                print(f"Warning: Unexpected path structure: {root}")
                continue
            
            cohort = path_after_results[0]
            
            # Determine if this is main analysis or subanalysis
            # Path structure: Cohort/[subanalysis/]outcome/classification_or_survival/method/...
            
            # Check if position 2 is classification/survival (main analysis) or another folder (subanalysis)
            if path_after_results[2] in ["classification", "survival"]:
                # Main analysis: Cohort/outcome/classification_or_survival/method/...
                subanalysis = None
                outcome = path_after_results[1]
                analysis_type = path_after_results[2]
                method = path_after_results[3]
                modality = path_after_results[-2]
            elif len(path_after_results) >= 9 and path_after_results[3] in ["classification", "survival"]:
                # Subanalysis: Cohort/subanalysis/outcome/classification_or_survival/method/...
                subanalysis = path_after_results[1]
                outcome = path_after_results[2]
                analysis_type = path_after_results[3]
                method = path_after_results[4]
                modality = path_after_results[-2]
            else:
                print(f"Warning: Could not determine structure for {root}")
                continue
            
            csv_data.append({
                'cohort': cohort,
                'subanalysis': subanalysis if subanalysis else None,
                'outcome': outcome,
                'analysis_type': analysis_type,  # classification or survival
                'method': method,  # cross_validation, standard, or evaluation
                'modality': modality,
                'auc_file': root_path / "eval_auc_ci.csv" if "eval_auc_ci.csv" in files else (root_path / "eval_cindex_ci.csv" if "eval_cindex_ci.csv" in files else None),
                'metrics_file': root_path / "eval_classification_metrics.csv" if "eval_classification_metrics.csv" in files else None,
                'path': root_path
            })
    
    return csv_data

# Find all CSV files
print(f"Scanning directory: {RESULTS_DIR}")
csv_files = find_csv_files_in_structure(RESULTS_DIR)
print(f"Found {len(csv_files)} seed_0 directories with CSV files")

if len(csv_files) == 0:
    print("ERROR: No CSV files found. Please check the RESULTS_DIR path.")
    exit(1)

# Display sample of found files
print("\nSample of found files:")
for i, item in enumerate(csv_files[:3]):
    print(f"\n{i+1}. Outcome: {item['outcome']}, Subanalysis: {item['subanalysis']}, Method: {item['method']}, Modality: {item['modality']}")
    print(f"   AUC file: {item['auc_file']}")
    print(f"   Metrics file: {item['metrics_file']}")

print("\n" + "="*80)
print("STEP 2: CONSOLIDATING FILES BY OUTCOME/METHOD")
print("="*80)

# Group by outcome, subanalysis, and method to create consolidated files
from collections import defaultdict

auc_groups = defaultdict(list)
metrics_groups = defaultdict(list)

for item in csv_files:
    key = (item['outcome'], item['subanalysis'], item['method'])
    
    if item['auc_file'] and item['auc_file'].exists():
        auc_groups[key].append(item)
    
    if item['metrics_file'] and item['metrics_file'].exists():
        metrics_groups[key].append(item)

print(f"Found {len(auc_groups)} unique AUC file groups")
print(f"Found {len(metrics_groups)} unique metrics file groups")

# Consolidate AUC files (line_plot files)
print("\nConsolidating AUC files...")
for (outcome, subanalysis, method), items in auc_groups.items():
    consolidated_rows = []
    
    for item in items:
        try:
            df = pd.read_csv(item['auc_file'])
            
            # Rename 'auc' or 'cindex' column to 'Score' if it exists
            # Rename 'auc' or 'cindex' column to 'Score' if it exists
            if 'auc' in df.columns:
                df = df.rename(columns={'auc': 'Score'})
            elif 'c_index' in df.columns:
                df = df.rename(columns={'c_index': 'Score'})
            
            # Add modality column if not present
            if 'Modality' not in df.columns:
                df['Modality'] = item['modality']
            else:
                # Ensure modality is set correctly
                df['Modality'] = item['modality']
            consolidated_rows.append(df)
        except Exception as e:
            print(f"Error reading {item['auc_file']}: {e}")
    
    if consolidated_rows:
        consolidated_df = pd.concat(consolidated_rows, ignore_index=True)
        
        # Create filename: line_plot-{subanalysis}-{method}-{outcome}.csv
        # If subanalysis is None (main analysis), format is: line_plot-{method}-{outcome}.csv
        if subanalysis:
            filename = f"line_plot-{subanalysis}-{method}-{outcome}.csv"
        else:
            filename = f"line_plot-{method}-{outcome}.csv"
        
        output_path = LINE_DATA_DIR / filename
        
        consolidated_df.to_csv(output_path, index=False)
        print(f"Created: {filename} ({len(consolidated_df)} rows, {len(items)} modalities)")

# Consolidate metrics files
print("\nConsolidating metrics files...")
for (outcome, subanalysis, method), items in metrics_groups.items():
    consolidated_rows = []
    
    for item in items:
        try:
            df = pd.read_csv(item['metrics_file'])
            # Add modality column if not present
            if 'Modality' not in df.columns:
                df['Modality'] = item['modality']
            else:
                # Ensure modality is set correctly
                df['Modality'] = item['modality']
            consolidated_rows.append(df)
        except Exception as e:
            print(f"Error reading {item['metrics_file']}: {e}")
    
    if consolidated_rows:
        consolidated_df = pd.concat(consolidated_rows, ignore_index=True)
        
        # Create filename: classification_metrics-{subanalysis}-{method}-{outcome}.csv
        # If subanalysis is None (main analysis), format is: classification_metrics-{method}-{outcome}.csv
        if subanalysis:
            filename = f"classification_metrics-{subanalysis}-{method}-{outcome}.csv"
        else:
            filename = f"classification_metrics-{method}-{outcome}.csv"
        
        output_path = METRICS_DIR / filename
        
        consolidated_df.to_csv(output_path, index=False)
        print(f"Created: {filename} ({len(consolidated_df)} rows, {len(items)} modalities)")

print("\n" + "="*80)
print("STEP 3: PROCESSING CONSOLIDATED FILES INTO FINAL DATAFRAME")
print("="*80)

# Now use the original processing logic on the consolidated files
data_folder = str(LINE_DATA_DIR)
metrics_folder = str(METRICS_DIR)

# create a empty dataframe with columns
df = pd.DataFrame(columns=['outcome', 'modality', 'subanalysis', 'cv-auc/c-index', 'cv-f1', 
                          'cv-specificity', 'cv-sensitivity', 'test-auc/c-index', 'test-f1', 
                          'test-specificity', 'test-sensitivity', 'ext_val-auc/c-index', 
                          'ext_val-f1', 'ext_val-specificity', 'ext_val-sensitivity'])

# Function to parse filename and extract components
def parse_filename(filename):
    # Remove .csv extension and split by dashes
    base_name = filename.replace('.csv', '')
    parts = base_name.split('-')
    
    # Format is either:
    # line_plot-{method}-{outcome}.csv (main analysis, 3 parts after splitting)
    # line_plot-{subanalysis}-{method}-{outcome}.csv (subanalysis, 4+ parts)
    
    # Extract method (cross_validation, standard, evaluation)
    method = None
    method_idx = None
    for i, part in enumerate(parts):
        if part in ['cross_validation', 'standard', 'evaluation']:
            method = part
            method_idx = i
            break
    
    if method_idx is None:
        return None, None, None
    
    # Outcome is everything after the method
    outcome = '-'.join(parts[method_idx + 1:])
    
    # Subanalysis is everything between line_plot and method
    # parts[0] is 'line_plot', parts[1:method_idx] is subanalysis (if exists)
    subanalysis_parts = parts[1:method_idx]
    if subanalysis_parts:
        subanalysis = '-'.join(subanalysis_parts)
    else:
        subanalysis = None
    
    return outcome, subanalysis, method

# Function to format score with confidence interval
def format_score_with_ci(score, ci_lower, ci_upper):
    # Calculate the +/- value (average of the differences from the score)
    lower_diff = score - ci_lower
    upper_diff = ci_upper - score
    ci_range = (lower_diff + upper_diff) / 2
    
    return f"{score:.3f} ± {ci_range:.3f}"

# Get all CSV files in data folder
csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
print(f"Processing {len(csv_files)} line plot files...")

# Process each file
rows = []
for file_path in csv_files:
    filename = os.path.basename(file_path)
    
    # Skip non-CSV files
    if not filename.endswith('.csv'):
        continue
    
    # Parse filename
    outcome, subanalysis, method = parse_filename(filename)
    
    # Read the CSV file
    try:
        file_df = pd.read_csv(file_path)
        
        # Rename 'auc' or 'cindex' column to 'Score' if it exists
        if 'auc' in file_df.columns:
            file_df = file_df.rename(columns={'auc': 'Score'})
        elif 'c_index' in file_df.columns:
            file_df = file_df.rename(columns={'c_index': 'Score'})
        
        # Process each row in the file
        for _, row in file_df.iterrows():
            modality = row['Modality']
            score = row['Score']
            ci_lower = row.get('ci_lower') or row.get('CI_Lower')
            ci_upper = row.get('ci_upper') or row.get('CI_Upper')
            
            # Format score with confidence interval
            formatted_score = format_score_with_ci(score, ci_lower, ci_upper)
            
            # Create row data
            row_data = {
                'outcome': outcome,
                'modality': modality,
                'subanalysis': subanalysis,
                'cv-auc/c-index': formatted_score if method == 'cross_validation' else None,
                'cv-f1': None,
                'cv-specificity': None,
                'cv-sensitivity': None,
                'test-auc/c-index': formatted_score if method == 'standard' else None,
                'test-f1': None,
                'test-specificity': None,
                'test-sensitivity': None,
                'ext_val-auc/c-index': formatted_score if method == 'evaluation' else None,
                'ext_val-f1': None,
                'ext_val-specificity': None,
                'ext_val-sensitivity': None
            }
            
            rows.append(row_data)
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Create the populated dataframe
populated_df = pd.DataFrame(rows)

# Group by outcome, modality, subanalysis and combine cv, test, and evaluation scores
final_rows = []
grouped = populated_df.groupby(['outcome', 'modality', 'subanalysis'])

for (outcome, modality, subanalysis), group in grouped:
    cv_score = group[group['cv-auc/c-index'].notna()]['cv-auc/c-index'].iloc[0] if any(group['cv-auc/c-index'].notna()) else None
    test_score = group[group['test-auc/c-index'].notna()]['test-auc/c-index'].iloc[0] if any(group['test-auc/c-index'].notna()) else None
    ext_val_score = group[group['ext_val-auc/c-index'].notna()]['ext_val-auc/c-index'].iloc[0] if any(group['ext_val-auc/c-index'].notna()) else None
    
    final_rows.append({
        'outcome': outcome,
        'modality': modality,
        'subanalysis': subanalysis,
        'cv-auc/c-index': cv_score,
        'cv-f1': None,
        'cv-specificity': None,
        'cv-sensitivity': None,
        'test-auc/c-index': test_score,
        'test-f1': None,
        'test-specificity': None,
        'test-sensitivity': None,
        'ext_val-auc/c-index': ext_val_score,
        'ext_val-f1': None,
        'ext_val-specificity': None,
        'ext_val-sensitivity': None
    })

# Create final dataframe
df = pd.DataFrame(final_rows)

print(f"\nPopulated dataframe with {len(df)} rows")

# Read metrics data and populate additional columns
metrics_files = glob.glob(os.path.join(metrics_folder, '*.csv'))
print(f"\nProcessing {len(metrics_files)} metrics files...")

# Process metrics files to populate f1, specificity, sensitivity columns
for file_path in metrics_files:
    filename = os.path.basename(file_path)
    
    if not filename.endswith('.csv'):
        continue
    
    # Parse filename - similar to line files but with classification_metrics prefix
    base_name = filename.replace('.csv', '').replace('classification_metrics-', '')
    parts = base_name.split('-')
    
    # Format is either:
    # classification_metrics-{method}-{outcome}.csv (main analysis)
    # classification_metrics-{subanalysis}-{method}-{outcome}.csv (subanalysis)
    
    # Extract method (cross_validation, standard, evaluation)
    method = None
    method_idx = None
    for i, part in enumerate(parts):
        if part in ['cross_validation', 'standard', 'evaluation']:
            method = part
            method_idx = i
            break
    
    if method_idx is None:
        continue
    
    # Outcome is everything after the method
    outcome = '-'.join(parts[method_idx + 1:])
    
    # Subanalysis is everything before method
    subanalysis_parts = parts[:method_idx]
    if subanalysis_parts:
        subanalysis = '-'.join(subanalysis_parts)
    else:
        subanalysis = None
    
    try:
        metrics_df = pd.read_csv(file_path)
        
        # Process each row in the metrics file
        for _, row in metrics_df.iterrows():
            modality = row['Modality']
            f1_score = f"{row['f1']:.3f}"
            specificity_score = f"{row['specificity']:.3f}"
            sensitivity_score = f"{row['sensitivity']:.3f}"
            
            # Find matching row in main dataframe
            mask = (df['outcome'] == outcome) & (df['modality'] == modality) & (df['subanalysis'] == subanalysis)
            matching_rows = df[mask]
            
            if len(matching_rows) > 0:
                idx = matching_rows.index[0]
                
                # Update the appropriate columns based on method
                if method == 'cross_validation':
                    df.loc[idx, 'cv-f1'] = f1_score
                    df.loc[idx, 'cv-specificity'] = specificity_score
                    df.loc[idx, 'cv-sensitivity'] = sensitivity_score
                elif method == 'standard':
                    df.loc[idx, 'test-f1'] = f1_score
                    df.loc[idx, 'test-specificity'] = specificity_score
                    df.loc[idx, 'test-sensitivity'] = sensitivity_score
                elif method == 'evaluation':
                    df.loc[idx, 'ext_val-f1'] = f1_score
                    df.loc[idx, 'ext_val-specificity'] = specificity_score
                    df.loc[idx, 'ext_val-sensitivity'] = sensitivity_score
    
    except Exception as e:
        print(f"Error processing metrics file {filename}: {e}")

print("Metrics data added successfully!")

# Sort by subanalysis first and then outcome
df = df.sort_values(by=['subanalysis', 'outcome'])

# Save complete dataframe
output_file = CONSOLIDATED_DIR / 'DL-res.csv'
df.to_csv(output_file, index=False)
print(f"\n{'='*80}")
print(f"COMPLETE! Final dataframe saved to: {output_file}")
print(f"{'='*80}")

# Final summary
print(f"\nFinal Summary:")
print(f"Total rows: {len(df)}")
print(f"Rows with CV data: {df['cv-auc/c-index'].notna().sum()}")
print(f"Rows with Test data: {df['test-auc/c-index'].notna().sum()}")
print(f"Rows with Evaluation data: {df['ext_val-auc/c-index'].notna().sum()}")
print(f"\nUnique outcomes: {df['outcome'].nunique()}")
print(f"Unique subanalyses: {df['subanalysis'].nunique()}")
print(f"Unique modalities: {df['modality'].nunique()}")

# Show sample data
print(f"\nSample of final data:")
print(df.head(10).to_string(index=False))