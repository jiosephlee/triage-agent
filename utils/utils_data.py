import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

"""We store unanimously required & unique characteristics of the datasets here."""
DATASETS = {
    "ktas_triage": {
        "test_path": "./data/kaggle/test.csv",
        "train_path": "./data/kaggle/train.csv",
        "embeddings": {
            'complaint':{
                "train": "./data/kaggle/KTAS_train_chiefcomplaint_embeddings.npy",
                "test": "./data/kaggle/KTAS_test_chiefcomplaint_embeddings.npy",
            },
            'diagnosis':{
                "train": "./data/kaggle/KTAS_train_diagnosis_embeddings.npy",
                "test": "./data/kaggle/KTAS_test_diagnosis_embeddings.npy",
            }
        },
        "format": "csv",
        "target_column": "KTAS_expert",
        "is_hippa": False,
    },
    "esi_handbook": {
        "test_path": "./data/ESI-Handbook/train.csv",
        "train_path": "./data/ESI-Handbook/test.csv",
        "embeddings": {
            "train": "./data/ESI-Handbook/train_embeddings.npy",
            "test": "./data/ESI-Handbook/test_embeddings.npy",
        },
        "format": "csv",
        "target_column": "acuity",
        "is_hippa": False,
    },
}
             
def load_dataset(filepath, format, start_index, end_index):
    if not filepath:
        raise ValueError("Dataset not found in _DATASETS.")
    if format == 'jsonl':
        data = load_jsonl(filepath, start_index, end_index)
    elif format == 'csv':
        data = pd.read_csv(filepath).loc[start_index:end_index]
    else:
        raise ValueError(f"Unsupported format: {format}")
    return data
    
def load_jsonl(filepath, start_index, end_index):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for i, line in enumerate(f) if i <= end_index and i >= start_index]
    return data

def load_predictions(filename, format='csv', save_path="./results/"):
    """Load predictions from a file."""
    if format == 'csv':
        predictions_file = os.path.join(save_path, f"{filename}.csv")
        predictions = pd.read_csv(predictions_file)
    else: 
        predictions_file = os.path.join(save_path, f"{filename}.txt")
        with open(predictions_file, 'r') as f:
            predictions = [json.loads(line.strip()) for line in f]
    return predictions

def get_stratified_df(df, target_col, test_size, seed=0):
    # Define stratified shuffle split
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    
    # Get the stratified sample indices
    for train_indices, test_indices in stratified_split.split(df, df[target_col]):
        stratified_train_df = df.iloc[train_indices]
        stratified_test_df = df.iloc[test_indices]
    
    return stratified_train_df.reset_index(drop=True), stratified_test_df.reset_index(drop=True)