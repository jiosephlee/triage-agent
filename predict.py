import yaml
import argparse
import pandas as pd
import json
from datetime import datetime
import utils.utils as utils
from utils.dataset import DATASETS
import utils.predictors as predictors
import time
import os
import re
import uuid # Or use timestamp + random string for run_id
import subprocess # For git hash
import traceback # For logging errors
from tqdm import tqdm 

def get_git_info():
    """Gets the current git commit hash and status."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        status_output = subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
        status = "clean" if not status_output else "dirty"
        return commit_hash, status
    except Exception as e:
        print(f"Warning: Could not get git info: {e}")
        return None, None

def log_experiment_run(log_file_path, run_data):
    """Appends a run's data as a JSON line to the master log file."""
    # Use file locking if parallel runs are a possibility, otherwise simple append is often fine
    try:
        with open(log_file_path, 'a') as f:
            f.write(json.dumps(run_data) + '\n')
    except Exception as e:
        print(f"ERROR: Could not write to master log file {log_file_path}: {e}")
        return "Error: No acuity number found" 
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LLM Triage Experiments.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--debug", action="store_true", default=None)

    cli_args = parser.parse_args()

    # --- 1. Load Configuration ---
    with open(cli_args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides (Example for debug)
    if cli_args.debug is not None:
        config['debug'] = cli_args.debug

    # --- 2. Setup Run Environment ---
    dataset_name = config['dataset']['name']

    timestamp_start_dt = datetime.now()
    timestamp_start_iso = timestamp_start_dt.isoformat()
    run_id = f"{timestamp_start_dt.strftime('%Y-%m-%d_%H:%M')}_{uuid.uuid4().hex[:6]}"
    git_hash, git_status = get_git_info()
    run_output_dir = os.path.join(config.get('results_base_dir', './results'), dataset_name, config['model']['name'], run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    master_log_file = os.path.join(config.get('results_base_dir', './results'), 'experiment_log.jsonl')

    # --- 3. Prepare Log Entry (can update status/end time later) ---
    run_data = {
        "run_id": run_id,
        "timestamp_start": timestamp_start_iso,
        "timestamp_end": None,
        "duration_seconds": None,
        "status": "running",
        "git_commit_hash": git_hash,
        "git_status": git_status,
        "config_file_path": os.path.abspath(cli_args.config),
        "parameters": config, # Log the final effective parameters
        "metrics": None,
        "output_dir": os.path.abspath(run_output_dir),
        "error_message": None,
        "notes": config.get("run_notes", None) # Add an optional 'run_notes' field to YAML
    }
    # Optional: Log initial 'running' status
    # log_experiment_run(master_log_file, run_data)

    try:
        # --- 4. Run Experiment ---
        print(f"Starting Run ID: {run_id}")
        print(f"Output Directory: {run_output_dir}")
        print(f"Effective Config: {json.dumps(config, indent=2)}")

        # ... [Your existing data loading logic using config] ...
        test_dataset = utils.load_dataset(dataset_name, config['dataset']['start_index'], config['dataset']['end_index'])

        # ... [Your existing predictor setup logic using config] ...
        if dataset_name == 'ktas-triage-multi':
            predictor = predictors.MultiTriagePredictor(dataset=dataset_name, strategy=config['predictor']['strategy'])
        elif dataset_name == 'ktas-triage':
            predictor = predictors.TriagePredictor(dataset=dataset_name, strategy=config['predictor']['strategy'])
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        predictions = [] # Load existing if needed based on config
        # ... [Your existing prediction loop logic using config] ...
        # Important: Modify predict() and other functions to accept config dict or specific params from it

        for i, row in tqdm(test_dataset.iterrows()): # Adjust slicing based on config/resume logic
            prompt, prediction = predictor.predict(text=row[DATASETS[dataset_name]['x_column']], 
                                 model_config=config['model'], 
                                 k_shots=config['predictor']['k_shots'], 
                                return_json=config['model']['return_json'], 
                                 use_json_schema=config['model']['use_json_schema'], 
                                 debug=config.get('debug', False)) # Pass config values
            predictions.append(prediction)
            # Checkpoint saving logic (optional, save within run_output_dir)

        predictions_df = pd.DataFrame(predictions)

        # --- 5. Save Artifacts ---
        predictions_filename = "predictions.csv"
        predictions_filepath = os.path.join(run_output_dir, predictions_filename)
        predictions_df.to_csv(predictions_filepath, index=False)
        print(f"Predictions saved to {predictions_filepath}")

        # Optional: Save a copy of the config used in the run directory
        config_copy_path = os.path.join(run_output_dir, "config_used.yaml")
        with open(config_copy_path, 'w') as f:
            yaml.dump(config, f)


        # --- 6. Evaluate ---
        target_column = DATASETS[dataset_name]['target_column']
        if config.get('evaluate', True):
            print("Evaluating predictions...")
            ground_truths = test_dataset[target_column]
            # Ensure Estimated_Acuity exists
            if target_column not in predictions_df.columns:
                 raise ValueError(f"'{target_column}' column not found in predictions DataFrame.")
            preds = predictions_df[target_column]
            metrics = utils.evaluate_predictions(preds, ground_truths, ordinal=True, by_class=True)
            print("Overall Metrics:", metrics)

            # Save metrics JSON
            metrics_filename = "metrics.json"
            metrics_filepath = os.path.join(run_output_dir, metrics_filename)
            with open(metrics_filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to {metrics_filepath}")

            run_data["metrics"] = metrics # Update run data
            run_data["status"] = "completed"

        else:
             print("Evaluation skipped.")
             run_data["status"] = "completed_no_eval"

    except Exception as e:
        print(f"ERROR: Run {run_id} failed!")
        traceback.print_exc() # Print detailed error traceback
        run_data["status"] = "failed"
        run_data["error_message"] = str(e)

    finally:
        # --- 7. Log Final Run Data ---
        timestamp_end_dt = datetime.now()
        run_data["timestamp_end"] = timestamp_end_dt.isoformat()
        run_data["duration_seconds"] = (timestamp_end_dt - timestamp_start_dt).total_seconds()
        log_experiment_run(master_log_file, run_data)
        print(f"Run {run_id} logged to {master_log_file} with status: {run_data['status']}")