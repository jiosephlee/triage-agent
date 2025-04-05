import yaml
import argparse
import pandas as pd
import json
from datetime import datetime
import utils.utils as utils
import utils.utils_data as utils_data
import utils.predictors as predictors
import time
import os
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

def extract_num(answer_text):
    # Use regex to find all floats and integers in the answer text
    matches = re.findall(r'\d+\.\d+|\d+', answer_text)
    
    # Convert all matches to floats for consistent processing
    matches = [float(num) for num in matches]

    if len(matches) == 1:
        return matches[0]  # Return the single found number
    elif len(matches) > 1:
        return f"Error: Multiple numbers found. Please verify the data: {answer_text}"
    else:
        return "Error: No acuity number found" 
    
def extract_acuity_from_text(text, debug):
    # Call another model to extract the acuity if necessary
    # Split the text into lines
    lines = text.splitlines()
    last_five_lines = lines[-5:]
    text = "\n".join(last_five_lines)
    answer_text = utils.query_llm(f"Extract the estimated acuity from the following information and output the number alone. If the estimate is uncertain, just choose one that is best.\n\n\"\"\"{text}.\"\"\"", 
                                       model= "gpt-4o-mini", debug=debug)
    num = extract_num(answer_text)
    time.sleep(1)
    if type(num) == str and 'Error' in num:
        return utils.query_llm(f"Extract the estimated acuity from the following information and output the number alone. If the estimate is uncertain, just choose one that is best.\n\n\"\"\"{text}.\"\"\"", 
                                       model= "gpt-4o-mini", debug=debug)
    else:
        return num

def predict(*, index, row, predictor, model, strategy, return_json, k_shots, serialization, vitals_off, bias, debug=False):
    prompt, response = predictor.predict(
        row=row,
        model=model,
        k_shots=k_shots,
        return_json=return_json,
        serialization_strategy=serialization,
        vitals_off=vitals_off,
        bias=bias,
        debug=debug)
    if return_json:
        try:
            response_data = json.loads(response)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            print("Raw response causing error:", response)
            response = utils.query_llm(response + "\n\nCan you format the above in proper JSON", model ='gpt-4o-mini',return_json=True)
            response_data = json.loads(response)
        if index==0:
            return {
                "prompt": prompt,
                "Estimated_Acuity": response_data['Acuity'],
                "Reasoning": response_data['Reasoning'] if 'CoT' in strategy else None,
                **row.to_dict()  # Include the original row's data for reference
            }
        else:
            return {
                "Estimated_Acuity": response_data['Acuity'],
                "Reasoning": response_data['Reasoning'] if 'CoT' in strategy else None,
                **row.to_dict()  # Include the original row's data for reference
            }
    else: 
        if index==0:
            return {
                "prompt": prompt,
                "Estimated_Acuity": extract_acuity_from_text(response, debug=debug),
                "Reasoning": response if 'CoT' in strategy else None,
                **row.to_dict()  # Include the original row's data for reference
            }
        else:
            return {
                "Estimated_Acuity": extract_acuity_from_text(response, debug=debug),
                "Reasoning": response if 'CoT' in strategy else None,
                **row.to_dict()  # Include the original row's data for reference
            }
            
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
    timestamp_start_dt = datetime.now()
    timestamp_start_iso = timestamp_start_dt.isoformat()
    run_id = f"{timestamp_start_dt.strftime('%Y-%m-%d_%H:%M')}_{uuid.uuid4().hex[:6]}"
    git_hash, git_status = get_git_info()

    dataset_name = config['dataset']['name']
    run_output_dir = os.path.join(config.get('results_base_dir', './results'), dataset_name, run_id)
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
        test_dataset = utils_data.load_dataset(dataset_name)

        # ... [Your existing predictor setup logic using config] ...
        predictor = predictors.Predictor(...)

        predictions = [] # Load existing if needed based on config
        # ... [Your existing prediction loop logic using config] ...
        # Important: Modify predict() and other functions to accept config dict or specific params from it

        for i, row in tqdm(test_dataset.iterrows()): # Adjust slicing based on config/resume logic
            prediction = predict(..., debug=config.get('debug', False)) # Pass config values
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
        if config.get('evaluate', True):
            print("Evaluating predictions...")
            ground_truths = test_dataset[utils_data._DATASETS[dataset_name]['target_column']]
            # Ensure Estimated_Acuity exists
            if 'Estimated_Acuity' not in predictions_df.columns:
                 raise ValueError("'Estimated_Acuity' column not found in predictions DataFrame.")
            preds = predictions_df['Estimated_Acuity']
            metrics = utils.evaluate_predictions(preds, ground_truths, ordinal=True, by_class=True)
            print("Overall Metrics:", metrics)

            # Save metrics JSON
            metrics_filename = "metrics.json"
            metrics_filepath = os.path.join(run_output_dir, metrics_filename)
            with open(metrics_filepath, 'w') as f:
                # Convert numpy types for JSON serialization if necessary
                metrics_serializable = utils.make_dict_json_serializable(metrics) # Assumes you have/create this helper
                json.dump(metrics_serializable, f, indent=2)
            print(f"Metrics saved to {metrics_filepath}")

            run_data["metrics"] = metrics_serializable # Update run data
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