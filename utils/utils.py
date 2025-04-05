from utils.keys import OPENAI_API_KEY, TOGETHER_API_KEY, DATABRICKS_TOKEN, ANTHROPIC_KEY, GEMINI_KEY
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, 
                             classification_report, mean_absolute_error, mean_squared_error)
from sklearn.model_selection import StratifiedShuffleSplit
from openai import OpenAI
import pandas as pd
import anthropic
import os
import time 
import re
import json 
from google import genai
from google.genai import types
from pydantic import BaseModel

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

client_safe = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://adb-4750903324350629.9.azuredatabricks.net/serving-endpoints"
)

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

gemini_client = genai.Client(api_key=GEMINI_KEY)

class Acuity(BaseModel):
    """This is for gemini & openai to return a json object with the acuity level"""
    Acuity: int
    
class AcuityWithReasoning(BaseModel):
    Acuity: int
    Reasoning: str

        
###################
## Evaluation
###################

def evaluate_predictions(predicted_answers, correct_answers, ordinal=True,flexibility=1, by_class=False):
    """
    Evaluate predictions with standard metrics, Quadratic Weighted Kappa, and optionally per-class analysis.

    Args:
        predicted_answers (list): List of predicted class labels.
        correct_answers (list): List of true class labels.
        flexibility (int): Tolerance for flexibility in metrics (e.g., ±1).
        by_class (bool): Whether to compute metrics by class.

    Returns:
        dict: A dictionary containing overall and optionally per-class evaluation metrics.
    """
    # Standard metrics
    accuracy = accuracy_score(correct_answers, predicted_answers)
    precision = precision_score(correct_answers, predicted_answers, average='weighted')
    recall = recall_score(correct_answers, predicted_answers, average='weighted')
    f1 = f1_score(correct_answers, predicted_answers, average='weighted')
    
    # Adjust predictions for flexibility
    adjusted_predictions = [
        true if abs(true - pred) <= flexibility else pred
        for true, pred in zip(correct_answers, predicted_answers)
    ]

    adjusted_accuracy = accuracy_score(correct_answers, adjusted_predictions)
    adjusted_precision = precision_score(correct_answers, adjusted_predictions, average='weighted')
    adjusted_recall = recall_score(correct_answers, adjusted_predictions, average='weighted')
    adjusted_f1 = f1_score(correct_answers, adjusted_predictions, average='weighted')

    # Error metrics (MAE and MSE)
    mae = mean_absolute_error(correct_answers, predicted_answers)
    mse = mean_squared_error(correct_answers, predicted_answers)
    # Metrics by class
    if by_class:
        report = classification_report(correct_answers, predicted_answers, output_dict=True)

    # Consolidate results
    results = {
        "overall": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mae": mae,
            "mse": mse,
            "adjusted_accuracy": adjusted_accuracy,
            "adjusted_precision": adjusted_precision,
            "adjusted_recall": adjusted_recall,
            "adjusted_f1": adjusted_f1,
        }
    }
    if ordinal:
        qwk = cohen_kappa_score(correct_answers, predicted_answers, weights="quadratic")
        results['overall']["quadratic_kappa"] = qwk
    if by_class:
        results["by_class"] = report

    return results

###################
## API Calling
###################

def query_llm(prompt, max_tokens=1000, temperature=0, top_p=0, max_try_num=10, model="gpt-4o-mini", debug=False, return_json=False, logprobs=False, is_prompt_full=False):
    if debug:
        if is_prompt_full:
            print(f"System prompt: {prompt['system']}")
            print(f"User prompt: {prompt['user']}")
        else:
            print(prompt)
        print(f"Model: {model}")
    curr_try_num = 0
    while curr_try_num < max_try_num:
        try:
            if 'gpt' in model:
                response = query_gpt(prompt, model, temperature, top_p, logprobs, return_json, is_prompt_full=is_prompt_full)
            elif 'claude' in model:
                response = query_claude(prompt, model, temperature, max_tokens, is_prompt_full=is_prompt_full)
                if return_json:
                    return re.sub(r'(?<!\\)\n', '', response)
                return response
            elif 'gemini' in model:
                full_prompt = f"{prompt['system']}\n\n{prompt['user']}"
                return query_gemini(full_prompt, model, temperature, max_tokens, is_prompt_full=is_prompt_full)
            if debug:
                print(response.choices[0].message.content.strip())
            if logprobs:
                return response.choices[0].message.content.strip(), response.choices[0].logprobs
            return response.choices[0].message.content.strip()
        except Exception as e:
            if 'gpt' in model:
                print(f"Error making OpenAI API call: {e}")
            else: 
                print(f"Error making API call: {e}")
            curr_try_num += 1
            time.sleep(10)
    return None

def query_gpt_safe(prompt, model="openai-gpt-4o-high-quota-chat", return_json=False, temperature=0.0, max_tokens=1000, debug=False, tries=0, is_prompt_full=False):
    time.sleep(0.5)
    if debug:
        print(prompt)
    try:
        if is_prompt_full:
            # Format chat prompt with system and user messages
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        if 'o3' in model:
            if return_json:
                response = client_safe.chat.completions.create(
                    model=model,
                    reasoning_effort="medium",
                    messages=messages,
                    response_format={"type": "json_object"}
                )
            else:
                response = client_safe.chat.completions.create(
                    model=model,
                    reasoning_effort="medium",
                    messages=messages,
                )
        elif return_json:
            response = client_safe.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
        else:
            response = client_safe.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        response = response.choices[0].message.content.strip()
        if debug:
            print(response)
        return response
    except Exception as e:
        if tries >= 1:
            print(f"Error in query_gpt_safe after {tries + 1} tries: {e}")
            return None
            
        print("Error in query_gpt_safe. Waiting before retrying...")
        time.sleep(10)  # Wait 10 seconds before retrying
        return query_gpt_safe(prompt, model, return_json, temperature, max_tokens, debug, tries + 1, is_prompt_full)

def query_claude(prompt: str | dict, model: str, temperature: float, max_tokens: int, is_prompt_full: bool = False):
    try:
        if is_prompt_full:
            # Format chat prompt with system and user messages
            messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = claude_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error calling Claude: {e}")
        return None

def query_gemini(message, model, temperature=0, max_tokens=1000):
    response = gemini_client.models.generate_content(
        model=model,
        contents=message,
        config=types.GenerateContentConfig(
            temperature=temperature,
            seed=0,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=Acuity,
        ),
    )

    return response.text
 
def query_gpt(prompt: str | dict, model: str = 'gpt-4o-mini', temperature: float = 0, top_p: float = 0, logprobs: bool = False, return_json: bool = False, is_prompt_full: bool = False):
    if is_prompt_full:
        # Format chat prompt with system and user messages
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    if return_json:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
    elif logprobs:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            top_logprobs=3,
            seed=0
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=0
        )
    return response