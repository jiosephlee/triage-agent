import time
import re
from utils.keys import OPENAI_API_KEY, DATABRICKS_TOKEN, ANTHROPIC_KEY, GEMINI_KEY
import os
from openai import OpenAI
import anthropic
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

def query_llm(prompt, max_tokens=1000, temperature=0, top_p=0, max_try_num=10, model="gpt-4o-mini", debug=False, return_json=False, json_schema=None, logprobs=False, system_prompt_included=True, is_hippa=False):
    if debug:
        if system_prompt_included:
            print(f"System prompt: {prompt['system']}")
            print(f"User prompt: {prompt['user']}")
        else:
            print(prompt)
        print(f"Model: {model}")
    if is_hippa and ('gpt' not in model and 'o3' not in model):
        raise ValueError("HIPPA compliance requires GPT models")
    curr_try_num = 0
    while curr_try_num < max_try_num:
        try:
            if 'gpt' in model or 'o3' in model:
                response = query_gpt(prompt, model=model, temperature=temperature, top_p=top_p, return_json=return_json, json_schema=json_schema, logprobs=logprobs, system_prompt_included=system_prompt_included, is_hippa=is_hippa, debug=debug)
                if logprobs:
                    return response.choices[0].message.content.strip(), response.choices[0].logprobs
            elif 'claude' in model:
                response = query_claude(prompt, model=model, temperature=temperature, max_tokens=max_tokens, system_prompt_included=system_prompt_included, debug=debug)
                if return_json:
                    return re.sub(r'(?<!\\)\n', '', response)
                return response
            elif 'gemini' in model:
                full_prompt = f"{prompt['system']}\n\n{prompt['user']}"
                return query_gemini(full_prompt, model=model, temperature=temperature, max_tokens=max_tokens, system_prompt_included=system_prompt_included, debug=debug)
            return response
        except Exception as e:
            if 'gpt' in model:
                print(f"Error making OpenAI API call: {e}")
            else: 
                print(f"Error making API call: {e}")
            curr_try_num += 1
            time.sleep(10)
    return None

def query_gpt(prompt: str | dict, model: str = 'gpt-4o-mini', temperature: float = 0, top_p: float = 0, logprobs: bool = False, return_json: bool = False, json_schema = None, system_prompt_included: bool = False, is_hippa: bool = False, debug: bool = False):
    """OpenAI API wrapper; For HIPPA compliance, use client_safe e.g. model='openai-gpt-4o-high-quota-chat'"""
    temp_client = client_safe if is_hippa else client
    if system_prompt_included:
        # Format chat prompt with system and user messages
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    if 'o3' in model:
        api_params = {
            "model": model,
            "reasoning_effort": "high",
            "messages": messages,
        }
    else:
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "seed": 0
        }
    if logprobs:
        api_params["logprobs"] = logprobs
        api_params["top_logprobs"] = 3

    if return_json:
        if json_schema is None:
            api_params["response_format"] = {"type": "json_object"}
            completion = temp_client.chat.completions.create(**api_params)
            response = completion.choices[0].message.content.strip()
        else:
            print(json_schema)
            api_params["response_format"] = json_schema
            completion = temp_client.chat.completions.create(**api_params)
            response = completion.choices[0].message.content.strip()
            # completion = client.beta.chat.completions.parse(**api_params)
            # response = completion.choices[0].message.parsed
    else: 
        completion = temp_client.chat.completions.create(**api_params)
        response = completion.choices[0].message.content.strip()
    if debug:
        print(f"Response: {response}")
    if logprobs:
        return response, completion.choices[0].logprobs
    else:
        return response

def query_claude(prompt: str | dict, model: str, temperature: float, max_tokens: int, system_prompt_included: bool = False):
    try:
        if system_prompt_included:
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

def query_gemini(message, model, temperature=0, max_tokens=1000, return_json=False, response_schema=None):
    config = types.GenerateContentConfig(
        temperature=temperature,
        seed=0,
        max_output_tokens=max_tokens,
    )
    
    if return_json:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema
    
    response = gemini_client.models.generate_content(
        model=model,
        contents=message,
        config=config,
    )

    return response.text
 