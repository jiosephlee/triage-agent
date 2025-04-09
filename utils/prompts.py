import pandas as pd
import utils.utils_triage as utils_triage

ALPACA_PROMPT = """### Instruction: {instruction}

### Input: {input}

### Response: {response}"""

BLACKBOX_LLM_PROMPT = """### Instructions
{}

### Patient's Clinical Presentation
{}"""

BLACKBOX_LLM_PROMPT_WITH_EXAMPLES = """### Instructions
{}

### Examples 
{}

### Patient's Clinical Presentation
{}"""

INSTRUCTIONS = {    
    "ktas-triage": {
        "vanilla": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": "Based on the patient's clinical presentation, please determine the appropriate KTAS acuity level on a scale from 1 to 5. KTAS 1 indicates the highest priority and KTAS 5 indicates the lowest priority."
        },
        "cot": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": """Your task is to determine the patient's KTAS acuity based on the patient's clinical presentation, where KTAS 1 indicates the highest priority and KTAS 5 indicates the lowest priority. Let's understand the KTAS levels and solve this step by step.

** Step 1: Understand KTAS Levels **
KTAS 1: Immediate aggressive treatment needed for life-threatening conditions. Requires immediate medical examination.

KTAS 2: Potential threats to life, limb or body function requiring quick intervention. Physician's/nurse's reevaluation needed within 15 minutes.

KTAS 3: Conditions that could lead to serious problems potentially requiring emergency intervention. Causes significant discomfort or impacts daily functions. Physician evaluation needed within 30 minutes.

KTAS 4: Condition associated with age-related factors or possibility of pain/complications. Treatment and re-verification within 1-2 hours. Physician evaluation needed within 60 minutes.

KTAS 5: Conditions from chronic problems with possible exacerbation. May be delayed or scheduled. Physician evaluation needed within 120 minutes.

** Step 2: Evaluate the Patient's Condition **
** Step 3: Determine Final KTAS Level Based on Assessment **"""
        },
        "autocot": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS). Think carefully step by step.",
            "user": "Based on the patient's clinical presentation, please determine the appropriate KTAS acuity level on a scale from 1 to 5. KTAS 1 indicates the highest priority and KTAS 5 indicates the lowest priority."
        },
        "fewshot": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": """Your task is to determine the patient's KTAS acuity based on the patient's clinical presentation, where KTAS 1 indicates the highest priority (requiring immediate, life-saving intervention), and KTAS 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their KTAS acuity levels, followed by a new patient case. Use the examples and your knowledge of KTAS protocol to determine the appropriate KTAS acuity level for the new patient."""
        },
        "fewshotcot": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": """Your task is to determine the patient's KTAS acuity based on the patient's clinical presentation, where KTAS 1 indicates the highest priority (requiring immediate, life-saving intervention), and KTAS 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their KTAS acuity levels, followed by a new patient case. Let's analyze the examples and solve the problem step by step. 

** Step 1: Assess the Patient's Risk**
** Step 2: Analyze and Compare with Other Patient Cases **
** Step 3: Determine the KTAS acuity level for this patient based on the examples, KTAS protocol, and your assessment of the patient's clinical presentation **"""
        },
        "kate": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": """Your task is to determine the patient's KTAS acuity based on the patient's clinical presentation, where KTAS 1 indicates the highest priority (requiring immediate, life-saving intervention), and KTAS 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their KTAS acuity levels, followed by a new patient case. Use the examples and your knowledge of KTAS protocol to determine the appropriate KTAS acuity level for the new patient."""
        },
        "katecot": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": """Your task is to determine the patient's KTAS acuity based on the patient's clinical presentation, where KTAS 1 indicates the highest priority (requiring immediate, life-saving intervention), and KTAS 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their KTAS acuity levels, followed by a new patient case. Let's analyze the examples and solve the problem step by step. 

** Step 1: Assess the Patient's Risk**
** Step 2: Analyze and Compare with Similar Patient Cases **
** Step 3: Determine the KTAS acuity level for this patient based on the examples, KTAS protocol, and your assessment of the patient's clinical presentation **"""
        },
    },
    "triage-handbook": {
        "vanilla": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": "Based on the patient's clinical presentation, please determine the appropriate ESI acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed)."
        },
        "selfconsistency": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). Let's first understand the problem and solve the problem step by step.

** Step 1: Assess Immediate Risk for ESI Level 1 **

- Is the patient dying or at immediate risk of death?
- Does the patient require an immediate life-saving intervention?

** Step 2: Evaluate Key Clinical Signs for ESI Level 2 **

Assess High-Risk Situations: Identify patients with potentially life-threatening conditions who do not require immediate life-saving interventions but should not wait. Use the rule of thumb: “Would I use my last open bed for this patient?” Key indicators include:
- Severe chest pain, suspected acute coronary syndrome (ACS)
- Respiratory distress (e.g., severe asthma, pleural effusion)
- Neurological deficits (e.g., stroke symptoms, meningitis suspicion)
- Uncontrolled bleeding or significant trauma

Evaluate Mental Status Changes: Classify patients with new onset confusion, lethargy, or disorientation as ESI Level 2. This includes:
- Suspected stroke (sudden motor/speech deficits)
- Altered mental status from metabolic, toxic, or neurological causes
- Pediatric patients with signs of sepsis or dehydration

Prioritize Severe Pain or Distress: Patients reporting severe pain (≥7/10) or severe psychological distress may qualify for Level 2 if their condition necessitates immediate intervention. Examples:
- Sickle cell crisis or acute oncologic pain
- Severe abdominal pain in elderly patients
- Trauma with high pain burden requiring urgent evaluation

** Step 3: Differentiate between ESI 3, 4 and 5 **
- Resource prediction is an integral part of the ESI for patients identified as ESI level 3 (many resources), 4 (one resource), or 5 (no resources).
- Resources include: Labs (blood, urine), ECG, X-rays, CT-MRI-ultrasound, angiography, IV fluids (hydration) , IV, IM or nebulized medications, Specialty consultation, Simple procedure (lac repair, Foley cath), Complex procedure (conscious sedation) 
- Check Danger Zone Vitals: If the patient has very high heart rate, breathing rate, or oxygen saturation below 92%, consider upgrading to ESI Level 2. Use these thresholds:
    - HR: >180 (infants), >160 (toddlers), >140 (children), >100 (older children/adults)
    - RR: >50 (infants), >40 (toddlers), >30 (children), >20 (older children/adults)
    - SaO₂: <92%

** Step 4: If it's not clear by now, determine the appropriate ESI acuity level to the best of your ability. **"""
        },
        "autocot": 
            {
            "system": "You are an expert on the Emergency Severity Index (ESI). Think carefully step by step.",
            "user": "Based on the patient's clinical presentation, please determine the appropriate ESI acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed)."
        },
        "demonstrationcot": 
            {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Based on the patient's clinical presentation, please determine the appropriate ESI acuity level on a scale from 1 to 5. ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). Here's a demonstration of how to do this:

Case: A 34-year-old obese female presents to triage complaining of generalized abdominal pain (pain scale rating: 6/10) for 2 days. She has vomited several times and states her last bowel movement was 3 days ago. She has a history of back surgery, takes no medications, and is allergic to peanuts. Vital signs: T 97.8° F, HR 104, RR 16, BP 132/80, SpO2 99 percent.
Rationale: This patient will need a minimum of two or more resources: lab, IV fluids, perhaps IV medication for nausea, and a CT scan. The triage nurse would review the patient's vital signs and consider the heart rate. The heart rate falls just outside the accepted parameter for the age of the patient but could be due to pain or exertion. In this case, the decision should be to assign the patient to ESI level 3."""
        },
        "cot": 
            {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). Let's first understand the problem and solve the problem step by step.

** Step 1: Assess Immediate Risk for ESI Level 1 **

- Is the patient dying or at immediate risk of death?
- Does the patient require an immediate life-saving intervention?

** Step 2: Evaluate Key Clinical Signs for ESI Level 2 **

Assess High-Risk Situations: Identify patients with potentially life-threatening conditions who do not require immediate life-saving interventions but should not wait. Use the rule of thumb: “Would I use my last open bed for this patient?” Key indicators include:
- Severe chest pain, suspected acute coronary syndrome (ACS)
- Respiratory distress (e.g., severe asthma, pleural effusion)
- Neurological deficits (e.g., stroke symptoms, meningitis suspicion)
- Uncontrolled bleeding or significant trauma

Evaluate Mental Status Changes: Classify patients with new onset confusion, lethargy, or disorientation as ESI Level 2. This includes:
- Suspected stroke (sudden motor/speech deficits)
- Altered mental status from metabolic, toxic, or neurological causes
- Pediatric patients with signs of sepsis or dehydration

Prioritize Severe Pain or Distress: Patients reporting severe pain (≥7/10) or severe psychological distress may qualify for Level 2 if their condition necessitates immediate intervention. Examples:
- Sickle cell crisis or acute oncologic pain
- Severe abdominal pain in elderly patients
- Trauma with high pain burden requiring urgent evaluation

** Step 3: Differentiate between ESI 3, 4 and 5 **
- Resource prediction is an integral part of the ESI for patients identified as ESI level 3 (many resources), 4 (one resource), or 5 (no resources).
- Resources include: Labs (blood, urine), ECG, X-rays, CT-MRI-ultrasound, angiography, IV fluids (hydration) , IV, IM or nebulized medications, Specialty consultation, Simple procedure (lac repair, Foley cath), Complex procedure (conscious sedation) 
- Check Danger Zone Vitals: If the patient has very high heart rate, breathing rate, or oxygen saturation below 92%, consider upgrading to ESI Level 2. Use these thresholds:
    - HR: >180 (infants), >160 (toddlers), >140 (children), >100 (older children/adults)
    - RR: >50 (infants), >40 (toddlers), >30 (children), >20 (older children/adults)
    - SaO₂: <92%

** Step 4: If it's not clear by now, determine the appropriate ESI acuity level to the best of your ability. **
     """
        },
        "fewshot": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Use the examples and your knowledge of ESI protocol to determine the appropriate ESI acuity level for the new patient."""
        },
        "fewshotcot": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Let's analyze the examples and solve the problem step by step. 

** Step 1: Assess the Patient's Risk & Resource Needs **
** Step 2: Analyze and Compare with Other Patient Cases **
** Step 3: Determine the ESI acuity level for this patient based on the examples and your assessment of the patient's clinical presentation **"""
        },
        "kate": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Use the examples and your knowledge of ESI protocol to determine the appropriate ESI acuity level for the new patient."""
        },
        "kateautocot": {
            "system": "You are an expert on the Emergency Severity Index (ESI). Think carefully step by step.",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Use the examples and your knowledge of ESI protocol to determine the appropriate ESI acuity level for the new patient."""
        },
        "katecot": {
            "system": "You are an expert on the Emergency Severity Index (ESI).",
            "user": """Your task is to determine the patient's ESI acuity based on the patient's clinical presentation, where ESI 1 indicates the highest priority (requiring immediate, life-saving intervention), and ESI 5 indicates the lowest priority (non-urgent, minimal resources needed). 
Below are some example cases with their ESI acuity levels, followed by a new patient case. Let's analyze the examples and solve the problem step by step. 

** Step 1: Assess the Patient's Risk & Resource Needs **
** Step 2: Analyze and Compare with Similar Patient Cases **
** Step 3: Determine the ESI acuity level for this patient based on the examples and your assessment of the patient's clinical presentation **"""
        }
    },
    "ktas-triage-multi": {
        "vanilla": {
            "system": "You are an expert on the Korean Triage and Acuity Scale (KTAS).",
            "user": "Based on the patients' clinical presentation, which patient is the most acutely ill and should be seen first?"
        },
    }
}


# Function to convert a CSV row into instruction, input, and output strings.
def format_instruction_prompt_for_blackbox(text, strategy='vanilla', dataset='ktas-triage', return_json=False, use_json_schema=False, examples=None):
    """
    Format instruction prompt based on strategy.
    
    Args:
        row: The patient data row
        strategy: The prompting strategy to use ('vanilla', 'cot', 'fewshot', etc.)
        dataset: The dataset being used (default: 'triage-mimic')
        serialization: How to format the patient data ('natural', 'spaces', etc.)
        return_json: Whether to request JSON-formatted output
        examples: Previously labeled examples for few-shot learning (required for some strategies)
    
    Returns:
        dict: Formatted prompt with 'system' and 'user' messages
    
    Raises:
        ValueError: If strategy requires examples but none provided, or if strategy unsupported
    """
    # Validate inputs
    dataset = dataset.lower()
    
    if dataset not in INSTRUCTIONS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if strategy not in INSTRUCTIONS[dataset]:
        raise ValueError(f"Strategy {strategy} not supported for dataset {dataset}")
        
    # Get base prompt template & serialize patient data
    prompt = INSTRUCTIONS[dataset][strategy].copy()  # Create copy to avoid modifying original
    
    # Add JSON formatting instructions if needed
    if return_json and not use_json_schema:
        json_instruction = (
            """ Output your response strictly in JSON format with the following keys:
"Reasoning": <the step by step rationale>,
"Acuity": <the assigned acuity level as an integer between 1 and 5>,
""" if 'cot' in strategy else """ Output your response strictly in JSON format with the following key:
"Acuity": <the assigned acuity level as an integer between 1 and 5>"""
        )
        prompt['system'] += json_instruction
    
    # Handle different strategy types
    # if 'kate' in strategy or 'fewshot' in strategy:
    #     # patient_description = utils_triage.format_row(text, dataset=dataset, serialization=serialization)
    #     if not examples:
    #         raise ValueError(f"Strategy {strategy} requires examples but none were provided")
    #     prompt['user'] = BLACKBOX_LLM_PROMPT_WITH_EXAMPLES.format(
    #         prompt['user'],
    #         examples,
    #         patient_description
    #     )
    # else:
    #     prompt['user'] = BLACKBOX_LLM_PROMPT.format(
    #         prompt['user'],
    #         patient_description
    #     )
    prompt['user'] = BLACKBOX_LLM_PROMPT.format(
        prompt['user'],
        text
    )
    return prompt

# def format_instruction_prompt_for_finetuning(text, EOS_TOKEN, dataset='triage-mimic', split='train'):
#     patient_description = format_row(text, dataset=dataset)
    
#     # Define the instruction for the model.
#     if dataset=='triage-mimic' or dataset=='triage-handbook':
#         instruction = "Based on the clinical presentation, determine the Emergency Severity Index (ESI) acuity for the following patient."
#         output_text = f"The ESI acuity for this patient is {row['acuity']}."
#     elif dataset=='ktas-triage':
#         instruction = "Based on their clinical presentation, determine the KTAS acuity for the following patient."
#         output_text = f"The KTAS acuity for this patient is {row['KTAS_expert']}."
#     else: 
#         raise Exception("Invalid dataset name")
    
#     if split == 'train':
#         text = ALPACA_PROMPT.format(instruction=instruction, input=patient_description, response=output_text) + EOS_TOKEN
#     elif split == 'test':
#         text = ALPACA_PROMPT.format(instruction=instruction, input=patient_description, response='')
#     return {"text": text}
