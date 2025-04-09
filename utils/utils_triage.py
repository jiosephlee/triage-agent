import pandas as pd
TEMPLATES = {
"spaces": "age   sex   temperature   heartrate   resprate   sbp   dbp   pain   chiefcomplaint   diagnosis in ED   injury   arrival   mental\n{age} years   {sex}   {temperature}°F   {heartrate} bpm   {resprate} breaths/min   {sbp} mmHg   {dbp} mmHg   {pain}   {chiefcomplaint}   {suspicion}   {injury}   {arrival}   {mental}",
"commas": "age, sex, temperature, heartrate, resprate, sbp, dbp, pain, chiefcomplaint, diagnosis in ED, injury, arrival, mental\n{age} years, {sex}, {temperature}°F, {heartrate} bpm, {resprate} breaths/min, {sbp} mmHg, {dbp} mmHg, {pain}, {chiefcomplaint}, {suspicion}, {injury}, {arrival}, {mental}",
"newline": "age: {age} years\nsex: {sex}\ntemperature: {temperature}°F\nheartrate: {heartrate} bpm\nresprate: {resprate} breaths/min\nsbp: {sbp} mmHg\ndbp: {dbp} mmHg\npain: {pain}\nchiefcomplaint: {chiefcomplaint}\ndiagnosis in ED: {suspicion}\ninjury: {injury}\narrival mode: {arrival}\nmental: {mental}",
}
import utils.utils as utils
import utils.api as api
import time

def convert_arrival(arrival_transport):
    mapping = {
        "WALKING": " arriving on foot",
        "119 AMBULANCE": " arriving by public ambulance",
        "PRIVATE VEHICLE": " arriving by private vehicle",
        "PRIVATE AMBULANCE": " arriving by private ambulance",
        'PUBLIC TRANSPORTATION': "arriving by public transportation",
        'WHEELCHAIR': 'came on a wheelchair'
    }
    return mapping.get(arrival_transport.upper(), "")

def format_row(row, serialization='natural'):
    """ Create a natural language description of the patient."""
    if serialization in TEMPLATES:
        template = TEMPLATES[serialization]
        return template.format(
            temperature=row.get('BT', 'N/A'),
            heartrate=row.get('HR', 'N/A'),
            resprate=row.get('RR', 'N/A'),
            sbp=row.get('SBP', 'N/A'),
            dbp=row.get('DBP', 'N/A'),
            pain=row.get('NRS_pain', 'N/A'),
            chiefcomplaint=row.get('Chief_complain', 'N/A'),
            suspicion=row.get('Diagnosis in ED', 'N/A'),
            injury=row.get('Injury', 'N/A'),
            arrival=row.get('Arrival mode', 'N/A'),
            mental=row.get('Mental', 'N/A'),
            age=row.get('Age', 'N/A'),
            sex=row.get('Sex', 'N/A'),            
            )
    # --- Triage-ktas version ---
    age = f"{int(row['Age'])}-year-old " if pd.notna(row.get("Age")) else ""
    
    # Map the Sex column: assuming 1 = Female, 2 = Male.
    sex = row.get("Sex")
    if pd.isna(sex):
        gender_str = "person"
        pronoun = "They have"
    else:
        if sex == 'Female':
            gender_str = "woman"
            pronoun = "She has"
        elif sex == 'Male':
            gender_str = "man"
            pronoun = "He has"
        else:
            gender_str = "person"
            pronoun = "They have"
    
    arrival_text = convert_arrival(row.get("Arrival mode"))
    
    chief_text = f" with a chief complaint of '{row['Chief_complain']}'" if pd.notna(row.get("Chief_complain")) else ""
    
    # Include information about injury if applicable.
    injury = row.get("Injury")
    injury_text = ""
    if pd.notna(injury):
        if injury == 'Yes':
            injury_text = " who sustained an injury"
    
    # Prepare the vital signs.
    vitals = {}
    vitals["temperature"] = f" temperature of {row['BT']}°C" if pd.notna(row.get("BT")) else ""
    vitals["heartrate"] = f", heart rate of {row['HR']} bpm" if pd.notna(row.get("HR")) else ""
    vitals["resprate"] = f", respiratory rate of {row['RR']} breaths/min" if pd.notna(row.get("RR")) else ""
    vitals["sbp"] = f", systolic blood pressure of {row['SBP']} mmHg" if pd.notna(row.get("SBP")) else ""
    vitals["dbp"] = f", diastolic blood pressure of {row['DBP']} mmHg" if pd.notna(row.get("DBP")) else ""
    
    # Handle pain:
    # 'Pain' is a flag (0 or 1) indicating whether the patient feels pain.
    # 'NRS_pain' provides the actual pain level (and may be NA).
    # Since 'Pain' is never null, we can safely convert it to an integer.
    pain_flag = int(row["Pain"])
    if pain_flag == 1:
        if pd.notna(row.get("NRS_pain")):
            vitals["pain"] = f", and reports pain with a level of {row['NRS_pain']}."
        else:
            vitals["pain"] = ", and reports pain but no level was provided"
    else:
        vitals["pain"] = ""
    # Mental status, if available.
    if pd.notna(row.get("Mental")):
        vitals["mental"] = f" He is mentally {row['Mental']}"
    else:
        vitals["mental"] = ""
    
    missing_vitals = [key for key, value in vitals.items() if value == ""]
    
    description = (
        f"A {age}{gender_str}{injury_text} arrives at the emergency department"
        f"{arrival_text}{chief_text}. "
        f"{pronoun}{''.join(vitals.values())}."
        f" The patient is suspected to have {row['Diagnosis in ED']}."
    )
    if missing_vitals:
        missing_str = ", ".join(missing_vitals).replace("_", " ")
        description += f" Data on {missing_str} is missing."
    
    return description

def extract_acuity_from_text(text, debug):
    """Extract the estimated acuity from the text using a LLM."""
    lines = text.splitlines()
    last_five_lines = lines[-5:]
    text = "\n".join(last_five_lines)
    answer_text = api.query_llm(f"Extract the estimated acuity from the following response. If the estimate is uncertain, just choose one that is best. Output the number alone.\n\n\"\"\"{text}.\"\"\"", 
                                       model= "gpt-4o-mini", system_prompt_included=False, debug=debug)
    num = utils.extract_num(answer_text)
    time.sleep(1)
    if type(num) == str and 'Error' in num:
        return utils.extract_num(api.query_llm(f"Extract the estimated acuity from the following information and output the number alone. If the estimate is uncertain, just choose one that is best.\n\n\"\"\"{text}.\"\"\"", 
                                       model= "gpt-4o-mini", system_prompt_included=False, debug=debug))
    else:
        return num
    
    
def extract_patient_from_text(text, debug):
    """Extract the index of the patient that was chosen from the text using a LLM."""
    answer_text = api.query_llm(f"Extract the index (starting from 0) of the patient that was chosen from the following response. Output the number alone.\n\n\"\"\"{text}.\"\"\"", 
                                       model= "gpt-4o-mini", system_prompt_included=False, debug=debug)
    num = utils.extract_num(answer_text)
    time.sleep(1)
    if type(num) == str and 'Error' in num:
        return utils.extract_num(api.query_llm(f"Extract the index (starting from 0) of the patient that was chosen from the following information and output the number alone.\n\n\"\"\"{text}.\"\"\"", 
                                       model= "gpt-4o-mini", debug=debug))
    else:
        return num