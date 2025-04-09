import numpy as np
import utils.utils as utils
import utils.prompts as prompts
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from utils.api import query_llm
import utils.utils_triage as utils_triage
from utils.dataset import DATASETS
from pydantic import BaseModel
from tooldantic import ToolBaseModel, OpenAiResponseFormatGenerator

class CustomSchemaGenerator(OpenAiResponseFormatGenerator):
    is_inlined_refs = True
    
class BaseModel(ToolBaseModel):
    _schema_generator = CustomSchemaGenerator
       
class BasePredictor():
    """
    Base predictor class that defines the interface for all prediction strategies.
    """
    def predict(self, *, model_config, return_json=True, debug=False):
        """
        Base prediction method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the predict method")

class TriagePredictor(BasePredictor):
    """
    The predictor class is responsible for:
    1) Routing the data to the correct model/strategy
    2) Logic for each prediction strategy
    """
    class Acuity(BaseModel):
        acuity: int
    
    class AcuityWithReasoning(BaseModel):
        reason: str
        acuity: int
       
    def __init__(self, dataset, strategy, training_set=None):
        """
        :param model: The LLM to use for predictions.
        :param strategy: Prediction strategy ("FewShot", "CoT", 'vanilla', "SelfConsistency").
        :param debug: Whether to enable debug mode.
        :param training_set: The training dataset for few-shot prompting.
        """
        self.dataset = dataset
        self.strategy = strategy
        self.target = DATASETS[self.dataset]['target_column']
        self.hippa = DATASETS[self.dataset]['is_hippa']
        self.training_set = training_set
        
        if training_set is not None:
            self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embeddings for training set data."""
        # Set up the encoder model
        model_name = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
        self.symptom_encoder = SentenceTransformer(model_name)
        
        # Handle dataset-specific embedding loading
        if self.dataset.lower() == 'triage-ktas':
            self._load_ktas_embeddings()
        else:
            self._load_standard_embeddings()
    
    def _load_ktas_embeddings(self):
        """Load embeddings specific to the KTAS dataset."""
        complaint_embeddings = np.load(
            DATASETS[self.dataset]['training_complaint_embeddings_filepath'], 
            allow_pickle=True
        )
        diagnosis_embeddings = np.load(
            DATASETS[self.dataset]['training_diagnosis_embeddings_filepath'], 
            allow_pickle=True
        )
        
        self.complaint_embeddings = complaint_embeddings
        self.diagnosis_embeddings = diagnosis_embeddings
        self.embeddings_cache = np.concatenate([complaint_embeddings, diagnosis_embeddings], axis=1)
    
    def _load_standard_embeddings(self):
        """Load embeddings for standard datasets."""
        embeddings_path = DATASETS[self.dataset]['training_embeddings_filepath']
        self.embeddings_cache = np.load(embeddings_path, allow_pickle=True)
    
    def predict(self, *, text, model_config, k_shots, return_json=True, use_json_schema=True, debug=False):
        """
        Route prediction based on strategy.
        """
        if self.strategy in ["fewshot", "fewshotcot"]:
            prompt, response = self.few_shot_prediction(text=text, model_config=model_config, k_shots=k_shots, return_json=return_json, use_json_schema=use_json_schema, debug=debug)
        elif self.strategy in ["kate", "katecot", "kateautocot"]:
            prompt, response = self.kate_prediction(text=text, model_config=model_config, k_shots=k_shots, return_json=return_json, use_json_schema=use_json_schema, debug=debug)
        elif self.strategy in ['vanilla', "vanillav0", "autocot", "cot","demonstrationcot"]:
            prompt, response = self.zero_shot_prediction(text=text, model_config=model_config, return_json=return_json, use_json_schema=use_json_schema, debug=debug)
        elif self.strategy in ["selfconsistency"]:
            prompt, response = self.self_consistency_prediction(text=text, model_config=model_config, return_json=return_json, use_json_schema=use_json_schema, debug=debug)
        elif self.strategy in ["multiagent"]:
            prompt, response = self.multi_agent_prediction(text=text, model_config=model_config, return_json=return_json, use_json_schema=use_json_schema, debug=debug)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
            
        if return_json:
            try:
                response_data = json.loads(response)
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
                print("Raw response causing error:", response)
                response = query_llm(response + "\n\nCan you format the above in proper JSON", model='gpt-4o-mini', return_json=True)
                response_data = json.loads(response)
            return prompt, response_data
        else:
            response = {
                'acuity': utils_triage.extract_acuity_from_text(response, debug),
                'reasoning': response
            }
            return prompt, response

    def zero_shot_prediction(self, *, text, model_config, return_json=False, use_json_schema=False, debug=False, **kwargs):
        """Zero-shot prediction implementation."""

        prompt = prompts.format_instruction_prompt_for_blackbox(
            text=text,
            strategy=self.strategy,
            dataset=self.dataset.lower(),
            return_json=return_json,
            use_json_schema=use_json_schema
        )
        
        if use_json_schema:
            json_schema = self.AcuityWithReasoning.model_json_schema() if 'CoT' in self.strategy else self.Acuity.model_json_schema()
            response = query_llm(prompt, model=model_config['name'], return_json=return_json, debug=debug, json_schema=json_schema, is_hippa=self.hippa)
        else:
            response = query_llm(prompt, model=model_config['name'], return_json=return_json, debug=debug, is_hippa=self.hippa)
        
        return prompt, response

    def few_shot_prediction(self, *, text, model_config, k_shots=5, return_json=False, use_json_schema=False, debug=False, **kwargs):
        """Few-shot prediction implementation."""
        if self.training_set is None:
            raise ValueError("Few-shot strategy requires a training set of examples.")
        
        _, examples = utils.get_stratified_df(self.training_set, target_col=self.target, test_size=k_shots)
        
        formatted_examples = "\n\n".join([
            prompts.format_row(example, dataset=self.dataset.lower()) + 
            f"\nAcuity Level: {example[self.target]}"
            for _, example in examples.iterrows()
        ])
        
        prompt = prompts.format_instruction_prompt_for_blackbox(
            text=text,
            strategy=self.strategy,
            dataset=self.dataset.lower(),
            return_json=return_json,
            examples=formatted_examples
        )
        
        if use_json_schema:
            json_schema = self.AcuityWithReasoning.schema() if 'CoT' in self.strategy else self.Acuity.schema()
            response = query_llm(prompt, model=model_config, return_json=return_json, debug=debug, json_schema=json_schema, is_hippa=self.hippa)
        else:
            response = query_llm(prompt, model=model_config['name'], return_json=return_json, debug=debug, is_hippa=self.hippa)
        
        return prompt, response

    def self_consistency_prediction(self, *, text, model_config, return_json=False, use_json_schema=False, debug=False, num_trials=5, **kwargs):
        """Self-consistency prediction implementation."""
        responses = []
        
        for _ in range(num_trials):
            prompt = prompts.format_instruction_prompt_for_blackbox(
                text=text,
                strategy='SelfConsistency',
                dataset=self.dataset.lower(),
                return_json=True
            )
            
            if use_json_schema:
                json_schema = self.Acuity.schema()
                response = query_llm(prompt, temperature=1.0, model=model_config['name'], return_json=True, debug=debug, json_schema=json_schema, is_hippa=self.hippa)
            else:
                response = query_llm(prompt, temperature=1.0, model=model_config['name'], return_json=True, debug=debug, is_hippa=self.hippa)
            
            try:
                response_data = json.loads(response)
                responses.append(response_data['Acuity'])
            except (json.JSONDecodeError, KeyError) as e:
                if debug:
                    print(f"Error parsing response: {e}")
                    print(f"Raw response: {response}")
                continue
        
        if not responses:
            raise ValueError("No valid responses received from any trial")
        
        from statistics import mode
        final_acuity = mode(responses)
        
        if return_json:
            final_response = json.dumps({
                "Acuity": final_acuity,
                "Reasoning": f"Based on {len(responses)} trials, the most common prediction was {final_acuity}. All predictions: {responses}"
            })
        else:
            final_response = f"Based on {len(responses)} trials, the most common prediction was {final_acuity}. All predictions: {responses}"
        
        return prompt, final_response

    def kate_prediction(self, *, text, model_config, k_shots=5, return_json=False, use_json_schema=False, debug=False, **kwargs):
        """KATE prediction implementation."""
        if self.training_set is None:
            raise ValueError("KATE strategy requires a training set of examples.")
        
        examples = self._retrieve_top_k_examples(text, k_shots)
        
        formatted_examples = "\n\n".join([
            prompts.format_row(example, dataset=self.dataset.lower()) + 
            f"\nAcuity Level: {example[self.target]}"
            for _, example in examples.iterrows()
        ])
        
        prompt = prompts.format_instruction_prompt_for_blackbox(
            text=text,
            strategy=self.strategy,
            dataset=self.dataset.lower(),
            return_json=return_json,
            examples=formatted_examples
        )
        
        if use_json_schema:
            json_schema = self.AcuityWithReasoning if 'CoT' in self.strategy else self.Acuity
            response = query_llm(prompt, model=model_config['name'], return_json=return_json, debug=debug, json_schema=json_schema, is_hippa=self.hippa)
        else:
            response = query_llm(prompt, model=model_config['name'], return_json=return_json, debug=debug, is_hippa=self.hippa)
        
        return prompt, response
    
    def get_top_k_similar(self, embedding, embeddings, k):
        """
        Find the top-k most similar samples to a given embedding.
        """
        similarities = np.dot(embeddings, embedding)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices, similarities[top_k_indices]

    def _retrieve_top_k_examples(self, text, row, k):
        """
        Retrieve the top K most similar examples to the prompt based on Ada embeddings.
        First searches by concatenated chief complaint and diagnosis embeddings, then refines by vital signs.

        :param row: The input row containing chief complaint and vital signs.
        :param k: Number of examples to retrieve.
        :return: DataFrame of top K examples.
        """
        if self.dataset.lower() == 'triage-ktas':
            # Handle cases where either symptom or diagnosis is empty/null
            symptom = row['Chief_complain']
            diagnosis = row['Diagnosis in ED']
            
            if pd.isna(symptom) or symptom == '':
                # Only use diagnosis embedding if symptom is empty
                combined_embedding = self.symptom_encoder.encode(diagnosis)
                top_k_indices, _ = self.get_top_k_similar(combined_embedding, self.diagnosis_embeddings, k*3)
            elif pd.isna(diagnosis) or diagnosis == '':
                # Only use symptom embedding if diagnosis is empty  
                combined_embedding = self.symptom_encoder.encode(symptom)
                top_k_indices, _ = self.get_top_k_similar(combined_embedding, self.complaint_embeddings, k*3)
            else:
                # Combine both embeddings if neither is empty
                symptom_embedding = self.symptom_encoder.encode(symptom)
                diagnosis_embedding = self.symptom_encoder.encode(diagnosis)
                combined_embedding = np.concatenate([symptom_embedding, diagnosis_embedding])
                # Search using embedding
                top_k_indices, _ = self.get_top_k_similar(combined_embedding, self.embeddings_cache, k*3)
            
        elif self.dataset.lower() == 'triage-handbook':
            symptom = row['Clinical Vignettes']
            embeddings = self.symptom_encoder.encode(symptom)
            top_k_indices, _ = self.get_top_k_similar(embeddings, self.embeddings_cache, k*3)
            top_k_samples = self.training_set.loc[top_k_indices]
            # print(top_k_samples)
            return top_k_samples
        
        # Then, narrow down to k by vitals
        if self.dataset.lower() == 'triage-mimic':
            vital_sign_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
        elif self.dataset.lower() == 'triage-ktas':
            vital_sign_columns = ['NRS_pain', 'SBP', 'DBP', 'HR', 'RR', 'BT']
        
        numeric_vital_signs = [col for col in vital_sign_columns if pd.api.types.is_numeric_dtype(row[col])]

        query_vital_signs = row[numeric_vital_signs].values.astype(float)
        top_25_vital_signs = self.training_set.loc[top_k_indices, numeric_vital_signs]
        top_25_vital_signs = top_25_vital_signs.apply(pd.to_numeric, errors='coerce')  # Coerce invalid values to NaN
        top_25_vital_signs = top_25_vital_signs.dropna()  # Drop rows with NaN values
        
        # Ensure the resulting values are numpy floats
        top_25_vital_signs = top_25_vital_signs.values.astype(float)
        query_vital_signs_normalized = query_vital_signs / np.linalg.norm(query_vital_signs)
        top_25_vital_signs_normalized = top_25_vital_signs / np.linalg.norm(top_25_vital_signs, axis=1, keepdims=True)
        
        vital_signs_similarity = np.dot(top_25_vital_signs_normalized, query_vital_signs_normalized)
        top_k_vital_indices = np.argsort(vital_signs_similarity)[-k:][::-1]
        
        # Retrieve top-k samples
        top_k_vital_samples = self.training_set.loc[top_k_indices[top_k_vital_indices]]
        
        # # Add the 'pain' column back to the result (unchanged)
        # top_k_vital_samples['pain'] = self.training_set.loc[top_k_indices[top_k_vital_indices], 'pain']
        
        return top_k_vital_samples
    
    
class MultiTriagePredictor(BasePredictor):
    """
    The predictor class is responsible for:
    1) Routing the data to the correct model/strategy
    2) Logic for each prediction strategy
    """
       
    def __init__(self, dataset, strategy):
        """
        :param model: The LLM to use for predictions.
        :param strategy: Prediction strategy ("FewShot", "CoT", 'vanilla', "SelfConsistency").
        :param debug: Whether to enable debug mode.
        :param training_set: The training dataset for few-shot prompting.
        """
        self.dataset = dataset
        self.strategy = strategy
        self.target = DATASETS[self.dataset]['target_column']
        self.hippa = DATASETS[self.dataset]['is_hippa']


    def predict(self, *, text, model_config, k_shots, return_json=True, use_json_schema=True, debug=False):
        """
        Route prediction based on strategy.
        """
        if self.strategy == 'vanilla':
            prompt, response = self.zero_shot_prediction(text=text, model_config=model_config, return_json=return_json, use_json_schema=use_json_schema, debug=debug)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
            
        if return_json:
            try:
                response_data = json.loads(response)
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
                print("Raw response causing error:", response)
                response = query_llm(response + "\n\nCan you format the above in proper JSON", model='gpt-4o-mini', return_json=True)
                response_data = json.loads(response)
            return prompt, response_data
        else:
            response = {
                self.target: utils_triage.extract_patient_from_text(response, debug),
                'reasoning': response
            }
            return prompt, response
        
    def zero_shot_prediction(self, *, text, model_config, return_json=False, use_json_schema=False, debug=False, **kwargs):
        """Zero-shot prediction implementation."""

        prompt = prompts.format_instruction_prompt_for_blackbox(
            text=text,
            strategy=self.strategy,
            dataset=self.dataset.lower(),
            return_json=return_json,
            use_json_schema=use_json_schema
        )

        response = query_llm(prompt, model=model_config['name'], return_json=return_json, debug=debug, is_hippa=self.hippa)
        
        return prompt, response