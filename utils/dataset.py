"""We store unanimously required & unique characteristics of the datasets here."""
DATASETS = {
    "ktas-triage": {
        "test_path": "./data/ktas-triage/test_text.csv",
        "train_path": "./data/ktas-triage/train.csv",
        "embeddings": {
            'complaint':{
                "train": "./data/ktas-triage/KTAS_train_chiefcomplaint_embeddings.npy",
                "test": "./data/ktas-triage/KTAS_test_chiefcomplaint_embeddings.npy",
            },
            'diagnosis':{
                "train": "./data/ktas-triage/KTAS_train_diagnosis_embeddings.npy",
                "test": "./data/ktas-triage/KTAS_test_diagnosis_embeddings.npy",
            }
        },
        "format": "csv",
        "x_column": "patient_case",
        "target_column": "acuity",
        "is_hippa": True,
    },
    "ktas-triage-multi": {
        "test_path": "./data/ktas-triage-multi/formatted_patient_cases.csv",
        "format": "csv",
        "x_column": "patient_case",
        "target_column": "patient_index",
        "is_hippa": True,
    },
    "esi-handbook": {
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
             