from telco_churn.config import settings
from imblearn.pipeline import Pipeline
from telco_churn.features.preprocessing import build_preprocessor, build_smote, recode_transformer


def build_pipeline(classifier):
    pipeline = Pipeline(
        [
            ("recode", recode_transformer),
            ("smote", build_smote()),
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
    return pipeline