from config import Settings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTENC

def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), Settings.ohe_cols),
            ("binary", OrdinalEncoder(categories=[["No", "Yes"]]*len(Settings.binary_cols) ), Settings.binary_cols),
            ("scaler" , StandardScaler(), Settings.numeric_cols),
        ],
        remainder="passthrough"
        
    )

    return preprocessor




def build_smote():
    categorical_features_for_smote = Settings.ohe_cols + Settings.binary_cols

    smote = SMOTENC(categorical_features=categorical_features_for_smote, random_state=Settings.random_state)
    return smote


def recode_no_service(X):

    X = X.copy()
    for col in Settings.service_cols_to_recode:
        X[col] = X[col].replace("No internet service", "No")
    return X

recode_transformer = FunctionTransformer(
    recode_no_service, validate=False, feature_names_out="one-to-one"
)