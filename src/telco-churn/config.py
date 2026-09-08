from pathlib import Path
from pydantic_settings import BaseSettings , SettingsConfigDict

class Settings(BaseSettings):
    # model_path: Path
    # data_raw_path: Path
    # data_processed_path: Path
    # test_size: float
    # random_state: int
    # cols_to_drop: list[str]
    # ohe_cols: list[str]
    # binary_cols: list[str]
    # numeric_cols: list[str]
    # service_cols_to_recode: list[str]
    # mlflow_tracking_uri: str



    # model_path: Path = Path(__file__).parent.parent / "model" / "model.pkl"
    # data_path: Path = Path(__file__).parent.parent / "data" / "processed" / "cleaned_data.csv"
    test_size: float = 0.2
    random_state: int = 42

    cols_to_drop: list[str] = ["TotalCharges", "MonthlyCharges",
                                "gender", "PhoneService", "MultipleLines", "customerID"]
    ohe_cols: list[str] = ["InternetService", "Contract", "PaymentMethod"]
    binary_cols: list[str] = [
                    "Partner", "Dependents", "PaperlessBilling",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies",
                ]
    numeric_cols: list[str] = ["tenure"]
    service_cols_to_recode: list[str] = [ "OnlineSecurity", "OnlineBackup"
                                         , "DeviceProtection", "TechSupport", 
                                         "StreamingTV", "StreamingMovies",]

    data_raw_path: Path = Path(__file__).parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data_processed_path: Path = Path(__file__).parent.parent / "data" / "processed" / "cleaned_data.csv"
    mlflow_tracking_uri: str = "http://localhost:5000"  
    mlflow_experiment_name: str = "telco-churn"
    

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

Settings = Settings() #why is this needed?