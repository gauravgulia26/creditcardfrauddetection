from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import pandas as pd
from src.logger import logger
from src.utils import validated, LoadYaml
from src.config import TRAIN_DIR
from pydantic import Field
import pathlib
import mlflow
import joblib
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Fraud Detection")

yaml_file = LoadYaml(filepath="params.yaml")


@validated
def ReadData(
    train_file_path: pathlib.Path = Field(
        description="File Path of the Traning File", default=TRAIN_DIR
    )
):
    df = pd.read_csv(train_file_path)
    logger.info("Dataframe Loaded Successfully")
    return df


@validated
def MakeDepIndep(
    df: pd.DataFrame = Field(description="Training Set DF"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate X and Y from Original Dataframe
    Args:
        df (pd.DataFrame, optional): _description_. Defaults to Field(description='Training Set DF').
    """

    df = df.drop(columns=yaml_file.training.drop)
    X = df.drop(columns=yaml_file.training.target)
    y = df[yaml_file.training.target]
    logger.info("Indepedent and Dependent Features Created Successfully")
    return X, y


@validated
def TrainModel(
    x: pd.DataFrame = Field(description="Independent Features Df"),
    y: pd.Series = Field(description="Dependent Features Series"),
) -> BaseEstimator:
    """
    Train a Sklearn Model on a given Data
    Args:
        x (pd.DataFrame, optional): _description_. Defaults to Field(description='Independent Features Df').
        y (pd.DataFrame, optional): _description_. Defaults to Field(description='Dependent Features Df').

    Returns:
        BaseEstimator: Trained Sklearn Model for Prediction
    """
    params = yaml_file.random_forest
    classifier = RandomForestClassifier(**params)
    logger.info("Random Forest Training Started")
    model = classifier.fit(x, y)
    return model


def main():
    with mlflow.start_run(
        run_name="Training Model",
        description="This Mlflow run is used to track the Model Training Stage",
    ):
        df = ReadData()
        X, Y = MakeDepIndep(df=df)
        training_metrics = {
            "train_rows": X.shape[0],
            "train_columns": X.shape[1],
            "test_Series_rows": Y.shape[0],
        }
        mlflow.log_dict(training_metrics, artifact_file=yaml_file.mlflow.model_metrics_file)
        mlflow.log_params(yaml_file.random_forest, synchronous=False)
        model = TrainModel(x=X, y=Y)
    return model


if __name__ == "__main__":
    main()
