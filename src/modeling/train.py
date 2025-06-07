import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import pandas as pd
from src.logger import logger
from src.utils import validated, LoadYaml
from src.config import TRAIN_DIR, SAVED_MODEL_PATH
from pydantic import Field
import pathlib
import mlflow
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

        # Log dataset metrics
        training_metrics = {
            "train_rows": X.shape[0],
            "train_columns": X.shape[1],
            "test_Series_rows": Y.shape[0],
        }
        mlflow.log_dict(training_metrics, artifact_file=yaml_file.mlflow.model_metrics_file)

        # Log model parameters
        mlflow.log_params(yaml_file.random_forest)

        # Train the model
        model = TrainModel(x=X, y=Y)

        # Calculate and log model performance metrics
        y_pred = model.predict(X)

        metrics = {
            "accuracy": accuracy_score(Y, y_pred),
            "precision": precision_score(Y, y_pred),
            "recall": recall_score(Y, y_pred),
            "f1_score": f1_score(Y, y_pred),
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Create input example for model signature
        input_example = X.head(1)

        # Log the model with signature
        mlflow.sklearn.log_model(
            model,
            "random_forest_model",
            registered_model_name="fraud_detection_model",
            input_example=input_example,
        )

        # Save model using joblib
        os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)
        joblib.dump(model, SAVED_MODEL_PATH)
        logger.info(f"Model saved successfully at {SAVED_MODEL_PATH}")

        logger.info("Model training and logging completed successfully")

    return model


if __name__ == "__main__":
    main()
