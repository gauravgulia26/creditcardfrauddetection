from src.logger import logger, TrackedException
from src.config import TEST_DIR, SAVED_MODEL_PATH
import pandas as pd
from src.utils import validated, LoadYaml
from pydantic import Field, ValidationError
import pathlib
import mlflow
import joblib
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Fraud Detection")

yaml_file = LoadYaml(filepath="params.yaml")


@validated
def LoadTestData(
    file_path: pathlib.Path = Field(description="Filepath of Test.csv", default=TEST_DIR),
) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info("Test File Loaded")
    except ValidationError as e:
        err = TrackedException(message=e)
        logger.error(e)
    except Exception as e:
        err = TrackedException(message=e)
        logger.error(e)
    return df


@validated
def LoadModel(
    model_path: pathlib.Path = Field(description="Path to saved model", default=SAVED_MODEL_PATH)
) -> BaseEstimator:
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        err = TrackedException(message=e)
        logger.error(f"Error loading model: {e}")
        raise


@validated
def PrepareTestData(df: pd.DataFrame = Field(description="Test DataFrame")) -> pd.DataFrame:
    try:
        df = df.drop(columns=yaml_file.training.drop)
        X = df.drop(columns=yaml_file.training.target, errors="ignore")
        logger.info("Test data prepared successfully")
        return X
    except Exception as e:
        err = TrackedException(message=e)
        logger.error(f"Error preparing test data: {e}")
        raise


def main():
    with mlflow.start_run(
        run_name="Model Prediction",
        description="This MLflow run is used to track model predictions on test data",
    ):
        # Load test data
        test_df = LoadTestData()

        # Load the model
        model = LoadModel()

        # Prepare test data
        X_test = PrepareTestData(df=test_df)

        # Log test dataset metrics
        test_metrics = {"test_rows": X_test.shape[0], "test_columns": X_test.shape[1]}
        mlflow.log_dict(test_metrics, artifact_file=yaml_file.mlflow.model_metrics_file)

        # Make predictions
        y_pred = model.predict(X_test)

        # If target column exists in test data, calculate and log performance metrics
        if yaml_file.training.target in test_df.columns:
            y_test = test_df[yaml_file.training.target]
            metrics = {
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred),
                "test_recall": recall_score(y_test, y_pred),
                "test_f1_score": f1_score(y_test, y_pred),
            }
            mlflow.log_metrics(metrics)
            logger.info("Performance metrics calculated and logged")

        # Add predictions to the test dataframe
        test_df["predictions"] = y_pred

        # Save predictions
        output_path = f"{pathlib.Path(TEST_DIR).parent}/predictions.csv"
        test_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

        logger.info("Prediction process completed successfully")
        return test_df


if __name__ == "__main__":
    main()
