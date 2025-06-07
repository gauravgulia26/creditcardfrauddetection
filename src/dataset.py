from src.logger import logger, TrackedException
from src.config import TRAIN_DIR, TEST_DIR
from src.utils import LoadDataKaggle, LoadYaml, validated
from pydantic import validate_call, Field
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Fraud Detection")


try:
    df = LoadDataKaggle(file_path="params.yaml")
except Exception as e:
    err = TrackedException(message=e)
    logger.error(err)

try:
    yaml_file = LoadYaml(filepath="params.yaml")
except Exception as e:
    err = TrackedException(message=e)
    logger.error(e)

METRICS_FILE = yaml_file.mlflow.metrics_file


@validated
def SplitData(
    df: pd.DataFrame = Field(description="Original Pandas Dataframe"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df, test_size=yaml_file.dataset.split_size, random_state=yaml_file.dataset.random_state
    )
    return train, test


@validated
def SaveData(train: pd.DataFrame, test: pd.DataFrame) -> str:
    try:
        train.to_csv(TRAIN_DIR, index=False)
        test.to_csv(TEST_DIR, index=False)
    except Exception as e:
        err = TrackedException(message=e)
        logger.error(e)
    logger.info("Training and Testing set Created and Saved Successfully")
    return "Make Dataset Complete"


def main():
    with mlflow.start_run(
        run_name="Dataset Creation",
        log_system_metrics=True,
        description="This Mlflow Run is used to Track the Dataset Creation Stage",
    ):
        mlflow.log_dict(yaml_file, artifact_file=METRICS_FILE)
        train, test = SplitData(df=df)
        creation_metrics = {
            "train_rows": train.shape[0],
            "train_columns": train.shape[1],
            "test_rows": test.shape[0],
            "test_columns": test.shape[1],
        }
        mlflow.log_metrics(creation_metrics)
        SaveData(train=train, test=test)


if __name__ == "__main__":
    main()
