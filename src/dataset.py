from src.logger import logger, TrackedException
from src.config import TRAIN_DIR, TEST_DIR
from src.utils import LoadDataKaggle, LoadYaml, validated
from pydantic import validate_call, Field
import pandas as pd
from sklearn.model_selection import train_test_split

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
        train.to_csv(TRAIN_DIR,index=False)
        test.to_csv(TEST_DIR,index=False)
    except Exception as e:
        err = TrackedException(message=e)
        logger.error(e)
    logger.info("Training and Testing set Created and Saved Successfully")
    return "Make Dataset Complete"
