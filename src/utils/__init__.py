# Utilities Functions for Reusability

import kagglehub
from kagglehub import KaggleDatasetAdapter
from pydantic import Field, validate_call, ValidationError
import pathlib
import yaml
from box import ConfigBox
from src.logger import logger, TrackedException
import pandas as pd
from rich import print
from functools import wraps


def validated(func):
    return validate_call(config=dict(arbitrary_types_allowed=True))(func)


@validated
def LoadYaml(
    filepath: pathlib.Path = Field(description="Pathlib Path of 'params.yaml' file"),
) -> ConfigBox:
    """
    Utility Function to Load Yaml File as ConfigBox
    Args:
        filepath (pathlib.Path): "Pathlib Path of 'params.yaml' file"

    Returns:
        ConfigBox: A ConfigBox object of Yaml File
    """
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    return ConfigBox(config)


@validated
def LoadDataKaggle(
    file_path: pathlib.Path = Field(description="Params.yaml File Path", default=None),
) -> pd.DataFrame:
    """
    Utility Function to Fetch Data from Kaggle
    Args:
        file_path (pathlib.Path, optional): _description_. Defaults to Field( description="Params.yaml File Path", default=None ).

    Returns:
        pd.DataFrame: Dataframe of Dataset
    """
    try:
        yaml = LoadYaml(filepath=file_path)
    except Exception as e:
        err = TrackedException(message=e)
        logger.error(e)
    dataset_name = yaml.kaggle.dataset_name
    kaggle_path = yaml.kaggle.file_path
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, dataset_name, kaggle_path)

    return df
