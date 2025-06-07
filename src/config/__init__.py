from src.constants import PROJECT_ROOT
from src.utils import LoadYaml
from src.logger import logger, TrackedException

try:
    yaml_file = LoadYaml(filepath="params.yaml")
except Exception as e:
    err = TrackedException(message=e)
    logger.error(e)

DATA_DIR = yaml_file.dataset.data_dir
TRAIN_FILE = yaml_file.dataset.train_file_name
TEST_FILE = yaml_file.dataset.test_file_name

TRAIN_DIR = f"{PROJECT_ROOT.as_posix()}/{DATA_DIR}/processed/{TRAIN_FILE}"
TEST_DIR = f"{PROJECT_ROOT.as_posix()}/{DATA_DIR}/processed/{TEST_FILE}"

MODEL_DIR = "models"
MODEL_NAME = "best_model.joblib"
SAVED_MODEL_PATH = f"{PROJECT_ROOT.as_posix()}/{MODEL_DIR}/{MODEL_NAME}"
