from logpunch.customlogger import CustomLogger, LoggerConfig
from logpunch.customexception import TrackedException

cfg = LoggerConfig(log_dir='logs')

logger = CustomLogger(config=cfg).get_logger()