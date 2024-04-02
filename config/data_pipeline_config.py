from ml_collections import config_dict
from .data_simulation_config import get_simulation_config

simulation_config = get_simulation_config()

def _commom_config():
    cfg = config_dict.ConfigDict()
    cfg.IMAGE_SIZE = 128
    cfg.INPUT_SHAPE = (simulation_config.IMAGE_HEIGHT, 
                       simulation_config.IMAGE_WIDTH, 
                       simulation_config.CHANNELS_IMG)
    return cfg

def get_train_val_pipeline_config():
    cfg = _commom_config()
    cfg.COMPRESSION = simulation_config.COMPRESSION
    cfg.CROP_PADDING = 32
    cfg.YAML_FILE_PATH = '/home/anirudhan/project/image-fusion/config/dataset.yml'
    return cfg

def get_test_pipeline_config():
    cfg = _commom_config()
    # this is the mean and std of real Real Mff dataset
    cfg.MEAN = [81.594,  89.681, 78.611]
    cfg.STD = [71.661, 71.096, 86.162]
    cfg.FOLDER = '/home/anirudhan/project/image-fusion/data/RealMFF'
    return cfg