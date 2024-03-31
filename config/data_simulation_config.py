from ml_collections import config_dict

def get_simulation_config():
    cfg = config_dict.ConfigDict()

    cfg.BATCH_SIZE = 200
    cfg.NUM_EPOCHS = 100
    cfg.IMAGE_HEIGHT = 512
    cfg.IMAGE_WIDTH = 512
    cfg.CHANNELS_IMG = 3
    cfg.NUM_WORKERS = 1
    cfg.PREFETCH_FACTOR = 16
    cfg.PIN_MEMORY = True
    cfg.LOAD_MODEL = True
    cfg.TRAIN_IMG_DIR = '/home/anirudhan/project/image-fusion/data/AM_2k/AM-2k'
    cfg.VAL_IMG_DIR = '/home/anirudhan/project/image-fusion/data/AM_2k/valSim'
    cfg.TF_RECORD_DIR = f'/home/anirudhan/project/image-fusion/data/memmaps/train_images{cfg.NUM_EPOCHS}.tfrecords.gz'

    cfg.FILTER_SIZE = 3
    cfg.DEVIATION = 20
    cfg.MULTIPLE_BLUR_CHOICES = 90

    data = {
        'image': [],
        'mask': [],
        'input_img_1': [],
        'input_img_2': [],
    }

    split = 'train'

    cfg.DATA = data
    cfg.SPLIT = split
    cfg.COMPRESSION = None

    return cfg

