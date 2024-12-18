import ml_collections

def get_default_configs():
    config = ml_collections.ConfigDict()
    
    config.name = "model transfusion"
    
    config.image_size = 128
    config.keep_checkpoints = 3
    config.level = 1
    config.learning_rate = 0.01
    config.e = 1e-3
    config.a = .5

    config.warmup_epochs = 5
    config.momentum = 0.9
    config.epochs = 200
    config.log_frequency = 10

    config.half_precision = False
    config.batch_size = 32
    config.mini_batch_size = 32
    config.log_every_steps = True
    config.no_train_steps = 2000
    
    config.plots_save_dir = "/home/anirudhan/project/fusion/results/plots"
    config.checkpoint_dir = "/home/anirudhan/project/fusion/results/checkpoints"
    config.log_dir = "/home/anirudhan/project/fusion/results/logs"
    
    return config
