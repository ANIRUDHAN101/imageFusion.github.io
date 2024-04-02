
#%%
from train.train import *
import jax
import einops
import jaxwt as jwt
from functools import partial
import flax.linen as nn

Conv3x3 = partial(nn.Conv, kernel_size=(3,3), padding='SAME')
Conv7x7 = partial(nn.Conv, kernel_size=(7,7), padding='SAME')
rearange_width_first = lambda x: einops.rearrange(x, "n h w c -> n c h w")
rearange_height_first = lambda x: einops.rearrange(x, "n c h w -> n c w h")
Conv1x1 = partial(nn.Conv, padding='SAME')

class ConvBlock(nn.Module):
    dimention : int

    @nn.compact
    def __call__(self, image, train=False):
        feature = Conv3x3(self.dimention)(image)
        feature = nn.activation.gelu(feature)
        feature = nn.BatchNorm(use_running_average = not train)(feature)
        return feature

class FusionModel(nn.Module):
    @nn.compact
    def __call__(self, image1, image2, train=False):
        feature1 = ConvBlock(16)(image1, train)
        feature2 = ConvBlock(16)(image2, train)
        return feature1 + feature2

no_train_samples = 1650

workdir = '/home/anirudhan/project/fusion/results'
# checkpoint_manager = create_checkpoints_manager(config, '/content')
config = get_default_configs()
config.image_size = 128

rng = jax.random.PRNGKey(0)

writer = metric_writers.create_default_writer(
    logdir=workdir, just_logging=jax.process_index() != 0
)

if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
local_batch_size = config.batch_size // jax.process_count()

steps_per_epoch = (
    no_train_samples // config.batch_size
)

if config.no_train_steps <= 0:
    num_steps = int(steps_per_epoch * config.batch_size)
else :
    num_steps = config.no_train_steps
    
steps_per_checkpoint = steps_per_epoch * 10
base_learning_rate = config.learning_rate * config.batch_size / 256.0

model = FusionModel()

learning_rate_fn = create_learning_rate_fn(
    config, base_learning_rate, steps_per_epoch
)
#%%
state = create_train_state(rng, config, model, learning_rate_fn)

step_offset = int(state.step)
# state = jax_utils.replicate(state)
train_step = functools.partial(train_step, learning_rate_fn=learning_rate_fn)

print(jax.tree_map(lambda x: x.shape, state.params))
print(jax.tree_map(lambda x: x.dtype, state.params))
# steps_per_epoch = (
#       no_train_samples // config.batch_size
#   )

# if config.no_train_steps <= 0:
#     num_steps = int(steps_per_epoch * config.batch_size)

# else :
#     num_steps = config.no_train_steps

# steps_per_checkpoint = steps_per_epoch * 10
# base_learning_rate = config.learning_rate * config.batch_size / 256.0

# model = ImageFusion()

# learning_rate_fn = create_learning_rate_fn(
#     config, base_learning_rate, steps_per_epoch
# )

# state = create_train_state(rng, config, model, learning_rate_fn)
# checkpoint_manager = create_checkpoints_manager(config, '/content/checkpoints')
# # state = restore_last_checkpoint(checkpoint_manager)
# # step_offset > 0 if restarting from checkpoint
# step_offset = int(state.step)
# state = jax_utils.replicate(state)

# # p_train_step = jax.pmap(
# #     functools.partial(train_step, learning_rate_fn=learning_rate_fn),
# #     axis_name='batch',
# # )
# train_step = functools.partial(train_step, learning_rate_fn=learning_rate_fn)
# # p_eval_step = jax.pmap(eval_step, axis_name='batch')

# train_metrics = []
# hooks = []
# if jax.process_index() == 0:
#     hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
# train_metrics_last_t = time.time()
# train_iter = create_input_iter(numpy_train_data)

# logging.info('Initial compilation, this might take some minutes...')
# for step, batch in zip(range(step_offset, num_steps), train_iter):
#     state, metrics = train_step(state, batch)
#     logging.info(f'Step {step}: {metrics}')
#     for h in hooks:
#         h(step)

#     if step == step_offset:
#         logging.info('Initial compilation completed.')

#     if config.get('log_every_steps'):
#         train_metrics.append(metrics)
#         if (step + 1) % config.log_every_steps == 0:
#             train_metrics = common_utils.get_metrics(train_metrics)
#             summary = {
#                 f'train_{k}': v
#                 for k, v in jax.tree_util.tree_map(
#                     lambda x: x.mean(), train_metrics
#                 ).items()
#             }
#             summary['steps_per_second'] = config.log_every_steps / (
#                 time.time() - train_metrics_last_t
#             )
#             writer.write_scalars(step + 1, summary)
#             train_metrics = []
#             train_metrics_last_t = time.time()

#     if (step + 1) % steps_per_epoch == 0:
#         epoch = step // steps_per_epoch
#         eval_metrics = []

#         # sync batch statistics across replicas
#         state = sync_batch_stats(state)

#     if step == no_train_samples // config.batch_size * config.epochs - 1: break