import sys
sys.path.append('/home/anirudhan/project/fusion')

import functools
import os
import shutil
import time
from typing import Any
from functools import partial
import jax
import optax
import ml_collections
import numpy as np
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state
from flax import jax_utils
from flax import linen as nn
from flax.training import orbax_utils
from clu import metric_writers, periodic_actions
from absl import logging
import orbax
from clu import metric_writers
from clu import periodic_actions
from config.jax_train_config import get_default_configs
from utils.train import check_and_replace_nan
from src.loss.jax import charbonnier_loss, wavelet_loss, ssim_loss, fft_loss
import jax.numpy as jnp
from utils.train import save_plot, denormalize_val_image

config = get_default_configs()
CHECKPOINT_DIR = config.checkpoint_dir

if os.path.exists(CHECKPOINT_DIR):
    shutil.rmtree(CHECKPOINT_DIR)


def get_dtype():
    #platform = 'tpu'  # setting it as bloat16 due to performance improvements
    platform = jax.local_devices()[0].platform
    if config.half_precision:
        if platform == 'tpu':
            dtype = jnp.bfloat16
        else:
            dtype = jnp.float16
    else:
        dtype = jnp.float32
    return dtype


def prepare_train_data(data):
    local_device_count = jax.local_device_count()
    dtype = get_dtype()

    def prepare_data(data):
        data = check_and_replace_nan(data)
        # data = jnp.array(data, dtype=dtype)
        return data

    return jax.tree_util.tree_map(prepare_data, data)


def create_input_iter(data):
    itr = map(prepare_train_data, data)
    return itr


class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale
    key: jax.Array


def create_learning_rate_fn(config: ml_collections.ConfigDict, base_learning_rate: float, steps_per_epoch: int):
    warmup_fn = optax.linear_schedule(
        init_value=0,
        end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch
    )

    cosine_epochs = max(1, config.epochs - config.warmup_epochs)

    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch
    )

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch],
    )

    return schedule_fn

# different implementation of create train state function
def _init_model(rng, config, model):
    image_shape = (1, config.image_size, config.image_size, 3)
    dtype = get_dtype()

    @partial(jax.jit, static_argnames=['train'])
    def init(*args, **kwargs):
        return model.init(*args, **kwargs)

    variables = model.init(rng, image1=jnp.ones(image_shape, dtype), image2=jnp.ones(image_shape, dtype), train=True)
    platform = jax.local_devices()[0].platform
    dynamic_scale = None

    return variables, platform, dynamic_scale


def create_train_state(rng, config, model, learning_rate_fn=None):
    main_key, params_key, dropout_key = jax.random.split(rng, 3)
    variables, platform, dynamic_scale = _init_model(params_key, config, model)
    
    # train state using lookahead optimizer and  multi-step optimizer
    if learning_rate_fn is None:
        optimizer = optax.adam(
            learning_rate=config.learning_rate,
        )
        tx = optax.MultiSteps(optimizer, every_k_schedule=config.batch_size // config.mini_batch_size)

    else:
        tx = optax.sgd(
            learning_rate=learning_rate_fn,
            momentum=config.momentum,
            nesterov=True
        )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        key=dropout_key,
        tx=tx,
        batch_stats=variables['batch_stats'],
        dynamic_scale=dynamic_scale,
    ), main_key

def create_checkpoints_manager(config, save_dir):
    async_checkpointer = orbax.checkpoint.AsyncCheckpointer(
        orbax.checkpoint.PyTreeCheckpointHandler(), timeout_secs=50)

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        save_dir, async_checkpointer, options)
    return checkpoint_manager

def save_checkpoints(state, checkpoint_manager, config):
    ckpt = {'model': state}
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpoint_manager.save(state.step, ckpt, save_kwargs={'save_args': save_args})
    checkpoint_manager.wait_until_finished()

def restore_last_checkpoint(checkpoint_manager):
    step = checkpoint_manager.latest_step()
    state = checkpoint_manager.restore(step)
    return state

@functools.partial(jax.jit, static_argnames=('learning_rate_fn'), donate_argnums=(0, 1, 3))
def train_step(state, batch, learning_rate_fn, dropout_key=None):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        gt_image = batch['image']
        mask = batch['mask']
        image1 = batch['input_img_1']
        image2 = batch['input_img_2']

        predicted_image, updates = state.apply_fn({
            'params': params,
            'batch_stats': state.batch_stats
        },
            image1=image1, image2=image2,
            train=True,
            rngs={'dropout': dropout_train_key},
            mutable=['batch_stats']
        )

        loss = charbonnier_loss(predicted_image, gt_image) + config.a * wavelet_loss(predicted_image, gt_image)

        # weight_penalty_parms = jax.tree_util.tree_leaves(params)
        # weight_decay = 0.0001
        # weight_l2 = sum(
        #     jnp.sum(x ** 2) for x in weight_penalty_parms if x.ndim > 1
        # )
        # weight_panelty = weight_decay * 0.5 * weight_l2
        # loss = loss + weight_panelty
        return loss, updates

    metrics = {}

    step = state.step
    dynamic_scale = state.dynamic_scale

    if learning_rate_fn is not None:
        metrics['lr'] = learning_rate_fn(step)

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
    else:
        is_fin = None
        dynamic_scale = None
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)

    new_state = aux[1]
    new_state = apply_grads(state, grads, dynamic_scale, new_state['batch_stats'],is_fin)
    loss = aux[0]
    
    metrics['train loss'] = loss

    return new_state, metrics


@functools.partial(jax.jit, donate_argnums=(0, 1, 2, 3, 4))
def apply_grads(state, grads, dynamic_scale, batch_stats, is_fin):
    new_state = state.apply_gradients(grads=grads, batch_stats=batch_stats)
    if dynamic_scale:
        new_state = new_state.replace(
            opt_state=jax.tree_util.tree_map(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state,
                state.opt_state,
            ),
            params=jax.tree_util.tree_map(
                functools.partial(jnp.where, is_fin), new_state.params, state.params
            ),
            dynamic_scale=dynamic_scale,
        )
    return new_state

@jax.jit
def eval_step(state, batch):
    gt_image = batch['image']
    mask = batch['mask']
    image1 = batch['input_img_1']
    image2 = batch['input_img_2']

    predicted_image = state.apply_fn({
        'params': state.params,
        'batch_stats': state.batch_stats
    },
        image1=image1, image2=image2,
        train=False
    )

    # loss = charbonnier_loss(predicted_image, gt_image) + config.a * wavelet_loss(predicted_image, gt_image)
    loss = charbonnier_loss(predicted_image, gt_image) + fft_loss(predicted_image, gt_image)
    return loss

def test_step(state, dataset, val_data_size):
    test_loss = []
    for i, image in enumerate(dataset):
        prediction = state.apply_fn({'params': state.params,
                                     'batch_stats': state.batch_stats
                                     },
                                    image1=image['imageA'], image2=image['imsgeB'],
                                    train=False)
        prediction = jax.device_put(prediction, jax.devices()[0])
        prediction = np.asarray(prediction)
        loss = ssim_loss(denormalize_val_image(prediction), denormalize_val_image(image['Fusion']))
        test_loss.append(loss)
        if i == 0:
            save_plot(image['imageA'][0], image['imsgeB'][0], prediction[0], image['Fusion'][0])
        if i == val_data_size:
            break
    return np.mean(test_loss)
