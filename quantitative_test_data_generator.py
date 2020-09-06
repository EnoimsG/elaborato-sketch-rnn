import numpy as np
import time
import random
import codecs
import collections
import os
import math
import json
import tensorflow as tf

import sketch_rnn_train
from model import *
from utils import *

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

os.environ["CUDA_VISIBLE_DEVICES"]= "1"
tf.logging.info("Using gpu:" + os.environ["CUDA_VISIBLE_DEVICES"])


def load_env_compatible(data_dir, model_dir):
    """Loads environment for inference mode (used in jupyter notebook)."""
    model_params = get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        data = json.load(f)
    fix_list = ['conditional', 'is_training', 'use_input_dropout',
                'use_output_dropout', 'use_recurrent_dropout']
    for fix in fix_list:
        data[fix] = (data[fix] == 1)
    model_params.parse_json(json.dumps(data))
    return sketch_rnn_train.load_dataset(data_dir, model_params, inference_mode=True)


def load_model_compatible(model_dir):
    """Loads model for inference mode, used in jupyter notebook."""
    model_params = get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        data = json.load(f)
    fix_list = ['conditional', 'is_training', 'use_input_dropout',
                'use_output_dropout', 'use_recurrent_dropout']
    for fix in fix_list:
        data[fix] = (data[fix] == 1)
    model_params.parse_json(json.dumps(data))

    model_params.batch_size = 1  # only sample one at a time
    eval_model_params = copy_hparams(model_params)
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = 0
    sample_model_params = copy_hparams(eval_model_params)
    sample_model_params.max_seq_len = 1  # sample one point at a time
    return [model_params, eval_model_params, sample_model_params]


def encode(input_strokes):
    strokes = to_big_strokes(
        input_strokes, eval_hps_model.max_seq_len).tolist()
    strokes.insert(0, [0, 0, 1, 0, 0])
    seq_len = [len(input_strokes)]
    # draw_strokes(to_normal_strokes(np.array(strokes)))
    return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]


def decode(z_input=None, draw_mode=True, temperature=0.1, factor=0.2):
    z = None
    if z_input is not None:
        z = [z_input]
    sample_strokes, m = sample(
        sess, sample_model, seq_len=eval_model.hps.max_seq_len, temperature=temperature, z=z)
    strokes = to_normal_strokes(sample_strokes)
    if draw_mode:
        draw_strokes(strokes, factor)
    return strokes


data_dir = "dataset/quickdraw"
model_dir = "models/"

cat_experiments = [1, 4, 7, 10, 13, 16]
tractor_experiments = [2, 5, 8, 11, 14, 17]
number_of_samples = 1000

for i in cat_experiments:
    print("Running cat experiment:", str(i))
    [train_set, valid_set, test_set, hps_model, eval_hps_model,
        sample_hps_model] = load_env_compatible(data_dir, model_dir + str(i))
    # construct the sketch-rnn model here:
    sketch_rnn_train.reset_graph()
    model = Model(hps_model)
    eval_model = Model(eval_hps_model, reuse=True)
    sample_model = Model(sample_hps_model, reuse=True)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # loads the weights from checkpoint into our model
    sketch_rnn_train.load_checkpoint(sess, model_dir + str(i))

    results = []
    for n in range(number_of_samples):
        # 1. get a sample drawing from the test set
        stroke = test_set.random_sample()
        # 2. encode and get z
        z = encode(stroke)
        # 3. append decoded version
        results.append(decode(z, temperature=0.8, draw_mode=False))
    np.save('experiments_results/cat/' + str(i) +
            '.npy', np.array(results), allow_pickle=True)

for i in tractor_experiments:
    print("Running tractor experiment:", str(i))
    [train_set, valid_set, test_set, hps_model, eval_hps_model,
        sample_hps_model] = load_env_compatible(data_dir, model_dir + str(i))
    # construct the sketch-rnn model here:
    sketch_rnn_train.reset_graph()
    model = Model(hps_model)
    eval_model = Model(eval_hps_model, reuse=True)
    sample_model = Model(sample_hps_model, reuse=True)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # loads the weights from checkpoint into our model
    sketch_rnn_train.load_checkpoint(sess, model_dir + str(i))

    results = []
    for n in range(number_of_samples):
        # 1. get a sample drawing from the test set
        stroke = test_set.random_sample()
        # 2. encode and get z
        z = encode(stroke)
        # 3. append decoded version
        results.append(decode(z, temperature=0.8, draw_mode=False))
    np.save('experiments_results/tractor/' + str(i) +
            '.npy', np.array(results), allow_pickle=True)
