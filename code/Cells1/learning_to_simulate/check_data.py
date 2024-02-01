import collections
import functools
import json
import os
import pickle
import sys
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tree
import config

from learning_to_simulate import learned_simulator
from learning_to_simulate import noise_utils
from learning_to_simulate import reading_utils
import cProfile

import IPython.display as display

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.')
flags.DEFINE_enum('eval_split', 'test', ['train', 'valid', 'test'],
                  help='Split to use when running evaluation.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_integer('num_steps', int(2e7), help='Number of steps of training.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_string('model_path', None,
                    help=('The path for saving checkpoints of the model. '
                          'Defaults to a temporary directory.'))
flags.DEFINE_string('output_path', None,
                    help='The path for saving outputs (e.g. rollouts).')

FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}


def batch_concat(dataset, batch_size):
    """We implement batching as concatenating on the leading axis."""

    # We create a dataset of datasets of length batch_size.
    windowed_ds = dataset.window(batch_size)

    # The plan is then to reduce every nested dataset by concatenating. We can
    # do this using tf.data.Dataset.reduce. This requires an initial state, and
    # then incrementally reduces by running through the dataset

    # Get initial state. In this case this will be empty tensors of the
    # correct shape.
    initial_state = tree.map_structure(
        lambda spec: tf.zeros(  # pylint: disable=g-long-lambda
            shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),
        dataset.element_spec)

    # We run through the nest and concatenate each entry with the previous state.
    def reduce_window(initial_state, ds):
        return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))

    return windowed_ds.map(
        lambda *x: tree.map_structure(reduce_window, initial_state, x))


def prepare_inputs(tensor_dict):
    """Prepares a single stack of inputs by calculating inputs and targets.

  Computes n_particles_per_example, which is a tensor that contains information
  about how to partition the axis - i.e. which nodes belong to which graph.

  Adds a batch axis to `n_particles_per_example` and `step_context` so they can
  later be batched using `batch_concat`. This batch will be the same as if the
  elements had been batched via stacking.

  Note that all other tensors have a variable size particle axis,
  and in this case they will simply be concatenated along that
  axis.



  Args:
    tensor_dict: A dict of tensors containing positions, and step context (
    if available).

  Returns:
    A tuple of input features and target positions.

  """
    # Position is encoded as [sequence_length, num_particles, dim] but the model
    # expects [num_particles, sequence_length, dim].
    pos = tensor_dict['position']
    pos = tf.transpose(pos, perm=[1, 0, 2])

    # The target position is the final step of the stack of positions.
    target_position = pos[:, -1]

    # Remove the target from the input.
    tensor_dict['position'] = pos[:, :-1]

    # Compute the number of particles per example.
    num_particles = tf.shape(pos)[0]
    # Add an extra dimension for stacking via concat.
    tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]

    if 'step_context' in tensor_dict:
        # Take the input global context. We have a stack of global contexts,
        # and we take the penultimate since the final is the target.
        tensor_dict['step_context'] = tensor_dict['step_context'][-2]
        # Add an extra dimension for stacking via concat.
        tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
    return tensor_dict, target_position


def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())


def _read_metadata2(data_path):
    with open(data_path, 'rt') as fp:
        return json.loads(fp.read())




def parse_serialized_simulation_example(example_proto, metadata):
    """Parses a serialized simulation tf.SequenceExample.

  Args:
    example_proto: A string encoding of the tf.SequenceExample proto.
    metadata: A dict of metadata for the dataset.

  Returns:
    context: A dict, with features that do not vary over the trajectory.
    parsed_features: A dict of tf.Tensors representing the parsed examples
      across time, where axis zero is the time axis.

  """

    context, parsed_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=_CONTEXT_FEATURES,
        sequence_features=_FEATURE_DESCRIPTION)

    print("Context", context)
    print("Parsed features: ", parsed_features)
    for feature_key, item in parsed_features.items():
        print("F key: ", feature_key)
        convert_fn = functools.partial(
            convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
        parsed_features[feature_key] = tf.py_function(
            convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])

    # There is an extra frame at the beginning so we can calculate pos change
    # for all frames used in the paper.
    position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]

    # Reshape positions to correct dim:
    parsed_features['position'] = tf.reshape(parsed_features['position'],
                                             position_shape)
    # Set correct shapes of the remaining tensors.
    sequence_length = metadata['sequence_length'] + 1
    if 'context_mean' in metadata:
        context_feat_len = len(metadata['context_mean'])
        parsed_features['step_context'] = tf.reshape(
            parsed_features['step_context'],
            [sequence_length, context_feat_len])
    # Decode particle type explicitly
    context['particle_type'] = tf.py_function(
        functools.partial(convert_fn, encoded_dtype=np.int64),
        inp=[context['particle_type'].values],
        Tout=[tf.int64])
    context['particle_type'] = tf.reshape(context['particle_type'], [-1])
    return context, parsed_features


@tf.function
def check(data_path, batch_size, split, model='one_step'):
    metadata = _read_metadata(data_path)

    ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
    ds = ds.map(functools.partial(
        parse_serialized_simulation_example, metadata=metadata))

    # Splits an entire trajectory into chunks of 7 steps.
    # Previous 5 velocities, current velocity and target.
    split_with_window = functools.partial(
        reading_utils.split_trajectory,
        window_length=INPUT_SEQUENCE_LENGTH + 1)
    ds = ds.flat_map(split_with_window)
    # Splits a chunk into input steps and target steps
    ds = ds.map(prepare_inputs)
    # If in train mode, repeat dataset forever and shuffle.
    # ds = ds.repeat()
    # ds = ds.shuffle(512)
    # Custom batching on the leading axis.
    # ds = batch_concat(ds, batch_size)

    print(ds)

@tf.function
def checkx(data_path, batch_size, split, model='one_step'):
    metadata = _read_metadata(data_path)

    ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
    ds = ds.map(functools.partial(
        parse_serialized_simulation_example, metadata=metadata))
    for v in ds:
        print(v)

@tf.function
def check2(data_path="/tmp/datasets/Sand", batch_size=101, split="train"):
    ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
    for raw_record in ds.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)


def decode_fn(record_bytes):
    print("RBY:", record_bytes)
    # return record_bytes
    return tf.io.parse_single_sequence_example(record_bytes, context_features=_CONTEXT_FEATURES,sequence_features=_FEATURE_DESCRIPTION)

def decode_fn2(record_bytes):
    return record_bytes
    # return tf.io.parse_single_sequence_example(record_bytes, context_features=_CONTEXT_FEATURES,sequence_features=_FEATURE_DESCRIPTION)

def convert_to_tensor(x, encoded_dtype):
  if len(x) == 1:
    out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
  else:
    out = []
    for el in x:
      out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
  out = tf.convert_to_tensor(np.array(out))
  return out

tf.enable_eager_execution()
def check3(data_path="/tmp/datasets/Sand", split="train"):
    ds = tf.data.TFRecordDataset(os.path.join(data_path, f'{split}.tfrecord'))
    for record in ds.map(decode_fn):
        v1, v2 = record
        print(v1)
        print(v2)
        print(v1['particle_type'].get_shape())
        print(v2['position'].get_shape())
        print("Size", sys.getsizeof(v1['particle_type'].values))
        v = convert_to_tensor(v1['particle_type'].values, np.int64)
        print(v.shape)
        print(v)
        v2 = convert_to_tensor(v2['position'].values, _FEATURE_DTYPES['position']['in'])
        print(v2.shape)
        # print(v2['position'].values)
        break
tf.enable_eager_execution()
def check3x(data_path="/tmp/datasets/Sand", split="train"):
    ds = tf.data.TFRecordDataset(os.path.join(data_path, f'{split}.tfrecord'))
    metadata = _read_metadata(data_path)
    for record in ds.map(decode_fn):
        context, parsed_features = record

        for feature_key, item in parsed_features.items():
            print("F key: ", feature_key)
            convert_fn = functools.partial(
                convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
            parsed_features[feature_key] = tf.py_function(
                convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])
        context['particle_type'] = tf.py_function(
            functools.partial(convert_fn, encoded_dtype=np.int64),
            inp=[context['particle_type'].values],
            Tout=[tf.int64])
        context['particle_type'] = tf.reshape(context['particle_type'], [-1])

        position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]

        # Reshape positions to correct dim:
        parsed_features['position'] = tf.reshape(parsed_features['position'],
                                                 position_shape)
        print(parsed_features['position'].shape)
        print(context['particle_type'].shape)

        break

tf.enable_eager_execution()
def check4x(data_path="%s/train_0.tfdata" % config.TEST1_DIR):
    ds = tf.data.TFRecordDataset(data_path)
    metadata = _read_metadata2("%s/metadata.json" % config.TEST1_DIR)
    for record in ds.map(decode_fn):
        context, parsed_features = record

        for feature_key, item in parsed_features.items():
            print("F key: ", feature_key)
            convert_fn = functools.partial(
                convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
            parsed_features[feature_key] = tf.py_function(
                convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])
        context['particle_type'] = tf.py_function(
            functools.partial(convert_fn, encoded_dtype=np.int64),
            inp=[context['particle_type'].values],
            Tout=[tf.int64])
        context['particle_type'] = tf.reshape(context['particle_type'], [-1])

        position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]

        # Reshape positions to correct dim:
        parsed_features['position'] = tf.reshape(parsed_features['position'],
                                                 position_shape)
        print(parsed_features['position'].shape)
        print(context['particle_type'].shape)
tf.enable_eager_execution()
def check4(data_path="%s/train_0.tfdata" % config.TEST1_DIR):
    ds = tf.data.TFRecordDataset(data_path)
    for record in ds.map(decode_fn):

        v1, v2 = record
        print(v1['particle_type'].get_shape())
        print(v2['position'].get_shape())
        v = convert_to_tensor(v1['particle_type'].values, np.int64)
        print(v.shape)
        v2 = convert_to_tensor(v2['position'].values, _FEATURE_DTYPES['position']['in'])
        print(v2.shape)
        break


@tf.function
def check_cells(data_path, meta_path, batch_size=2):
    ds = tf.data.TFRecordDataset([data_path])
    metainfo = _read_metadata2(meta_path)

    ds = ds.map(functools.partial(
        parse_serialized_simulation_example, metadata=metainfo))

    # Splits an entire trajectory into chunks of 7 steps.
    # Previous 5 velocities, current velocity and target.
    split_with_window = functools.partial(
        reading_utils.split_trajectory,
        window_length=INPUT_SEQUENCE_LENGTH + 1)
    ds = ds.flat_map(split_with_window)
    # Splits a chunk into input steps and target steps
    ds = ds.map(prepare_inputs)
    # If in train mode, repeat dataset forever and shuffle.
    # ds = ds.repeat()
    # ds = ds.shuffle(512)
    # Custom batching on the leading axis.
    # ds = batch_concat(ds, batch_size)

    print(ds)


@tf.function
def check_cells_mutiples(data_paths, meta_paths, batch_size=2):
    n_file = len(data_paths)
    dss = []
    for i in range(n_file):
        data_path = data_paths[i]
        meta_path = meta_paths[i]
        ds = tf.data.TFRecordDataset([data_path])
        metainfo = _read_metadata2(meta_path)

        ds = ds.map(functools.partial(
            parse_serialized_simulation_example, metadata=metainfo))

        # Splits an entire trajectory into chunks of 7 steps.
        # Previous 5 velocities, current velocity and target.
        split_with_window = functools.partial(
            reading_utils.split_trajectory,
            window_length=INPUT_SEQUENCE_LENGTH + 1)
        ds = ds.flat_map(split_with_window)
        # Splits a chunk into input steps and target steps
        ds = ds.map(prepare_inputs)
        dss.append(ds)
    ds = dss[0]
    for i in range(1, len(dss)):
        ds = ds.concatenate(dss[i])

    # If in train mode, repeat dataset forever and shuffle.
    ds = ds.repeat()
    ds = ds.shuffle(512)
    # Custom batching on the leading axis.
    ds = batch_concat(ds, batch_size)

    print(ds)


if __name__ == "__main__":

    # check(data_path="/tmp/datasets/Sand", batch_size=101, split="train")
    # checkx(data_path="/tmp/datasets/Sand", batch_size=101, split="train")
    # exit(-1)
    # check3()
    # check3x()
    check4x()
    # check4()
    exit(-1)
    # check_cells(data_path="%s/%s.tfdata" % (config.TEST1_DIR, 1), meta_path="%s/metadata_%s.json" % (config.TEST1_DIR, 1))
    data_paths = []
    meta_paths = []
    for i in range(3):
        data_paths.append("%s/%s.tfdata" % (config.TEST1_DIR, i))
        meta_paths.append("%s/metadata.json" % (config.TEST1_DIR))

    check_cells_mutiples(data_paths, meta_paths)
    # check3(data_path="/tmp/datasets/Sand", split="test")
