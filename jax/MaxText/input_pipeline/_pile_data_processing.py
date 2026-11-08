import math
import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import max_logging
import tensorflow as tf
import jax
from jax import numpy as jnp
import multihost_dataloading
from google.cloud import storage
from etils import epath


class PileDatasets():
    def __init__(self,
                mesh: str = None,
                name: str = 'pile',
                path: Optional[str] = None,
                num_infeed_hosts: int = 0,
                reset_for_eval: bool = False,
                batch_size: int = 8,
                seq_len: int = 2048,
                repeat: int = 1,
                seed: int = 9876,
                task_features: Optional[dict] = None,
                shuffle_buffer_size: Optional[int] = None,
                pad_id: int = 0,
                drop_remainder: bool = True,
                iter_file_nums: int = 2, # 100  500 steps/file,
                meta_dict: Optional[dict] = None,
                num_batches_to_skip: Optional[int] = None,
                only_eval: bool = False,
                zero_loss: bool = True,
                ):
        self.mesh = mesh
        self.name = name
        self.path = path
        self.num_infeed_hosts = num_infeed_hosts
        self.reset_for_eval = reset_for_eval
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.repeat = repeat
        self.seed = seed
        self.task_features = task_features
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pad_id = pad_id
        self.drop_remainder = drop_remainder
        self.iter_file_nums = iter_file_nums
        self.meta_dict = meta_dict
        self.num_batches_to_skip = num_batches_to_skip
        self.only_eval = only_eval
        self.zero_loss = zero_loss
        self.batch_padding_size = 0
        self.__post_init__()
        
    def __post_init__(self):
        if self.num_infeed_hosts == 0:
            self.num_infeed_hosts = jax.process_count()

        if not self.meta_dict or self.only_eval:
            self.meta_dict = {}
            self.init_meta()
        else:
            if self.meta_dict["file_in_data"] != 0:
                assert self.meta_dict["iter_file_nums"] == self.iter_file_nums, print(
                    f'iter_file_nums in meta_dict is not equal to cur args. => {self.meta_dict["iter_file_nums"]}≠'
                    f" {self.iter_file_nums}"
                )
            self.step_in_file = self.meta_dict.get('step_in_file')  # XD fix

        max_logging.log(f'meta_dict: {self.meta_dict}')
        self.seed = self.meta_dict['seed']
        self.dataset = self.load_tfrecord_dataset(fnames=self.path)
        self._peek = None
        self._state_before_peek = None

    def init_meta(self):
        self.meta_dict = {
                "seed": self.seed,
                "cur_files": self.meta_dict.get('cur_files', []),
                "file_in_data": 0,
                "step_in_file": 0,
                "iter_file_nums": self.iter_file_nums,
                "checkpoint_step": self.meta_dict.get('checkpoint_step', None),
            }
        self.step_in_file = 0

 #   def peek_padded(self):
  #      return self.get_next_padded()

    def reset(self):
        self.init_meta()
        self.dataset = self.load_tfrecord_dataset(fnames=self.path)

    def __iter__(self):
        return self.get_next_padded()
    
    def __next__(self):
        return self.get_next_padded()

    def get_next_padded(self):
        if self._peek is not None:
          output = self._peek
          self._peek = None
          self._state_before_peek = None
          return output
        unpadded = next(self.dataset)
        pad_size = int(self.batch_padding_size)
        if pad_size == 0:
            return unpadded
        return jax.tree_util.tree_map(
            lambda x: np.pad(x, [[0, pad_size]] + [[0, 0]] * (x.ndim - 1)),
            unpadded,
        )

    def get_global_batch_size(self, train_input):
        return self.batch_size * self.num_infeed_hosts

    def _parse_function(self, example_proto):
        feature_desc = {key: tf.io.VarLenFeature(tf.int64) for key in self.task_features}
        example = tf.io.parse_single_example(example_proto, feature_desc)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            example[name] = tf.sparse.to_dense(t, default_value=0)[ :self.seq_len]
        return example

    def convert(self, data):
        seq_len = self.seq_len
        model_needed_inputs = {}
        model_needed_inputs['inputs'] = data["input_ids"][:, : seq_len - 1]
        model_needed_inputs['targets'] = data["input_ids"][:, 1: seq_len]
        key = 'labels' if "labels" in data else 'input_ids'
        weights = data[key] >= 0 if self.zero_loss else data[key] > 0
        # label loss mask, origin bool type, but due the complie is int32
        model_needed_inputs['targets_segmentation'] = tf.cast(weights[:, 1:seq_len], dtype=tf.int32) 
        model_needed_inputs['inputs_segmentation'] = tf.ones_like(model_needed_inputs['inputs'])  # attention mask
        pos = tf.range(seq_len - 1)
        model_needed_inputs['inputs_position'] = model_needed_inputs['inputs_segmentation'] * pos
        model_needed_inputs['targets_position'] = model_needed_inputs['inputs_segmentation'] * pos  # no use, but complie have this key
        return model_needed_inputs

    def _load_file_dataset(self, fname):
        tf.random.set_seed(self.seed)
        ds = tf.data.Dataset.from_tensor_slices(fname)
        ds = ds.apply(tf.data.TFRecordDataset)
        # shard host data
        process_index = jax.process_index()
        # 在这里进行shard的话，不同的pod在相同的batch_size时，拿到的数据不一致
        ds = ds.shard(self.num_infeed_hosts, process_index)
        ds = ds.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        print(f'shuffle_buffer_size: {self.shuffle_buffer_size}')
        if self.shuffle_buffer_size is not None:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)
        padded_shapes = {key: self.seq_len for key in self.task_features}
        padding_values = {key: self.pad_id for key in self.task_features}
        ds = ds.padded_batch(
            batch_size=np.prod(self.batch_size),
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=True,
        )
        # lsp: batch之后进行shard。如果不进行shuffle，在batch化之前shard也行
        # ds = ds.shard(self.num_infeed_hosts, process_index)
        ds = ds.map(self.convert)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        if self.step_in_file: ds = ds.skip(self.step_in_file)  # XD fix
        # local data to global data
        ds = multihost_dataloading.MultiHostDataLoadIterator(ds, self.mesh)

        return ds

    def load_tfrecord_dataset(self, fnames):
        tf.random.set_seed(self.seed)
        assert isinstance(fnames, list)
        repeat_fnames = fnames * self.repeat
        N = math.ceil(len(repeat_fnames) / self.iter_file_nums)
        file_in_data = self.meta_dict["file_in_data"]
        max_logging.log(f'file_in_data: {file_in_data} N: {N}')
        for n in range(file_in_data, N, 1):
            fname = repeat_fnames[n * self.iter_file_nums : (n + 1) * self.iter_file_nums]
            self.meta_dict["cur_files"] = fname
            ds = self._load_file_dataset(fname)
            # ds = ds.as_numpy_iterator()
            for batch in ds:
                self.meta_dict["step_in_file"] += 1
                self.step_in_file += 1
                yield batch
            self.meta_dict["file_in_data"] += 1
            self.meta_dict["step_in_file"] = 0
            self.step_in_file = 0


SKIP_STEP_NAME = 'skip_file_and_step.json'
def record_file_and_step(step, config, train_input):  # lsp
    save_dir = epath.Path(config.checkpoint_dir)
    save_path = save_dir / str(step) / SKIP_STEP_NAME
    save_newest_path = save_dir / SKIP_STEP_NAME

    if not hasattr(train_input, 'meta_dict'):
        return
    meta_dict = train_input.meta_dict
    meta_dict['checkpoint_step'] = int(step)

    print(f'save_newest_path: {save_newest_path}')
    print(f'save_path: {save_path}')
    print(f'meta_dict: {meta_dict}')
    for k, v in meta_dict.items():
      print(k, type(v))

    # __import__('ipdb').set_trace()
    if jax.process_index() == 0:
      try:
        with save_newest_path.open('w', encoding='utf-8') as f1:
            json.dump(meta_dict, f1)

        with save_path.open('w', encoding='utf-8') as f2:
            json.dump(meta_dict, f2)
      except Exception as error:
        print(f'Write meta dict error: {error}')

    max_logging.log(f'Save skip_file_and_step successful... file_in_data: {meta_dict["file_in_data"]} || step_in_file: {meta_dict["step_in_file"]}')  # XD


def extract_pythia_datapath(dataset_path, eval_split):  # lsp
    if not dataset_path:
      return []
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    max_logging.log(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
    step_map_path = {}
    eval_pathes = []
    rerank = 0
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        if ".tfrecord" not in blob.name: continue
        try:
            step = int(blob.name.rsplit("pile.tfrecord.b", maxsplit=1)[-1])
        except:
            step = rerank
            rerank += 1
        path = f'gs://{os.path.join(bucket_name, blob.name)}'

        if eval_split in path:
            max_logging.log(f'eval path: {path}')
            eval_pathes.append(path)
            continue
        step_map_path[step] = path

    sorted_step_path = sorted(step_map_path.items(), key=lambda x: x[0])
    steps, pathes = zip(*sorted_step_path)
    if not isinstance(pathes, list):
        pathes = list(pathes)
    max_logging.log(f'pathes: {len(pathes)} eval_pathes: {len(eval_pathes)}')
    return pathes, eval_pathes


def extract_v3p5_longdata_files(dataset_path, eval_split=None):  # lsp
    random.seed(9876)
    client = storage.Client()
    #v3: us-east1-d -> common_datasets, v4: us-central2-b -> common_datasets_us-central2-b
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    train_files, valid_files = [], []
    train_long_files, train_short_files = [], []
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        if 'valid' in path:
            valid_files.append(path)
        else:
            if '.long' in path:
                train_long_files.append(path)
            else:
                train_short_files.append(path)
    # file size short：long = 1.5: 1, 为了保证short的token: long = 3: 7, 因此 short 取 (1 / 1.5) * (3 / 7) = 2 / 7
    short_k = min(3 * len(train_long_files) // 14, len(train_short_files))
    selected_short_files = random.sample(train_short_files, k=short_k)
    train_files = selected_short_files + train_long_files
    max_logging.log(f'selected_short_files: {len(selected_short_files)} train_long_files: {len(train_long_files)}')
    random.shuffle(train_files)
    max_logging.log(f'first 10 train files: {train_files[:10]}')
    valid_files = sorted(valid_files)
    max_logging.log(f'valid_files: {valid_files}')
    return train_files, valid_files


def extract_v3p5_data_files(dataset_path, eval_split):
    client = storage.Client()
    path = dataset_path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    # logging.info(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
    train_files, valid_files = [], []
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        if eval_split in path:
            valid_files.append(path)
        else:
            train_files.append(path)
    # train_files = sorted(train_files)
    # valid_files = sorted(valid_files)
    max_logging.log(f'Train file: {len(train_files)},  test file: {len(valid_files)}')
    max_logging.log(f'first 10 train files: {train_files[:10]}')
    max_logging.log(f'valid_files: {valid_files}')
    return train_files, valid_files


def extract_train_skip_step(job_log_dir, step, only_eval=False):  # lsp
    if job_log_dir is None:
        return {}
    model_dir = job_log_dir / "checkpoints"
    if step is not None:
        skip_file_and_step_path = model_dir / str(step) / SKIP_STEP_NAME
    else:
        skip_file_and_step_path = model_dir / SKIP_STEP_NAME
    max_logging.log(f"model_dir: {model_dir}")
    try:
        with skip_file_and_step_path.open('r', encoding='utf-8') as f:
            meta_dict = json.load(f)
        max_logging.log(f"Load skip_file_and_step_path: ’{skip_file_and_step_path}‘ Finished.......")
    except:
        max_logging.log(f"skip_file_and_step_path: ’{skip_file_and_step_path}‘ is not existed.......")
        meta_dict = {}

    if jax.process_index() == 0:
        mode = 'train_break_steps' if not only_eval else 'eval_metric_steps'
        back_meta_dict_dir = job_log_dir / mode
        if 'gs:' not in str(back_meta_dict_dir):
          os.makedirs(back_meta_dict_dir, exist_ok=True)
        back_meta_dict_path = back_meta_dict_dir /f'{meta_dict.get("checkpoint_step", None)}.json'
        with back_meta_dict_path.open('w', encoding='utf-8') as f1:
            json.dump(meta_dict, f1)
    return meta_dict


def make_pile_train_iterator(config, mesh, add_bos, add_eos):  # lsp
  train_name = f'{config.dataset_type}.train'
  eval_name = f'{config.dataset_type}.eval'
  if config.dataset_type == 'pile':
    train_pathes, eval_pathes = extract_pythia_datapath(config.dataset_path, config.eval_split)
  elif config.dataset_type == 'novel_4_32k':
    train_pathes, eval_pathes = extract_v3p5_longdata_files(config.dataset_path, config.eval_split)
  elif config.dataset_type == 'pretrain_4k':
    train_pathes, eval_pathes = extract_v3p5_data_files(config.dataset_path, config.eval_split)
  else:
    raise ValueError(f'Unknow ‘config.datase_dtype’={config.datase_dtype}')

  num_local_devices = jax.local_device_count()

  job_dir = epath.Path(config.run_name)
  try:
    only_eval = config.only_eval
  except:
    only_eval = False
  meta_dict = extract_train_skip_step(job_dir, step=config.training_num_batches_to_skip, only_eval=only_eval)
  # load_full_state_path
  print(f'meta_dict: {meta_dict}')

  task_features = ['input_ids']
  train_dataloader = PileDatasets(
                            mesh=mesh,
                            name=train_name, 
                            path=train_pathes, 
                            meta_dict=meta_dict,
                            batch_size=int(config.per_device_batch_size * num_local_devices),
                            seq_len=config.max_target_length,
                            repeat=config.epoch,
                            seed=config.data_shuffle_seed,
                            task_features=task_features,
                            shuffle_buffer_size=config.train_shuffle_buffer_size,
                            num_batches_to_skip=None,
                            only_eval=False,
                            zero_loss=config.zero_loss,
                            iter_file_nums=config.iter_file_nums,
                            )
  eval_dataloader = None
  if eval_pathes:
    eval_dataloader = PileDatasets(
                            mesh=mesh,
                            name=eval_name, 
                            path=eval_pathes, 
                            meta_dict={},
                            batch_size=int(config.eval_per_device_batch_size * num_local_devices),
                            seq_len=config.max_target_length,
                            repeat=config.epoch,
                            seed=config.data_shuffle_seed,
                            task_features=task_features,
                            shuffle_buffer_size=config.eval_shuffle_buffer_size,
                            num_batches_to_skip=None,
                            only_eval=False,
                            zero_loss=config.zero_loss,
                            iter_file_nums=config.iter_file_nums,
                            )
  return train_dataloader, eval_dataloader, None