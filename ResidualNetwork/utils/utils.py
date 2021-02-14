from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import runpy
import ast
import pprint
import copy
import os
import sys

import numpy as np
import six
from six import string_types
from six.moves import range
import tensorflow as tf

def create_logdir(args, base_config):

    logdir = base_config["logdir"]
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    tm_suf = datatime.datatime.now().strftime("%Y-%m-%d_%H-%M-%S")
    shutil.copy(args,config_file, os.path.join(logdir, "config_{}.py".format(tm_suf)))

    with open(os.path.join(logdir, "cmd-args_{}.log".format(tm_suf)), "w"):
        f.write(" ".join(sys.argv))

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_log = open(os.path.join(logdir, "stdout_{}.log".format(tm_suf)), "a", 1)
    stderr_log = open(os.path.join(logdir, "stderr_{}.log".format(tm_suf)), "a", 1)
    sys.stdout = Logger(sys.stdout, stdout_log)
    sys.stderr = Logger(sys.stderr, stderr_log)

    return old_stdout, old_stderr, stdout_log, stderr_log

def create_model(args, base_config, config_module, base_model, checkpoint = None):
    
    train_config = copy.deepcopy(base_config)
    eval_config = copy.deepcopy(base_config)
    infer_config = copy.deepcopy(base_config)

    if args.mode == "train":
        """
        将base_params和train_params整合到一起

        """
        if "train_params" in config_module:
            nested_update(train_config, copy.deepcopy(config_module["train_params"]))
            deco_print("Training config:")
            pprint.pprint(train_config)
    
    if args.mode == "eval":
        
        if "eval_params" in config_module:
            nested_update(eval_config, copy.deepcopy(config_module["eval_params"]))
        deco_print("Evaluation config:")
        pprint.pprint(eval_config)

    if args.mode == "infer":
        if args.infer_output_file is None:
            raise ValueError("infer_output_file parameter is reuqired in infer mode")
        # if "infer_params" in config_module:
        nested_update(infer_config, copy.deepcopy(config_module["eval_params"]))
        deco_print("Inference config:")
        pprint.pprint(infer_config)
    
    if args.mode == "interactive_infer":
        # if "infer_params" in config_module:
        nested_update(infer_config, copy.deepcopy(config_module["interactive_infer_params"]))
        deco_print("Inference config:")
        pprint.pprint(infer_config)
    
    if args.mode == "train":
        model = base_model(params = train_config, mode = "train", detection = args.detection, dataset = "dev", num_set = args.set)
        model.compile()
    
    elif args.mode == "eval":
        model = base_model(params = eval_config, mode = "eval", detection = args.detection, dataset = "dev", num_set = args.set)
        model.compile(force_var_reuse = False)
    
    elif args.mode == "infer":
        model = base_model(params = infer_config, mode = "infer", detection = args.detection, dataset = "test", num_set = args.set)
        model.compile(checkpoint = checkpoint)
        # model.compile(force_var_reuse = False)
    return model

def flatten_dict(dct):
    flat_dict = {}
    # 考虑原字典value也是字典的情况
    for key, value in dct.items():
        if isinstance(value, (int,float,string_types,bool)):
            flat_dict.update({key:value})
        elif isinstance(value, dict):
            flat_dict.update(
                    {key + '/' + k: v for k,v in flatten_dict(dct[key]).items()})
    return flat_dict

def nest_dict(flat_dict):
    """
    input flat_dict:
    {
        "optimizer": "Adam",
        "lr_policy_params/learning_rate": 0.0001
    }
    return nst_dict:
    {
        "optimizer": "Adam",
        "lr_policy_params": {"learning_rate": 0.0001}
    }
    """
    nst_dict = {}
    for key,value in flat_dict.items():
        nest_keys = key.split("/")
        cur_dict = nst_dict
        for i in range(len(nest_keys) - 1):
            if nest_keys[i] not in cur_dict:
                cur_dict[nest_keys[i]] = {}
            cur_dict = cur_dict[nest_keys[i]]
        cur_dict[nest_keys[-1]] = value
    
    return nst_dict

def nested_update(org_dict, upd_dict):
  for key, value in upd_dict.items():
    if isinstance(value, dict):
        if key in org_dict:
            if not isinstance(org_dict[key], dict):
                raise ValueError("Mismatch between org_dict and upd_dict at node {}".format(key))
            nested_update(org_dict[key], value)
        else:
            org_dict[key] = value
    else:
        org_dict[key] = value


def get_base_config(args):
    
    parser = argparse.ArgumentParser(description="Experiments parameters")
    parser.add_argument("--config_file", required = True, help = "Path to the config file")
    parser.add_argument("--dataset", default = "dev")
    parser.add_argument("--detection", required = True)
    parser.add_argument("--set", required = True)
    parser.add_argument("--mode", default = "train", help = "train, eval, infer, interactive_infer")
    parser.add_argument("--continue_learning", action = "store_true")
    args, unknown = parser.parse_known_args(args)

    # run_path返回一个top-level的字典
    config_module = runpy.run_path(args.config_file, init_globals = {'tf':tf})

    base_config = config_module.get('base_params', None)
    if base_config is None:
        raise ValueError("base_params has to be defined in the config file")
    
    # 这里返回一个class
    base_model = config_module.get('base_model', None)
    if base_model is None:
        raise ValueError("base_model class has to be defined in the config file")
    
    # 读完config_file之后，有一些之前的参数可能会被重写
    parser_unk = argparse.ArgumentParser()
    
    for pm, value in flatten_dict(base_config).items():
        if type(value) == int or type(value) == float or isinstance(value, string_types):
            parser_unk.add_argument("--" + pm, default = value, type = type(value))
        elif type(value) == bool:
            parser_unk.add_argument("--" + pm, default = value, type = ast.literal_eval)
    config_update = parser_unk.parse_args(unknown)
    nested_update(base_config, nest_dict(vars(config_update)))

    return args, base_config, base_model, config_module

def check_base_model_logdir(base_logdir, args, restore_best_checkpoint = False):

    if not base_logdir:
        return ''

    base_logdir = os.path.join(base_logdir, args.detection, args.set)
    
    if (not os.path.isdir(base_logdir)) or len(os.listdir(base_logdir)) == 0:
        raise IOError(" The log directory for the base model is empty or dose not exist.")
    
    if args.enable_logs:
        ckpt_dir = os.path.join(base_logdir, "logs")
        if not os.path.isdir(ckpt_dir):
            raise IOError("There is no folder for 'logs' in the base model logdir.\
                           If checkpoints exist, put them in the logs 'folder'")
    else:
        ckpt_dir = base_logdir

    if restore_best_checkpoint and os.path.isdir(os.path.join(ckpt_dir, 'best_models')):
        ckpt_dir = os.path.join(ckpt_dir, 'best_models')

    checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    if checkpoint is None:
        raise IOError(
                "There is no valid Tensorflow checkpoint in the {} directory. Can't load model.".format(ckpt_dir))

    return ckpt_dir

def check_logdir(args, base_config, restore_best_checkpoint=False):

    logdir = base_config['logdir']
    logdir = os.path.join(logdir, args.detection, args.set)
    checkpoint = None
    try:
        ckpt_dir = logdir
        if args.mode == "train":
            
            if os.path.isfile(logdir):
                raise IOError("There is a file with the same name as logdir")
            
            if os.path.isdir(logdir) and os.listdir(logdir) != []:
                if not args.continue_learning:
                    raise IOError("Log directory is not empty: {}".format(logdir))
                
                checkpoint = tf.train.latest_checkpoint(ckpt_dir)
                if checkpoint is None:
                    raise IOError("There is no valid Tensorflow checkpoint in the {} directory. Can't load model.".format(ckpt_dir))
            else:
                if args.continue_learning:
                    raise IOError("The log directory is empty or does not exist.")
        
        elif args.mode == "eval" or args.mode == "infer" or args.mode == "interactive_infer":
            
            if os.path.isdir(logdir) and os.listdir(logdir) != []:
                best_ckpt_dir = os.path.join(ckpt_dir, 'best_models')
                
                if restore_best_checkpoint and os.path.isdir(best_ckpt_dir):
                    deco_print("Restoring from the best checkpoint")
                    checkpoint = tf.train.latest_checkpoint(best_ckpt_dir)
                    ckpt_dir = best_ckpt_dir
                
                else:
                    deco_print("Restoring from the latest checkpoint")
                    checkpoint = tf.train.latest_checkpoint(ckpt_dir)

                if checkpoint is None:
                    raise IOError(" There is no valid Tensorflow checkpoint in the {} directory. Can't load model".format(ckpt_dir))
            else:
                raise IOError("{} does not exit or is empty, can't restore model.".format(ckpt_dir))
        
        return checkpoint    
    
    except IOError as e:
        raise
    

def deco_print(line, offset = 0, start="*** ", end = "\n"):
    if six.PY2:
        print((start + " " * offset + line).encode("utf-8"), end = end)
    else:
        print(start + " " * offset + line, end = end)

def mark_print(line):
    print("##########", line, "##########")

def clip_last_batch(last_batch, true_size):

    last_batch_clipped = []
    for val in last_batch:
        if isinstance(val, tf.SparseTensorValue):
            last_batch_clipped.append(clip_sparse(val, true_size))
        else:
            last_batch_clipped.append(val[:true_size])
    return last_batch_clipped

def clip_sparse(value, size):
    dense_shape_clipped = value.dense_shape
    dense_shape_clipped[0] = size

    indices_clipped = []
    values_clipped = []

    for idx_tuple, val in zip(value.indices, value.values):
        if idx_tuple[0] < size:
            indices_clipped.append(idx_tuple)
            values_clipped.append(val)

    return tf.SparseTensorValue(np.array(indices_clipped), np.array(values_clipped), dense_shape_clipped)

def check_params(config, required_dict, optional_dict):
    if required_dict is None or optional_dict is None:
        raise ValueError("Need required_dict or optional_dict")

    for pm, vals in required_dict.items():
        if pm not in config:
            raise ValueError("{} parameter has to be specified".format(pm))
        else:
            if vals == str:
                vals = string_types
            if vals and isinstance(vals, list) and config[pm] not in vals:
                raise ValueError("{} has to be one of {}".format(pm, vals))
            if vals and not isinstance(vals, list) and not isinstance(config[pm], vals):
                raise ValueError("{} has to be of type {}".format(pm, values))

    for pm, vals in optional_dict.items():
        if vals == str:
            vals = string_types
        if pm in config:
            if vals and isinstance(vals, list) and config[pm] not in vals:
                raise ValueError("{} has to be one of {}".format(pm, vals))
            if vals and not isinstance(vals, list) and not isinstance(config[pm], vals):
                raise ValueError("{} has to be of type {}".format(pm, values))
    
    for pm in config:
        if pm not in required_dict and pm not in optional_dict:
            raise ValueError("Unknown parameter: {}".format(pm))

def cast_types(input_dict, dtype):
    cast_input_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, tf.Tensor):
            if value.dtype == tf.float16 or value.dtype == tf.float32:
                if value.dtype.base_dtype != dtype.base_dtype:
                    cast_input_dict[key] = tf.cast(value, dtype)
                    continue
        
        if isinstance(value, dict):
            cast_input_dict[key] = cast_types(input_dict[key], dtype)
            continue
        
        if isinstance(value, list):
            cur_list = []
            for nest_value in value:
                if isinstance(nest_value, tf.Tensor):
                    if nest_value.dtype == tf.float16 or nest_value.dtype == tf.float32:
                        if nest_value.dtype != dtype.base_dtype:
                            cur_list.append(tf.cast(nest_value, dtype))
                            continue
                cur_list.append(nest_value)
            cast_input_dict[key] = cur_list
            continue
        cast_input_dict[key] = input_dict[key]
    return cast_input_dict

def mask_nans(x):
    """
    把输入里面的inf nan都转换为0
    """
    x_zeros = tf.zeros_like(x)
    x_mask = tf.is_finite(x)
    y = tf.where(x_mask, x, x_zeros)
    
    return y

def iterate_data(model, sess, compute_loss, mode, verbose, num_steps = None):
    total_time = 0.0
    results_per_batch = []

    size_defined = model.get_data_layer().get_size_in_samples() is not None
    if size_defined:
        dl_sizes = []

    if compute_loss:
        total_loss = 0.0

    total_samples = []
    fetches = []

    for worker_id in range(model.num_gpus):
        cur_fetches = [
                model.get_data_layer(worker_id).input_tensors,
                model.get_output_tensors(worker_id)]

        if compute_loss:
            cur_fetches.append(model.eval_losses[worker_id])
        if size_defined:
            dl_sizes.append(model.get_data_layer(worker_id).get_size_in_samples())
        try:
            total_objects = 0.0
            cur_fetches.append(model.get_num_objects_per_step(worker_id))
        except NotImplementedError:
            total_objects = None
            deco_print("WARNING: Can't compute number of objects per step, since train model does not define get_num_objects_per_step method")

        fetches.append(cur_fetches)
        total_samples.append(0.0)
    
    sess.run([model.get_data_layer(i).iterator.initializer for i in range(model.num_gpus)])
    
    step = 0
    processed_batches = 0
    if verbose:
        ending = ""

    while True:
        fetches_vals = {}
        if size_defined:
            fetches_to_run = {}
            for worker_id in range(model.num_gpus):
                if total_samples[worker_id] < dl_sizes[worker_id]:
                    fetches_to_run[worker_id] = fetches[worker_id]
            fetches_vals = sess.run(fetches_to_run)
        else:
            for worker_id, one_fetch in enumerate(fetches):
                try:
                    fetches_vals[worker_id] = sess.run(one_fetch)
                except tf.errors.OutOfRangeError:
                    continue
        for worker_id, fetches_val in fetches_vals.items():
            if compute_loss:
                inputs, outputs, loss = fetches_val[:3]
            else:
                inputs, outputs = fetches_val[:2]

            if total_objects is not None:
                total_objects += np.sum(fetches_val[-1])

            batch_size = inputs["source_tensors"][0].shape[0]
            total_samples[worker_id] += batch_size

            if size_defined:
                if total_samples[worker_id] > dl_sizes[worker_id]:
                    last_batch_size = dl_sizes[worker_id] % batch_size
                    for key, value in inputs.items():
                        inputs[key] = model.clip_last_batch(value, last_batch_size)
                    outputs = model.clip_last_batch(outputs, last_batch_size)

            processed_batches += 1

            if compute_loss:
                total_loss += loss * batch_size

            if mode == "eval":
                results_per_batch.append(model.evaluate(inputs, outputs))
            elif mode == "infer":
                results_per_batch.append(model.infer(inputs, outputs))
            else:
                raise ValueError("Unknown mode: {}".format(mode))
        
        if verbose:
            if size_defined:
                data_size = int(np.sum(np.ceil(np.array(dl_sizes) / batch_size)))
                if step == 0 or len(fetches_vals) == 0 or (data_size > 10 and processed_batches % (data_size // 10) == 0):
                    deco_print("Processed {}/{} batches.{}".format(processed_batches, data_size, ending))
                else:
                    deco_print("Processed {} batches{}.".format(processed_batches, ending), end = "\r")

        if len(fetches_vals) == 0:
            break
        step += 1

        if num_steps is not None and step > num_steps:
            break
    if compute_loss:
        return results_per_batch, total_loss, np.sum(total_samples)
    else:
        return results_per_batch
        

def collect_if_horovod(value, hvd, mode = "sum"):
    return value

def get_results_for_epoch(model, sess, compute_loss, mode, verbose = False):
    
    if compute_loss:
        results_per_batch, total_loss, total_samples = iterate_data(
                model, sess, compute_loss, mode, verbose)
    else:
        results_per_batch = iterate_data(
                model, sess, compute_loss, mode, verbose)

    if compute_loss:
        total_samples = collect_if_horovod(total_samples, None, "sum")
        total_loss = collect_if_horovod(total_loss, None, "sum")
    results_per_batch = collect_if_horovod(results_per_batch, None, None)

    if results_per_batch is None:
        if compute_loss:
            return None, None
        else:
            return None

    if compute_loss:
        return results_per_batch, total_loss / total_samples
    else:
        return results_per_batch

def log_summaries_from_dict(dict_to_log, output_dir, step):

    sm_writer = tf.summary.FileWriterCache.get(output_dir)
    for tag, value in dict_to_log.items():
        if isinstance(value, tf.Summary.Value):
            sm_writer.add_summary(tf.Summary(value = [value]), global_step - step)
        else:
            sm_writer.add_summary(tf.Summary(value = [tf.Summary.Value(tag = tag, simple_value = value)]), global_step = step)

        sm_writer.flush()

def get_interactive_infer_results(model, sess, model_input):
    fetches = [
            model.get_data_layer().input_tensors,
            model.get_output_tensors()]
    
    feed_dict = model.get_data_layer().create_feed_dict(model_input)

    inputs, outputs = sess.run(fetches, feed_dict = feed_dict)

    return model.infer(inputs, outputs)
