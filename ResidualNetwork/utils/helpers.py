from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import re
import time

import tensorflow as tf

FP32_TEST = re.compile(r'Loss_Optimization\/FP32-master-copy\/')

def get_assign_ops_and_restore_dict(filename, restore_all = False):
    """
    Helper function to read variable checkpoints from filename.
    """
    def check_name_and_shape(name, var, shape_map):
        if name in shape_map:
            if str(var.shape) == "<unknown>":
                return True
            if var.shape == shape_map[name]:
                return True
        return False

    assign_ops = []
    restore_dict = {}

    try:
        # read checkpoints 
        reader = tf.train.NewCheckpointReader(filename)
        var_to_shape_map = reader.get_variable_to_shape_map()

        variables = tf.trainable_variables()
        if restore_all:
            variables = tf.get_collection(tf.GraphKeys.VARIABLES)
        for var in variables:
            idx = var.name.find(":")
            if idx != -1:
                true_name = var.name[:idx]
            loss_idx = re.search("Loss_Optimization", true_name)
            
            if check_name_and_shape(true_name, var, var_to_shape_map):
                tensor = reader.get_tensor(true_name)
                if tensor.dtype != var.dtype.as_numpy_dtype():
                    assign_ops.append(var.assign(tf.cast(tensor, var.dtype)))
                else:
                    restore_dict[true_name] = var
            elif loss_idx:
                loss_idx = loss_idx.end()
                if FP32_TEST.search(true_name):
                    true_name = FP32_TEST.sub("", true_name)
                else:
                    true_name = (true_name[:loss_idx] 
                                 + "/Loss_Optimization/FP32-master-copy"
                                 + true_name[loss_idx:])
                if check_name_and_shape(true_name, var, var_to_shape_map):
                    tensor = reader.get_tensor(true_name)
                    if tensor.dtype != var.dtype.as_numpy_dtype():
                        assign_ops.append(var.assign(tf.cast(tensor, var.dtype)))
                    else:
                        restore_dict[true_name] = var
            else:
                print("not restoring {}".format(var.name))
                if true_name not in var_to_shape_map:
                    print("true name [{}]  was not in shape map".format(true_name))
                else:
                    if var.shape != var_to_shape_map[true_name]:
                        print("var.shape [{}] does not match var_to_shape_map[true_name] [{}]".format(var.shape, var_to_shape_map[true_name]))

                print("Run will mostly error due to this")
    
    except Exception as e:
        print(str(e))
        raise ValueError("Error in loading checkpoint")

    return assign_ops, restore_dict

def run_assign_and_saver(sess, filename, assign_ops, restore_dict):
    """
    Helper function to restore variables. All vars with the same dtype
    can be restored using tf.train.Saver(). All vars with different dtype 
    are restored using assign_ops
    """
    if restore_dict:
        restorer = tf.train.Saver(restore_dict)
        restorer.restore(sess, filename)
    if assign_ops:
        sess.run(assign_ops)
