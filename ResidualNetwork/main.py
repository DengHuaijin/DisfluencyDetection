from __future__ import absolute_import, print_function
from __future__ import division
from __future__ import unicode_literals

import os
import sys
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

from utils.funcs import train, evaluate
from utils.utils import get_base_config, check_base_model_logdir, check_logdir, create_model

if __name__ == "__main__":

    args, base_config, base_model, config_module = get_base_config(sys.argv[1:])

    if args.mode not in ["train", "eval", "infer"]:
        raise ValueError("{} mode is not supported, train, eval or infer".format(args.mode))

    load_model = base_config.get("load_model", None)
    restore_best_ckpt = base_config.get("restore_best_ckpt", False)
    base_ckpt_dir = check_base_model_logdir(load_model, args, restore_best_ckpt)

    base_config["load_model"] = base_ckpt_dir
    
    ckpt = check_logdir(args, base_config, restore_best_ckpt)

    if args.mode == "train":
        if ckpt is None:
            if base_ckpt_dir:
                print("Starting from base model")
        else:
            print("Restored ckpt from {}. Resuming training".format(ckpt))

    elif args.mode == "eval":
        print("Loading model friom {}".format(ckpt))


    with tf.Graph().as_default():
        model = create_model(args, base_config, config_module, base_model, ckpt)

        if args.mode == "train":
            train(model, eval_model = None, args = args)
        elif args.mode == "eval":
            evaluate(model, ckpt)
        elif args.mode == "infer":
            inter(model, ckpt)
