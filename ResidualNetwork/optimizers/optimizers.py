from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import collections
import six
import sys
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from utils.utils import mask_nans, check_params, deco_print

OPTIMIZER_CLS_NAMES = {
        "Adagrad": tf.train.AdagradOptimizer,
        "Adam": tf.train.AdamOptimizer,
        "Ftrl": tf.train.FtrlOptimizer,
        "Momentum": tf.train.MomentumOptimizer,
        "RMSProp": tf.train.RMSPropOptimizer,
        "SGD": tf.train.GradientDescentOptimizer,
        "AdamW": tf.contrib.opt.AdamWOptimizer,
        }

OPTIMIZER_SUMMARIES = [
        "learning_rate",
        "gradients",
        "gradients_norm",
        "gloabl_gradient_norm",
        "variables",
        "variable_norm",
        "larc_summaries",
        "loss_scale"
        ]

def get_regularization_loss(scope = None, name = "total_regularization_loss"):
    
    losses = tf.losses.get_regularization_loss(scope)
    # tf.add_n Adds all tensors element-wise
    if losses is not None:
        return tf.add_n(list(map(lambda x: tf.cast(x, tf.float32), losses)), name = name)
    else:
        return tf.constant(0.0)

def reduce_gradients(grads_and_vars, model = None):
    
    raise NotImplementedError("Reduce in tower-mode is not implemented")

def optimize_loss(loss,
                  optimizer,
                  optimizer_params,
                  learning_rate_decay_fn,
                  var_list = None,
                  dtype = tf.float32,
                  clip_gradients = None,
                  summaries = None,
                  larc_params = None,
                  loss_scaling = 1.0,
                  loss_scaling_params = None,
                  iter_size = 1,
                  skip_update_ph = None,
                  model = None):

    """
    Given loss and parameters for optimizer, returns a training op.
    """

    if summaries is None:
        summaries = ["learning_rate", "global_gradient_norm", "loss_scale"]
    else:
        for sumn in summaries:
            if sumn not in OPTIMIZER_SUMMARIES:
                raise ValueError(
                        "Summaries should be one of [{}], you provided {}.".format(
                            ",".join(OPTIMIZER_SUMMARIES), sumn))

    if clip_gradients is not None and larc_params is not None:
        raise AttributeError(
                "LARC and gradient norm clipping should not be used together")

    global_step = tf.train.get_or_create_global_step()
    lr = learning_rate_decay_fn(global_step)
    
    if "learning_rate" in summaries:
        tf.summary.scalar("learning_rate", lr)

    with tf.variable_scope("LossOptimization"):
        update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        """
        contro_flow_ops.with_dependencies 实现图节点之间的依赖控制
        with_dependencies(dependencies, output_tensor, name = None)

        """
        loss = control_flow_ops.with_dependencies(list(update_ops), loss)

        if optimizer == "AdamW":
            optimizer_params["weight_decay"] = optimizer_params["weight_decay"] * lr

        # Create optimizer, given specified parameters
        if isinstance(optimizer, six.string_types):
            if optimizer not in OPTIMIZER_CLS_NAMES:
                raise ValueError("Optimizer name should be one of [{}], you provided {}".format(", ".join(OPTIMIZER_CLS_NAMES), optimizer))
            optimizer = OPTIMIZER_CLS_NAMES[optimizer]

        opt = optimizer(learning_rate = lr, **optimizer_params)

        if isinstance(loss_scaling, six.string_types):
            loss_scaling = AutomaticLossScaler(algorithm = loss_scaling,
                                               params = loss_scaling_params)
        # if "loss_scale" in summaries:
        #    tf.summary.scalar("loss_scale", loss_scaling.loss_scale)

        #if dtype == "mixed":
        #    opt = MixedPrecisionOptimizerWrapper(opt, loss_scale = loss_scaling)

        """
        Compute gradients
        Inputs:
            var_list: A list or tuple of tf.Variable to update to minimize loss.
                      Defaults to the list of variables collected in the graph 
                      under the key GraphKeys.TRAINABLE_VARIABLES
        Returns:
            A list of (gradients, variable) pairs. Variable is always present but gradient can be None
        """
        grads_and_vars = opt.compute_gradients(
                loss, colocate_gradients_with_ops = True, var_list = var_list)
        # print("#################\n", grads_and_vars, "\n##################\n")

        """
        apply_gradients returns an Operation that applies gradients.
        Inputs
            grads_and_vars: List of (gradients, variable) pairs as returned by compute_gradients()
            global_step: Optional Varibale to increment by one after the variables have been updated
        Returns:
            If global_step was not None, that operation also increments gloabl_step
        """
        grad_updates = opt.apply_gradients(
                post_process_gradients(
                    grads_and_vars,
                    lr = lr,
                    clip_gradients = clip_gradients,
                    larc_params = larc_params,
                    summaries = summaries),
                global_step = global_step)
        
        # ensure the train tensor computes grad_updates
        # print("###########\n {} \n#########\n".format(grad_updates))
        train_tensor = control_flow_ops.with_dependencies([grad_updates], loss)
        # print("###########\n {} \n#########\n".format(train_tensor))
        return train_tensor, grads_and_vars

def post_process_gradients(grads_and_vars, summaries, lr, 
                           clip_gradients,larc_params):

    """ Apply post processing to gradients, i.e. clipping, LARC, summaries"""
    if "global_gradient_norm" in summaries:
        tf.summary.scalar(
                "global_gradient_norm",
                _global_norm_with_cast(grads_and_vars))

    if clip_gradients is not None:
        grads_and_vars = _clip_gradients_by_norm(grads_and_vars, clip_gradients)
    
    # Add histograms for variables, gradients and gradient norms
    for gradient, variable in grads_and_vars:
        if isinstance(gradient, tf.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient

        if isinstance(variable, tf.IndexedSlices):
            var_values = variable.values
        else:
            var_values = variable

        if grad_values is not None:
            var_name = variable.name.replace(":", "_")
            if "gradients" in summaries:
                tf.summary.histogram("gradients%s" % var_name, mask_nans(grad_values))
            if "gradient_norm" in summaries:
                tf.summary.scalar("gradient_norm%s" % var_name, tf.norm(grad_values))
            if "variabels" in summaries:
                tf.summary.histogram("variabels%s" % var_name, var_values)
            if "variable_norm" in summaries:
                tf.summary.scalar("varibale_norm%s" % var_name, tf.norm(var_values))

    if clip_gradients is not None and "global_gradient_norm" in summaries:
        tf.summary.scalar(
                "global_clipped_gradient_norm",
                _global_norm_with_cast(grads_and_vars))
    
    # LARC gradient re-scaling
    if larc_params is not None:
        check_params(
                config = larc_params,
                required_dict = {"larc_eta": float},
                optional_dict = {
                    "larc_mode": ["clip", "scale"],
                    "min_update": float,
                    "epsilon": float
                },
        )

        larc_eta = larc_params["larc_eta"]
        larc_mode = larc_params.get("larc_mode", "clip")
        min_update = larc_params.get("min_update", 1e-7)
        eps = larc_params.get("epsilon", 1e-7)

        grads_and_vars_larc = [None] * len(grads_and_vars)
        for idx, (g,v) in enumerate(grads_and_vars):
            var_dtype = v.dtype
            v_norm = tf.norm(tensor = tf.cast(v, tf.float32), ord = 2)
            g_norm = tf.norm(tensor = tf.cast(g, tf.float32), ord = 2)

            if larc_mode == "clip":
                larc_grad_update = tf.maximum(
                        larc_eta * v_norm / (lr * (g_norm + eps)),
                        min_update)

                if "larc_summaries" in summaries:
                    tf.summary.scalar("larc_clip_on/{}".format(v.name),
                                      tf.cast(tf.less(larc_grad_update, 1.0), tf.int32))

                larc_grad_update = tf.minimum(larc_grad_update, 1.0)
            else:
                larc_grad_update = tf.maximum(
                        larc_eta * v_norm / (g_norm + eps),
                        min_update)

            larc_grad_update = tf.saturate_cast(larc_grad_update, var_dtype)
            grads_and_vars_larc[idx] = (larc_grad_update * g, v)

            if "larc_summaries" in summaries:
                tf.summary.scalar("larc_grad_update/{}".format(v.name), larc_grad_update)
                tf.summary.scalar("larc_final_lr/{}".format(v.name), tf.cast(lr, var_dtype) * larc_grad_update)

        grads_and_vars = grads_and_vars_larc

    return grads_and_vars

def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
    """Clips gradients by global norm"""
    gradients, variables = zip(*grads_and_vars)
    dtypes = [var.dtype for var in variables]

    # Clips gradients in float32
    clipped_gradients, _ = _clip_by_global_norm(
            gradients,
            clip_gradients,
            use_norm = _global_norm_with_cast(grads_and_vars))
    
    # Convert gradients back to proper dtype
    clipped_gradients = [tf.cast(grad, dtype) for grad, dtype in zip(clipped_gradients, dtypes)]

    return list(zip(clipped_gradients, variables))

def _clip_by_global_norm(t_list, clip_norm, use_norm, name = None):
    """
    Clips values of multiple tensors by the ratio of the sum of their norms,
    Input
        t_list: a tuple or list of tensors 
        clip_norm: a clipping norm
    Return:
        list_clipped: a list of clipped tensors
        global_norm: the global norm of all tensors

    To perform the clipping, the values t_list[i] are set to:
        t_list[i] * clip_norm / max(global_norm, clip_norm)
    """
    if (not isinstance(t_list, collections.Sequence) or isinstance(t_list, six.string_types)):
        raise TypeError("t_list should be a sequence")

    t_list = list(t_list)

    with tf.name_scope(name, "clip_by_global_norm", t_list + [clip_norm]) as name:
        # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        scale = clip_norm * tf.minimum(1.0 / use_norm, tf.ones([1], dtype = use_norm.dtype) / clip_norm)

        values = [tf.cast(
            tf.convert_to_tensor(t.values if isinstance(t, tf.IndexedSlices) else t, name="t_%d" % i), 
            dtype = tf.float32)
            if t is not None else t for i,t in enumerate(t_list)]

        values_clipped = []
        for i,v in enumerate(values):
            if v is None:
                values_clipped.append(None)
            else:
                with tf.colocate_with(v):
                    values_clipped.append(tf.identity(v * scale, name = "%s_%d" % (name,i)))

        list_clipped = [
                tf.IndexedSlices(c_v, t.indices, t.dense_shape) if isinstance(t, tf.IndexedSlices) else c_v
                for (c_v), t in zip(values_clipped, t_list)]

    return list_clipped, use_norm

def _global_norm_with_cast(grads_and_vars):
    return tf.global_norm(list(map(
        lambda x: tf.cast(x, tf.float32),
        list(zip(*grads_and_vars))[0])))
