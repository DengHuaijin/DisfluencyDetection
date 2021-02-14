from __future__ import absolute_import, print_function, division
from __future__ import unicode_literals

import math
import os
import time

import tensorflow as tf

from utils.utils import deco_print, log_summaries_from_dict, get_results_for_epoch

class PrintSampleHook(tf.train.SessionRunHook):
    
    def __init__(self, every_steps, model):
        super(PrintSampleHook).__init__()
        self._timer = tf.train.SecondOrStepTimer(every_steps = every_steps)
        self._iter_count = 0
        self._global_step = None
        self._model = model
        
        # Using only the 1st GPU
        output_tensors = model.get_output_tensors(0)
        self._fetches = [model.get_data_layer(0).input_tensors, output_tensors]

    def begin(self):
        self._iter_count = 0
        self._global_step = tf.train.get_global_step()

    def before_run(self, run_context):
        if self._timer.should_trigger_for_step(self._iter_count):
            return tf.train.SessionRunArgs([self._fetches, self._global_step])
        return tf.train.SessionRunArgs([[], self._global_step])

    def after_run(self, run_context, run_values):
        results, step = run_values.results
        self._iter_count = step

        if not results:
            return 
        self._timer.update_last_triggered_step(self._iter_count - 1)

        input_values, output_values = results
        dict_to_log = self._model.maybe_print_logs(input_values, output_values, step)

        if self._model.params["save_summaries_steps"] and dict_to_log:
            log_summaries_from_dict(
                    dict_to_log,
                    self._model.params["logdir"],
                    step)

class PrintLossAndTimeHook(tf.train.SessionRunHook):

    def __init__(self, every_steps, model):
        super(PrintLossAndTimeHook).__init__()
        self._timer = tf.train.SecondOrStepTimer(every_steps = every_steps)
        self._every_steps = every_steps
        self._iter_count = 0
        self._global_step = None
        self._model = model
        self._fetches = [model.loss]
        self._last_time = time.time()

    def begin(self):
        self._iter_count = 0
        self._global_step = tf.train.get_global_step()

    def before_run(self, run_context):
        if self._timer.should_trigger_for_step(self._iter_count):
            return tf.train.SessionRunArgs([self._fetches, self._global_step])
        return tf.train.SessionRunArgs([[], self._global_step])

    def after_run(self, run_context, run_values):
        results, step = run_values.results
        self._iter_count = step

        if not results:
            return 

        self._timer.update_last_triggered_step(self._iter_count - 1)

        if self._model.steps_in_epoch is None:
            deco_print("Global step {}:".format(step), end = " ")
        else:
            deco_print("Epoch {}, global step {}:".format(step // self._model.steps_in_epoch, step), end = " ")

        loss = results[0]

        deco_print("Train loss: {:.4f}".format(loss), offset = 4)

        tm = (time.time() - self._last_time) / self._every_steps
        m, s = divmod(tm, 60)
        h, m = divmod(m ,60)

        deco_print("time per step = {}:{:02}:{:.3f}".format(int(h), int(m), s), start = " ")

        self._last_time = time.time()



