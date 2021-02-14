from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import numpy as np
import six
import abc
import copy
import time 

try:
    from inspect import signature
except ImportError:
    from funcsigs import signature

from utils.utils import deco_print, clip_last_batch
from utils.utils import check_params
from optimizers.optimizers import optimize_loss, get_regularization_loss

@six.add_metaclass(abc.ABCMeta)
class Model:
    """
    基类模型中实现了多GPU运算
    """

    """
    staticmethod表示不用将类实例化，可以直接调用
    """
    @staticmethod
    def get_required_params():
        """
        returns:
                Dict:
        必须包含在__init__的params中,
        在这里暂时删除了use_horovod
        """
        return {
                'batch_size_per_gpu':int,
                'data_layer': None,          # 任何继承DataLayer(data.data_layer.DataLayer)的类都可以作为输入
                }
    
    @staticmethod
    def get_optional_params():
        return {
                'logdir': str,
                'num_gpus': int,            # 不能和gpu_ids同时使用
                'gpu_ids': list,
                'load_model': str,
                
                'save_summaries_steps': None,
                'print_loss_steps': None,
                'print_samples_steps': None,
                'save_checkpoint_steps': None,
                'num_checkpoints': int,
                'restore_best_ckpt': bool,
                'eval_steps': int,
                'finetune': bool,
                'eval_batch_size_per_gpu': int,
                
                'random_seed': int,
                'num_epochs': int,          #num_epochs不能和max_steps同时使用
                'max_steps': int,
                
                'data_layer_params': dict,
                'optimizer': None,          #可以是class或者string，'Adam", "RMSProp"...
                'optimizer_params': dict,   #作为optimizer类的__init__输入 
                'freeze_variables_regex': None, #规定一些变量，训练时不被更新
                'initializer': None,        #tf的标准initializer
                'initializer_params': dict,
                'regularizer': None,        #tf的标准regularizer
                'regularizer_params': dict,
                'dtype': [tf.float16, tf.float32, 'mixed'],
                'lr_policy': None,          # 合法的tf 学习率变化函数，比如optimizers.lr_policies
                'lr_policy_params': dict,
                'max_grad_norm': float,     # maximum value of grad norm. 如果一些梯度超过了这个值会被修改
                'loss_scaling': None,       # float, Backoff, LogMax
                'loss_scaling_params': dict,
                'summaries': list,          # "learning_rate", "gradients", "gradients_norm", "global_gradients_norm", "variables"
                                            # "variable_norm", "loss_scale"
                'iter_size': int, 
                'larc_params': dict,        # LARC algorithm
                'processed_data_folder': str,
                }
	
    def __init__(self, params, mode="train", detection = "", dataset = "", num_set = ""):
        
        """
        tf的图不在这里创建，而是在self.compile()中

        params: dict
        
        mode: train - all parts of the graph will be built (model loss optimizer)
              eval - (model loss)
        """
        
        check_params(params, self.get_required_params(), self.get_optional_params())
        
        self._params = copy.deepcopy(params)
        self.detection = detection
        self.dataset = dataset
        self.num_set = num_set
        
        #parameter checks
        self._mode = mode
        self._interactive = False

        if self._mode == "interactive_infer":
            self._mode = "infer"
            self._interactive = True
        
        if self._mode not in ["train", "eval", "infer"]:
            raise ValueError("Mode has to be one of [train, eval, infer]")
        
        if "max_steps" in params and "num_epochs" in params:
            raise ValueError("You can't provide both of max_steps and num_epochs")
                
        if mode == "train":
            if "max_steps" not in params and "num_epochs" not in params:
                    raise ValueError("For the training mode, either max_steps or num_epochs has to be provided")
                        
        none_list = ["print_samples_steps", "print_loss_steps", "save_checkpoint_steps", "save_summaries_steps"]
        for param in none_list:
            if param not in self._params:
                self._params[params] = None
       
        self._params["num_checkpoints"] = self._params.get("num_checkpoints", 5)
        self._params["finetune"] = self._params.get("finetune", False)
        self._params["load_model"] = self._params.get("load_model", None)
        self._params["eval_batch_size_per_gpu"] = self._params.get("eval_batch_size_per_gpu", self._params["batch_size_per_gpu"])
        
        # checking that freq of samples and loss are aligned
        s_fr = self._params["print_samples_steps"]
        l_fr = self._params["print_loss_steps"]
        
        if s_fr is not None and l_fr is not None and s_fr % l_fr != 0:
                raise ValueError("print_sample_steps has to be the multiple of print_loss_steps")
        
        if "gpu_ids" in self._params:
            self._gpu_ids = self._params["gpu_ids"]
        elif "num_gpus" in self._params:
            self._gpu_ids = range(self._params["num_gpus"])
        else:
            raise ValueError("Either gpu_ids or num_gpus has to be specified in the config")
        
        if self._interactive and len(self._gpu_ids) > 1:
            raise ValueError("Interactive infer is meant to be used with 1 gpu")

        # setting random seed
        rs = self._params.get("random_seed", int(time.time()))
        tf.set_random_seed(rs)
        np.random.seed(rs)

        if "data_type" not in self._params:
            self._params["data_type"] = tf.float32
        
        dl_params = self._params.get("data_layer_params", {})
        dl_params["detection"] = self.detection
        dl_params["dataset"] = self.dataset
        dl_params["set"] = self.num_set
       
        """
        data_layer_params里面原来没有定义batch_size
        """
        if mode == "train":
            dl_params["batch_size"] = self._params["batch_size_per_gpu"]
        else:
            dl_params["batch_size"] = self._params["eval_batch_size_per_gpu"]
        
        dl_params["mode"] = self._mode
        dl_params["interactive"] = self._interactive
        dl_params["dtype"] = self._params["dtype"]
        
        self._data_layers = []
        """
        多GPU运算的话，每个GPU对应一个Speech2TextDataLayer
        Speech2TextDataLayer(params, model, num_workers, work_id)
        """
        for worker_id in range(self.num_gpus):
            self._data_layers.append(self._params["data_layer"](
                params = dl_params, model = self,
                num_workers = self.num_gpus, worker_id = worker_id))
        
        if self._mode == "train":
            
            if "max_steps" in self._params:
                slef._last_step = self._params["max_steps"]
                self._steps_in_epoch = None
            else:
                # doing a few steps if data size is not divisible by the batch size
                self._steps_in_epoch = self.get_data_layer().get_size_in_samples() // self.get_data_layer().params["batch_size"]
                
                if self._steps_in_epoch is None:
                    raise ValueError("The data_layer is not compatible with epoch execution")
                
                """ 
                多GPU计算中，在一个epoch中每个GPU各执行一部分steps
                batch_size超过samples时steps_in_epoch为0
                """
                self._steps_in_epoch //= self.num_gpus
                self._steps_in_epoch //= self._params.get("iter_size", 1)
                
                if self._steps_in_epoch == 0:
                    raise ValueError("Overall batch size is too big for this dataset")
                self._last_step = self._params["num_epochs"] * self._steps_in_epoch
                        
        self._outputs = [None] * self.num_gpus
        
        self.loss = None
        self.train_op = None
        self.eval_losses = None
        self._num_objects_per_step = None
        self.skip_update_ph = None
		
    def compile(self, force_var_reuse = False, checkpoint = None):
        """
        Tensorflow graph is built here.
        """
        if "initializer" not in self.params:
            initializer = None
        else:
            init_dict = self.params.get("initializer_params",{})
            initializer = self.params["initializer"](**init_dict)
        
        losses = []
        for gpu_cnt, gpu_id in enumerate(self._gpu_ids):
            """
            如果GPU>=2，启用reuse模式，即多个GPU上的图共用相同名称的变量
            单个GPU的话共用没有意义，所以这里用gpu_cnt>0判断一下
            """
            with tf.device("/gpu:{}".format(gpu_id)), tf.variable_scope(
            name_or_scope = tf.get_variable_scope(), 
            reuse = force_var_reuse or (gpu_cnt > 0),
            
            initializer = initializer,
            dtype = self.get_tf_dtype()):
                    
                deco_print("Building graph on GPU:{}".format(gpu_id))
               
                self.get_data_layer(gpu_cnt).build_graph()

                """
                这个input_tensors是带有@property属性的成员函数input_tensors()
                返回self._input_tensors
                """
                input_tensors = self.get_data_layer(gpu_cnt).input_tensors
               
                """
                _build_forward_pass_graph 在EncoderDecoder中实现

                """
                loss, self._outputs[gpu_cnt] = self._build_forward_pass_graph(input_tensors, gpu_id = gpu_cnt)
                
                if self._outputs[gpu_cnt] is not None and not isinstance(self._outputs[gpu_cnt], list):
                    raise ValueError("Decoder outputs have to be either None or list")
                if self._mode == "train" or self._mode == "eval":
                    losses.append(loss)
    
        # end of for gpu_ind loop
        if self._mode == "train":
            self.loss = tf.reduce_mean(losses)
        if self._mode == "eval":
            self.eval_losses = losses

        try:
            self._num_objects_per_step = [self._get_num_objects_per_step(worker_id) for worker_id in range(self.num_gpus)]
        except NotImplementedError:
            pass

        if self._mode == "train":
            if "lr_policy" not in self.params:
                lr_policy = None
            else:
                lr_params = self.params.get("lr_policy_params", {})

                func_params = signature(self.params["lr_policy"]).parameters
                
                if "decay_steps" in func_params and "decay_steps" not in lr_params:
                    lr_params["decay_steps"] = self._last_step
                    if "begin_decay_at" in func_params:
                        if "warmup_steps" in func_params:
                            lr_params["begin_decay_at"] = max(lr_params.get("begin_decay_at", 0), lr_prams.get("warmup_steps", 0))
                        lr_params["decay_steps"] -= lr_params.get("begin_decay_at", 0)
                
                if "steps_per_epoch" in func_params and "steps_per_epoch" not in lr_params and "num_epochs" in self.params:
                    lr_params["steps_per_epoch"] = self.steps_in_epoch
                
                lr_policy = lambda gs:self.params["lr_policy"](global_step = gs, **lr_params)

            if self.params.get("iter_size", 1) > 1:
                self.skip_update_ph = tf.placeholder(tf.bool)

            var_list = tf.trainable_variables()
            freeze_variables_regex = self.params.get("freeze_variables_regex", None)
            
            if freeze_variables_regex is not None:
                pattern = re.compile(freeze_variables_regex)
                var_list = [var for var in tf.trainable_variables() if not pattern.match(var.name)]

            self.train_op, _ = optimize_loss(
                loss = tf.cast(self.loss, tf.float32),
                dtype = self.params["dtype"],
                optimizer = self.params["optimizer"],
                optimizer_params = self.params["optimizer_params"],
                var_list = var_list,
                clip_gradients = self.params.get("max_grad_norm", None),
                learning_rate_decay_fn = lr_policy,
                summaries = self.params.get("summaries", None),
                larc_params = self.params.get("larc_params", None),
                loss_scaling = self.params.get("loss_scaling", 1.0),
                loss_scaling_params = self.params.get("loss_scaling_params", None),
                iter_size = self.params.get("iter_size", 1),
                skip_update_ph = self.skip_update_ph,
                model = self
                )
            
            tf.summary.scalar(name = "train_loss", tensor = self.loss)
            if self.steps_in_epoch:
                tf.summary.scalar(
                    name = "epoch",
                    tensor = tf.floor(tf.train.get_global_step() / tf.constant(self.steps_in_epoch, dtype = tf.int64))
                    )

            if freeze_variables_regex is not None:
                deco_print("Complete list of variables:")
                for var in tf.trainable_variables():
                    deco_print("{}".format(var.name), offset = 2)
            
            deco_print("Trainable variables:")
            total_params = 0
            unknown_shapes = False
            
            for var in var_list:
                var_params = 1
                deco_print("{}".format(var.name), offset = 2)
                deco_print("shape: {}, {}".format(var.get_shape(), var.dtype), offset = 2)

                if var.get_shape():
                    for dim in var.get_shape():
                        var_params *= dim.value
                    total_params += var_params
                else:
                    unknown_shapes = True
            
            if unknown_shapes:
                deco_print("Encountered unknown variable shape, can't compute total number of parameters")
            else:
                deco_print("Total trainable parameters: {}".format(total_params))

    
    @abc.abstractmethod
    def _build_forward_pass_graph(self, input_tensors, gpu_id = 0):
        pass
                    
    def evaluate(self, input_values, output_values):
        """
        This function is not abstract and does not have to be implemented 
        in derived classes. But if evaluation functionality is required,
        overwriting this function can be a useful way to add it.

        Returns:
        list: all necessary values for evaluation finilization.
        """
        return []

    def finilize_evaluation(self, results_per_batch, training_step = None):
        """
        Args:
        results_per_batch(list)
        training_step(int): current training step

        Returns:
        dict: dictionary with values that need to be logged to Tensorboard.
        """
        return {}

    def infer(self, input_values, output_values):

        return []

    def finilize_inference(self, results_per_batch, output_file):

        pass

    def clip_last_batch(self, last_batch, true_size):

        return clip_last_batch(last_batch, true_size)

    def get_output_tensors(self, worker_id = 0):
        
        return self._outputs[worker_id]

    def get_data_layer(self, worker_id = 0):
        
        return self._data_layers[worker_id]
            
    def get_tf_dtype(self):
        
        if self.params["dtype"] == "mixed":
            return tf.float16
        else:
            return self.params["dtype"]

    def _get_num_objects_per_step(self, worker_id = 0):
        
        raise NotImplementedError()

    def get_num_objects_per_step(self, worker_id = 0):

        if self._num_objects_per_step:
            return self._num_objects_per_step[worker_id]
        else:
            raise NotImplementedError

    def maybe_print_logs(self, input_values, output_values, training_step):

        return {}

    @property
    def params(self):

        return self._params
            
    @property
    def steps_in_epoch(self):
        
        return self._steps_in_epoch

    @property
    def last_step(self):

        return self._last_step

    @property
    def num_gpus(self):
        
        return len(self._gpu_ids)

    @property
    def mode(self):
        return self._mode
