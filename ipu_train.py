from typing import Union

from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import get_parameter_names

from poptorch import DataLoaderMode, PoplarExecutor

def get_optimizer_scheduler(model, ipu_config, args, ipu_train_dataloader):
    '''
        Get optimizer and scheduler
    '''
    
    #Count number of training step
    train_dataset_is_sized = True#isinstance(train_dataset, collections.abc.Sized)
    total_train_batch_size = args.per_device_train_batch_size * ipu_config.batch_size_factor()
    if train_dataset_is_sized:
        # No need to divide by the number of gradient accumulation steps as poptorch already accounts for that.
        # num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_update_steps_per_epoch = len(ipu_train_dataloader)
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                args.max_steps % num_update_steps_per_epoch > 0
            )

            # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
            # the best we can do.
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = len(ipu_train_dataloader) * args.num_train_epochs
    else:
        # see __init__. max_steps is set when the dataset has no __len__
        max_steps = args.max_steps
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_train_epochs = sys.maxsize
        num_update_steps_per_epoch = max_steps
        num_train_samples = args.max_steps * total_train_batch_size



    #Get optimizer

    #if optimizer is None:
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    if args.lamb or args.lamb_no_bias_correction:
        optimizer_cls = LAMB
        optimizer_kwargs = {
            "max_weight_norm": None,
            "bias_correction": not args.lamb_no_bias_correction,
            "eps": args.adam_epsilon,
        }
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            # TODO: disabled max_grad_norm because it make things fail, fix it.
            #  "max_grad_norm": self.args.max_grad_norm,
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
            "bias_correction": False,
        }

    first_order_type = torch.float16 if ipu_config.enable_half_first_order_momentum else torch.float32
    optimizer_kwargs["lr"] = args.learning_rate
    optimizer_kwargs["loss_scaling"] = args.loss_scaling
    optimizer_kwargs["accum_type"] = first_order_type
    optimizer_kwargs["first_order_momentum_accum_type"] = first_order_type
    optimizer_kwargs["second_order_momentum_accum_type"] = torch.float32

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    if args.lamb or args.lamb_no_bias_correction:
        optimizer.variable_attrs.markAsConstant("max_weight_norm")

    optimizer.variable_attrs.markAsConstant("weight_decay")
    
    
    #Get scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.get_warmup_steps(max_steps), #num_training_steps
        num_training_steps=max_steps,
    )
    optimizer._step_count = 1
    
    
    return optimizer, lr_scheduler


def wrap_model(model: Union[PreTrainedModel, PoplarExecutor], opts, training=True) -> PoplarExecutor:
        """
        Wraps a model for poptorch, either for training or for inference.
        Args:
            model (`~transformers.modeling_utils.PreTrainedModel` or `PoplarExecutor`): the model to wrap
            training (`bool`, *optional*, defaults to `True`): whether to wrap the model for training or not.
        Returns:
            The wrapped model.
        """
        wrapped = None
        if isinstance(model, PoplarExecutor):
            wrapped = model
        elif training:
            training_model = poptorch.trainingModel(
                model.train(), options=self.opts, optimizer=self.optimizer
            )
            wrapped = training_model
        else:
            inference_model = poptorch.inferenceModel(model.eval(), options=self.eval_opts)
            wrapped = inference_model

        # Attaching to device when the model that is being access was already compiled but detached from previous loop.
        if wrapped.isCompiled() and not wrapped.isAttachedToDevice():
            wrapped.attachToDevice()
        return wrapped
