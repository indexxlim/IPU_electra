import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, PreTrainedModel
import poptorch

from modeling_electra import ElectraModel, \
                             ElectraForMaskedLM, \
                             ElectraForPreTraining, \
                             ElectraForQuestionAnswering, \
                             ElectraForTokenClassification
from ipu_configuration import IPUConfig


logger = logging.getLogger(__name__)

_PRETRAINED_TO_PIPELINED_REGISTRY={}

# return_dict to False
# model wrapper because IPU does not support dictationay as input
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.wrapped = model#transformers.BertForQuestionAnswering.from_pretrained(pretrained_weights)

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.wrapped.forward(input_ids,
                                    token_type_ids,
                                    return_dict=False)

    def __getattr__(self, attr):
        try:
            return torch.nn.Module.__getattr__(self, attr)
        except AttributeError:
            return getattr(self.wrapped, attr)

def register(transformers_cls=None):
    def wrapper(cls):
        orig_cls = transformers_cls
        if orig_cls is None:
            found = False
            for base_cls in cls.__bases__:
                if base_cls != PipelineMixin:
                    orig_cls = base_cls
                    found = True
                    break
            if not found:
                raise ValueError(f"Was not able to find original transformers class for {cls}")
        _PRETRAINED_TO_PIPELINED_REGISTRY[orig_cls] = cls
        return cls

    return wrapper

class OnehotGather(nn.Module):
    """
    Gathers selected indices from a tensor by transforming the list of indices
    into a one-hot matrix and then multiplying the tensor by that matrix.
    """

    def __init__(self):
        super().__init__()
        self._is_half = False

    def half(self):
        super().half()
        # Tracing is always executed in float as there are missing
        # implementations of operations in half on the CPU.
        # So we cannot query the inputs to know if we are running
        # with a model that has had .half() called on it.
        # To work around it nn.Module::half is overridden
        self._is_half = True

    def forward(self, sequence, positions):
        """
        Gather the vectors at the specific positions over a batch.
        """
        num_classes = int(sequence.shape[1])
        one_hot_positions = F.one_hot(positions, num_classes)
        if self._is_half:
            one_hot_positions = one_hot_positions.half()
        else:
            one_hot_positions = one_hot_positions.float()
        return torch.matmul(one_hot_positions.detach(), sequence)

def recomputation_checkpoint(module: nn.Module):
    """Annotates the output of a module to be checkpointed instead of
    recomputed"""

    def recompute_outputs(module, inputs, outputs):
        return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)

    return module.register_forward_hook(recompute_outputs)


def get_layer_ipu(layers_per_ipu):
    # List of the IPU Id for each encoder layer
    layer_ipu = []
    for ipu, n_layers in enumerate(layers_per_ipu):
        layer_ipu += [ipu] * n_layers
    return layer_ipu

class PipelineMixin:
    @classmethod
    def from_transformers(cls, model: PreTrainedModel, ipu_config: IPUConfig):
        config = copy.deepcopy(model.config)
        pipelined_model = cls(config)
        pipelined_model.load_state_dict(model.state_dict())
        pipelined_model.ipu_config = copy.deepcopy(ipu_config)
        return pipelined_model

    @classmethod
    def from_pretrained_transformers(cls, model_name_or_path: str, ipu_config: IPUConfig, *model_args, **kwargs):
        # config = AutoConfig.from_pretrained(model_name_or_path)
        pipelined_model = cls.from_pretrained(model_name_or_path, *model_args, **kwargs)  # config=config)
        pipelined_model.ipu_config = copy.deepcopy(ipu_config)
        return pipelined_model

    @classmethod
    def from_model(cls, model: nn.Module):
        clone = copy.deepcopy(model)
        clone.__class__ = cls
        # Just needed so that .parallelize() does not throw an error
        clone.ipu_config = IPUConfig()
        return clone

    def _has_ipu_config_check(self):
        _ipu_config = getattr(self, "_ipu_config", None)
        if _ipu_config is None:
            raise AttributeError("No IPUConfig was found, please set the ipu_config attribute")

    @property
    def ipu_config(self):
        self._has_ipu_config_check()
        return self._ipu_config

    @ipu_config.setter
    def ipu_config(self, value):#: IPUConfig):
        # if not isinstance(value, IPUConfig):
        #     raise TypeError(f"ipu_config must be an instance of IPUConfig, but {type(value)} was provided")
        self._ipu_config = value

    def parallelize(self):
        """Transform the model to run in an IPU pipeline."""
        self._hooks = []
        self._has_ipu_config_check()
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is fully compatible with the
        original model.
        """
        # Remove hooks
        if hasattr(self, "_hooks"):
            for h in self._hooks:
                h.remove()
        # Remove poptorch Blocks
        for m in self.modules():
            if m is not self:
                poptorch.removeBlocks(m)
        return self

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.
        Args:
            only_trainable (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return only the number of trainable parameters
            exclude_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return only the number of non-embeddings parameters
        Returns:
            :obj:`int`: The number of parameters.
        """

        # TODO: actually overwrite this to handle SerializedEmbedding.
        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

class SerializedLinear(nn.Linear):
    """
    Exactly equivalent to `nn.Linear` layer, but with the matrix multiplication replaced with
    a serialized matrix multiplication: `poptorch.serializedMatMul`.
    The matrix multiplication is split into separate smaller multiplications, calculated one after the other,
    to reduce the memory requirements of the multiplication and its gradient calculation.
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        factor: Number of serialized multiplications. Must be a factor of
            the dimension to serialize on.
        bias: If set to False, the layer will not learn an additive bias.
            Default: True
        mode: Which dimension of the matmul to serialize on:
            for matrix A (m by n) multiplied by matrix B (n by p).
            * InputChannels: Split across the input channels (dimension m).
            * ReducingDim: Split across the reducing dimension (n).
            * OutputChannels: Split across the output channels (dimension p).
            * Disabled: Same as an ordinary matrix multiplication.
    """

    def __init__(
        self, in_features, out_features, factor, bias=False, mode=poptorch.MatMulSerializationMode.OutputChannels
    ):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.factor = factor

    def forward(self, x):
        if not self.training:
            output = super().forward(x)
        else:
            output = poptorch.serializedMatMul(x, self.weight.t(), self.mode, self.factor)
            if self.bias is not None:
                output += self.bias
        return output

class SerializedEmbedding(nn.Module):
    """
    Wrapper for `nn.Embedding` layer that performs the embedding look-up into
    smaller serialized steps in order to reduce memory in the embedding gradient
    calculation.
    Args:
        embedding: A `nn.Embedding` to wrap
        serialization_factor: The number of serialized embedding look-ups
    """

    def __init__(self, embedding: nn.Embedding, serialization_factor: int):
        super().__init__()
        self.serialization_factor = serialization_factor
        self.num_embeddings = embedding.num_embeddings

        # Num embeddings should be divisible by the serialization factor
        assert self.num_embeddings % self.serialization_factor == 0
        self.split_size = self.num_embeddings // self.serialization_factor
        self.split_embeddings = nn.ModuleList(
            [
                nn.Embedding.from_pretrained(
                    embedding.weight[i * self.split_size : (i + 1) * self.split_size, :].detach(),
                    freeze=False,
                    padding_idx=embedding.padding_idx if i == 0 else None,
                )
                for i in range(self.serialization_factor)
            ]
        )

    def deserialize(self):
        """
        Deserialize the internal wrapped embedding layer and return it as a
        `nn.Embedding` object.
        Returns:
            `nn.Embedding` layer
        """
        return nn.Embedding.from_pretrained(torch.vstack([l.weight for l in self.split_embeddings]), padding_idx=0)

    def forward(self, indices):
        # iterate through the splits
        x_sum = None
        for i in range(self.serialization_factor):
            # mask out the indices not in this split
            split_indices = indices - i * self.split_size
            mask = (split_indices >= 0) * (split_indices < self.split_size)
            mask = mask.detach()
            split_indices *= mask

            # do the embedding lookup
            x = self.split_embeddings[i](split_indices)

            # multiply the output by mask
            x *= mask.unsqueeze(-1)

            # add to partial
            if x_sum is not None:
                x_sum += x
            else:
                x_sum = x
        return x_sum

def outline_attribute(module: nn.Module, value: str):
    """Adds an attribute to a module. This attribute will be used
    when comparing operation equivalence in outlining. For example:
    layer1 = nn.Linear(...)
    layer2 = nn.Linear(...)
    layer3 = nn.Linear(...)
    layer4 = nn.Linear(...)
    outline_attribute(layer1, "A")
    outline_attribute(layer2, "A")
    outline_attribute(layer3, "B")
    The code for layer1 can be reused for layer2.
    But it can't be used for layer3 or layer4.
    """
    context = poptorch.Attribute(__outline={"layer": value})

    def enable(*args):
        context.__enter__()

    def disable(*args):
        context.__exit__(None, None, None)

    handles = []
    handles.append(module.register_forward_pre_hook(enable))
    handles.append(module.register_forward_hook(disable))
    return handles

class ElectraPipelineMixin(PipelineMixin):
    def parallelize(self):
        """
        Transform the Electra model body to run in an IPU pipeline
        - Adds pipeline stages to the model with BeginBlock)
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Adds recomputation checkpoints with recomputation_checkpoint)
        """
        super().parallelize()
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)

        print("--- Device Allocation ---")
        print("Embedding --> IPU 0")
        if self.ipu_config.embedding_serialization_factor > 1:
            self.electra.embeddings.word_embeddings = SerializedEmbedding(
                self.electra.embeddings.word_embedding, self.ipu_config.embedding_serialization_factor
            )
        self.electra.embeddings = poptorch.BeginBlock(self.electra.embeddings, "Embedding", ipu_id=0)
        hs = outline_attribute(self.electra.embeddings.LayerNorm, "embedding")
        self._hooks.extend(hs)
        for index, layer in enumerate(self.electra.encoder.layer):
            ipu = layer_ipu[index]
            #self.ipu_config.recompute_checkpoint_every_layer is always True 
            if index != self.config.num_hidden_layers - 1: 
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)

            self.electra.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            print(f"Encoder {index:<2} --> IPU {ipu}")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        """
        super().deparallelize()

        if self.ipu_config.embedding_serialization_factor > 1:
            self.electra.embeddings.word_embeddings = self.electra.embeddings.word_embeddings.deserialize()
        return self


@register(ElectraModel)
class PipelinedElectraModel(ElectraModel, PipelineMixin):
    def parallelize(self):
        super().parallelize()
        last_ipu = len(self.ipu_config.layers_per_ipu) - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=False,
        )

@register(ElectraForPreTraining)
class PipelinedElectraForPreTraining(ElectraForPreTraining, PipelineMixin):
    """
    ElectraForPretraining transformed to run in an IPU pipeline referring to huggingface/optimum-graphcore..

    Recommanded usage:
    ```
    model = PipelinedElectraForPreTraining.from_transformers(model, ipu_config)
    model.parallelize().half().train()
    ```
    
    """
    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
        """
        super().parallelize()
        #for 

'''
Inside model layer
electra = ElectraModel(config)
generator_predictions = ElectraGeneratorPredictions
generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)
'''
@register(ElectraForMaskedLM)
class PipelinedElectraForMaskedLM(ElectraForMaskedLM, ElectraPipelineMixin):
    """
    This pipelined model has two head layer are generator_predictions and generator_lm_head.
    Last ipu should leave space

    Recommended usage:
    ```
    model = PipelinedElectraForMaskedLM.from_transformers(model, ipu_config)
    ```
    model = PipelinedElectraForMaskedLM.from_pretrained_transformers(config.train_config.model_name_or_path, train_ipu_config, config=model_config)
    ```
    model.parallelize().half().train()
    """

    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    def parallelize(self):
        '''
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
        - Adds recomputation checkpoints
        '''
        super().parallelize()
        last_ipu = len(self.ipu_config.layers_per_ipu) - 1
        print(f"Generator Predictions --> IPU {last_ipu}")
        self.generator_predictions = poptorch.BeginBlock(self.generator_predictions, "Generator Predictions", ipu_id=last_ipu)
        print(f"Generator LM Head --> IPU {last_ipu}")
        self.generator_lm_head = poptorch.BeginBlock(self.generator_lm_head, "generator LM Head", ipu_id=last_ipu)
        return self
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        if self.training:
            generator_hidden_states = self.electra(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            generator_sequence_output = generator_hidden_states[0]

            prediction_scores = self.generator_predictions(generator_sequence_output)
            prediction_scores = self.generator_lm_head(prediction_scores)

            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)).float()

            return (masked_lm_loss,)
        else:
            return super().forward(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, 
                labels=labels, 
                return_dict=False 
            )


# Subclass the HuggingFace ElectraForQuestionAnswering model
class PipelinedElectraForQuestionAnswering(ElectraForQuestionAnswering, ElectraPipelineMixin):
    '''
    Pipeling ElectraForQuestionAnswering
    Recommanded usage
    
    ```
    model = PipelinedElectraForQuestionAnswering.from_transformers(gpu_model, ipu_config)
    ```
    model = PipelinedElectraForQuestionAnswering.from_pretrained_transformers(config.train_config.model_name_or_path, train_ipu_config, config=model_config)
    ```
    model.parallelize().half().train()
    '''

    def parallelize(self):
        super().parallelize()
        last_ipu = len(self.ipu_config.layers_per_ipu) - 1
        print(f"QA Outputs --> IPU {last_ipu}")
        self.qa_outputs = poptorch.BeginBlock(self.qa_outputs, "QA Outputs", ipu_id=last_ipu)
        return self

    # Model training loop is entirely running on IPU so we add Loss computation here
    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "start_positions": start_positions,
            "end_positions": end_positions
        }
        output = super().forward(**inputs)
        if self.training:
            final_loss = poptorch.identity_loss(output.loss, reduction="none")
            return final_loss, output.start_logits, output.end_logits
        else:
            return output.start_logits, output.end_logits

# For Named Entity Recognition
class PipelinedElectraForTokenClassification(ElectraForTokenClassification, ElectraPipelineMixin):
    '''
    Pipeling ElectraForTokenClassification (for the NER)
    Recommanded usage
    ```
    model = PipelinedElectraForTokenClassification.from_transformers(gpu_model, ipu_config)
    ```
    model = PipelinedElectraForTokenClassification.from_pretrained_transformers(model_name_or_path, train_ipu_config, config=model_config)
    ```
    model.parallelize().half().train()
    '''

    def parallelize(self):
        super().parallelize()
        last_ipu = len(self.ipu_config.layers_per_ipu) - 1
        print(f"classifier layer --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "classifier", ipu_ids=last_ipu)
        return self

    def forward(self, input_ids, token_type_ids, attention_mask):
        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }
        
        output = super().forward(**inputs)
        if self.training:
            final_loss = poptorch.identity_loss(output.loss, reduction="none")
            return final_loss, output.logits
        else:
            return output.logits
    