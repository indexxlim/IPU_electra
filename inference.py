import yaml
from easydict import EasyDict

import transformers
import poptorch
from finetune.run_squad_ipu import ipu_options
from pipeline_electra import PipelinedElectraForQuestionAnswering


config_file = 'finetune/squad_configurations.yaml'
config = EasyDict(yaml.load(open(config_file).read(), Loader=yaml.Loader))

valid_replication_factor = config.valid_config.valid_replication_factor
valid_device_iterations = config.valid_config.valid_device_iterations

val_opts = ipu_options(1, valid_replication_factor, valid_device_iterations, train_option=False)

valid_ipu_config = {"layers_per_ipu": config.valid_config.valid_layer_per_ipu,
            "recompute_checkpoint_every_layer": config.valid_config.valid_recompute_checkpoint_every_layer,
            "embedding_serialization_factor": config.valid_config.valid_embedding_serialization_factor}

valid_ipu_config = EasyDict(valid_ipu_config)
model_config = transformers.ElectraConfig.from_pretrained(config.train_config.model_name_or_path)

model = PipelinedElectraForQuestionAnswering.from_pretrained_transformers(config.train_config.saved_model_name, valid_ipu_config, config=model_config)
inference_model = poptorch.inferenceModel(model, val_opts)
