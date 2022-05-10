from pathlib import Path
import time
import yaml

import torch
import torch.nn as nn
import numpy as np
import transformers
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm, trange
from easydict import EasyDict

import popart
import poptorch

from finetune.squad_preprocessing import prepare_train_features, \
                                prepare_validation_features, \
                                tokenizer, PadCollate, postprocess_qa_predictions

from pipeline_electra import PipelinedElectraForQuestionAnswering

'''
### Get model
'''
def ipu_options(gradient_accumulation, replication_factor, device_iterations, train_option, seed=42):
    opts = poptorch.Options()
    opts.randomSeed(seed)
    opts.deviceIterations(device_iterations)
    
    # Use Pipelined Execution
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

    # Use Stochastic Rounding
    opts.Precision.enableStochasticRounding(train_option)
    
    # Half precision partials for matmuls and convolutions
    opts.Precision.setPartialsType(torch.float16)

    opts.replicationFactor(replication_factor)
    
    opts.Training.gradientAccumulation(gradient_accumulation)
    
    # Return all results from IPU to host
    opts.outputMode(poptorch.OutputMode.All)
    
    # Cache compiled executable to disk
    opts.enableExecutableCaching("./exe_cache")

    if train_option:
        # On-chip Replicated Tensor Sharding of Optimizer State
        opts.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings()
            # Optimizer state lives on IPU
            .useOnChipStorage(True)
            # Optimizer state sharded between replicas with zero-redundancy
            .useReplicatedTensorSharding(True))

        # Available Transcient Memory For matmuls and convolutions operations
        opts.setAvailableMemoryProportion({f"IPU{i}": mp
                                        for i, mp in enumerate([0.08,0.28,0.32,0.32,0.36,0.38,0.4,0.1])})
        
        ## Advanced performance options ##

        # Only stream needed tensors back to host
        opts._Popart.set("disableGradAccumulationTensorStreams", True)
        # Copy inputs and outputs as they are needed
        opts._Popart.set("subgraphCopyingStrategy", int(popart.SubgraphCopyingStrategy.JustInTime))
        # Parallelize optimizer step update
        opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                        int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
        opts._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])
        # Limit number of sub-graphs that are outlined (to preserve memory)
        opts._Popart.set("outlineThreshold", 10.0)
        # Only attach to IPUs after compilation has completed.
        opts.connectionType(poptorch.ConnectionType.OnDemand)
    return opts

def ipu_validataion_options(replication_factor, device_iterations):
    opts = poptorch.Options()
    opts.randomSeed(42)
    opts.deviceIterations(device_iterations)

    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement)
    )

    opts.Precision.enableStochasticRounding(False)

def get_optimizer(model):
    # Do not apply weight_decay for one-dimensional parameters
    regularized_params = []
    non_regularized_params = []
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 1:
                non_regularized_params.append(param)
            else:
                regularized_params.append(param)

    params = [
        {"params": regularized_params, "weight_decay": 0.01},
        {"params": non_regularized_params, "weight_decay": 0}
    ]
    optimizer = poptorch.optim.AdamW(params,
                                     lr=1e-4,
                                     weight_decay=0,
                                     eps=1e-6,
                                     bias_correction=True,
                                     loss_scaling=64,
                                     first_order_momentum_accum_type=torch.float16,
                                     accum_type=torch.float16)
    return optimizer    

def train(model, opts, optimizer, train_dl, num_epochs, samples_per_iteration):
    num_steps = num_epochs * len(train_dl)
    lr_scheduler = transformers.get_scheduler("cosine", optimizer, 0.1 * num_steps, num_steps)
    
    # Wrap the pytorch model with poptorch.trainingModel
    training_model = poptorch.trainingModel(model, opts, optimizer)
    
    # Compile model or load from executable cache
    batch = next(iter(train_dl))
    outputs = training_model.compile(batch["input_ids"],
                                     batch["attention_mask"],
                                     batch["token_type_ids"],
                                     batch["start_positions"],
                                     batch["end_positions"])
    # Training Loop
    for epoch in trange(num_epochs, desc="Epochs"):
        train_iter = tqdm(train_dl)
        for step, batch in enumerate(train_iter):
            start_step = time.perf_counter()
            
            # This completes a forward+backward+weight update step
            outputs = training_model(batch["input_ids"],
                                     batch["attention_mask"],
                                     batch["token_type_ids"],
                                     batch["start_positions"],
                                     batch["end_positions"])

            # Update the LR and update the poptorch optimizer
            lr_scheduler.step()
            training_model.setOptimizer(optimizer)
            step_length = time.perf_counter() - start_step
            step_throughput = samples_per_iteration / step_length
            loss = outputs[0].mean().item()
            train_iter.set_description(
                f"Epoch: {epoch} - "
                f"Step: {step} - "
                f"Loss: {loss:3.3f} - "
                f"Throughput: {step_throughput:3.3f} seq/s")
    
    # Detach the model from the device once training is over so the device is free to be reused for validation
    training_model.detachFromDevice()

def valid(model, opts, val_dl, samples_per_iteration):
    inference_model = poptorch.inferenceModel(model, opts)

    raw_predictions = [[], []]
    val_iter = tqdm(val_dl, desc="validation")
    for step, batch in enumerate(val_iter):
        start_step = time.perf_counter()
        outputs = inference_model(batch["input_ids"],
                                  batch["attention_mask"],
                                  batch["token_type_ids"])
        step_length = time.perf_counter() - start_step
        step_throughput = samples_per_iteration / step_length
        raw_predictions[0].append(outputs[0])
        raw_predictions[1].append(outputs[1])
        val_iter.set_description(
            f"Step: {step} - throughput: {step_throughput:3.3f} samples/s")
    inference_model.detachFromDevice()

    raw_predictions[0] = torch.vstack(raw_predictions[0]).float().numpy()
    raw_predictions[1] = torch.vstack(raw_predictions[1]).float().numpy()
    return raw_predictions

def main():
    config_file = 'finetune/squad_configurations.yaml'
    config = EasyDict(yaml.load(open(config_file).read(), Loader=yaml.Loader))

    datasets = load_dataset("squad_kor_v1", cache_dir=Path.home() / ".torch./dataset")

    train_dataset = datasets["train"].map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=datasets["train"].column_names,
        load_from_cache_file=True,
    )

    # Create validation features from dataset
    validation_features = datasets["validation"].map(
        prepare_validation_features,
        batched=True,
        num_proc=1,
        remove_columns=datasets["validation"].column_names,
        load_from_cache_file=True,
    )

    # Electra-base configuration
    # model_config = transformers.ElectraConfig(embedding_size=768,
    #                                     hidden_size=768,
    #                                     intermediate_size = 1024*3,
    #                                     num_hidden_layers=12,
    #                                     num_attention_heads=12,
    #                                     hidden_dropout_prob=0.1,
    #                                     attention_probs_dropout_prob=0.1,
    #                                     layer_norm_eps=1e-12)
    model_config = transformers.ElectraConfig.from_pretrained(config.train_config.model_name_or_path)
    train_ipu_config = {
        "layers_per_ipu": config.train_config.train_layers_per_ipu,
        "recompute_checkpoint_every_layer":config.train_config.train_recompute_checkpoint_every_layer,
        "embedding_serialization_factor": config.train_config.train_embedding_serialization_factor
    }

    train_ipu_config = EasyDict(train_ipu_config)
    model = PipelinedElectraForQuestionAnswering.from_pretrained_transformers(config.train_config.model_name_or_path, train_ipu_config, config=model_config)

    model.parallelize().half().train()


    # Runtime Configuration
    '''
    set replication_factor to 1
    it's mean the model using 8 IPU setted beforehand
    '''
    train_global_batch_size = config.train_config.train_global_batch_size
    train_micro_batch_size = config.train_config.train_micro_batch_size
    train_replication_factor = config.train_config.train_replication_factor
    gradient_accumulation = int(train_global_batch_size / train_micro_batch_size / train_replication_factor)
    train_device_iterations = config.train_config.train_device_iterations
    train_samples_per_iteration = train_global_batch_size * train_device_iterations
    num_epochs = config.train_config.num_epochs

    train_opts = ipu_options(gradient_accumulation, train_replication_factor, train_device_iterations, train_option=True)

    # Training
    sequence_length = config.train_config.sequence_length
    train_dl = poptorch.DataLoader(train_opts,
                               train_dataset,
                               batch_size=train_micro_batch_size,
                               shuffle=True,
                               drop_last=False,
                               collate_fn=PadCollate(train_global_batch_size,
                                                     {"input_ids": 0,
                                                      "attention_mask": 0,
                                                      "token_type_ids": 0,
                                                      "start_positions": sequence_length,
                                                      "end_positions": sequence_length}))

    optimizer = get_optimizer(model)
    train(model, train_opts, optimizer, train_dl, num_epochs, train_samples_per_iteration)

    model.save_pretrained(config.train_config.saved_model_name)

    #Validation
    valid_micro_batch_size = config.valid_config.valid_micro_batch_size
    valid_replication_factor = config.valid_config.valid_replication_factor
    valid_global_batch_size = valid_micro_batch_size * valid_replication_factor
    valid_device_iterations = config.valid_config.valid_device_iterations
    valid_samples_per_iteration = valid_global_batch_size * valid_device_iterations

    val_opts = ipu_options(1, valid_replication_factor, valid_device_iterations, train_option=False)

    valid_ipu_config = {"layers_per_ipu": config.valid_config.valid_layer_per_ipu,
              "recompute_checkpoint_every_layer": config.valid_config.valid_recompute_checkpoint_every_layer,
              "embedding_serialization_factor": config.valid_config.valid_embedding_serialization_factor}

    valid_ipu_config = EasyDict(valid_ipu_config)

    model = PipelinedElectraForQuestionAnswering.from_pretrained_transformers(config.train_config.saved_model_name, valid_ipu_config, config=model_config)
    model.parallelize().half().eval()

    val_dl = poptorch.DataLoader(val_opts,
                             validation_features.remove_columns(
                                 ['example_id', 'offset_mapping']),
                             batch_size=valid_micro_batch_size,
                             shuffle=False,
                             drop_last=False,
                             collate_fn=PadCollate(valid_global_batch_size,
                                                   {"input_ids": 0,
                                                    "attention_mask": 0,
                                                    "token_type_ids": 0}))
   

    raw_predictions = valid(model, val_opts, val_dl, valid_samples_per_iteration)

    final_predictions = postprocess_qa_predictions(datasets["validation"],
                                               validation_features,
                                               raw_predictions)

    metric = load_metric("squad")
    formatted_predictions = [{"id": k, "prediction_text": v}
                            for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]}
                for ex in datasets["validation"]]
    metrics = metric.compute(predictions=formatted_predictions, references=references)
    print(metrics)
   

if __name__ == "__main__":
    main()