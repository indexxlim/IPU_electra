from pathlib import Path
import time
import yaml

import torch
import transformers
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm, trange
from easydict import EasyDict

import popart
import poptorch

from pipeline_electra import PipelinedElectraForTokenClassification
from dataloader_ner import NERDataset, NERCollator

'''
Set Option
'''
def ipu_options(gradient_accumulation, replication_factor, device_iterations, train_option, seed=42):
    '''
    Set IPU Options 
    Batgch Parameters to watch out 
        1. micro_batch_size
        2. replication_factor
        3. device_iterations
    '''
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
    config_file = "finetune/ner_configuration.yaml"
    config = EasyDict(yaml.load(open(config_file).read(), Loader=yaml.loader))

    #data_path = 
    dataset = NERDataset()

    


    dataset = load_dataset()


    train()



if __name__ == "__main__":
    main()