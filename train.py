import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import torch.profiler
from contextlib import ExitStack
import logging

from tqdm import tqdm
import numpy as np
import random
import argparse
import os, gc, time
from datetime import timedelta
import piq

from mlsr_training.data import SequenceDataset, ExampleDataset
from mlsr_training.model import AMDNetTiny, KPNBaseline, CPNBaseline
from mlsr_training.utils import REPROJECT, rgb_to_luminance, convert_color_space, match_sub_arg_params, parse_sub_arg_params
from mlsr_training.loss import instantiate_losses, instantiate_metrics
import mlsr_training.cov

import optuna, pathlib
from typing import Any, Dict, List


reprojection_obj_bicubic = None     # Lazy init with w/h
reprojection_obj_bilinear = None    # Lazy init with w/h

def ticks_to_human_time(ticks:int, digits:int=3):
    seconds = ticks # ticks are seconds...
    i_sec, f_sec = divmod(round(seconds*10**digits), 10**digits)
    return ("{}.{:0%d.0f}" % digits).format(timedelta(seconds=i_sec), f_sec)

def do_batch(args:argparse.Namespace, model:torch.nn.Module, batch:List, loss_objs:Dict):
    # Extract buffers from batched tensors
    sr_color_b, lr_mv_b, hr_color_b = batch
    n,t,_,h,w = sr_color_b.shape

    global reprojection_obj_bicubic
    if reprojection_obj_bicubic is None:
        reprojection_obj_bicubic = REPROJECT(w, h, 'bicubic').to(device=args.device)
    global reprojection_obj_bilinear
    if reprojection_obj_bilinear is None:
        reprojection_obj_bilinear = REPROJECT(w, h, 'bilinear').to(device=args.device)

    # Move all data to GPU asynchronously
    sr_color_b = sr_color_b.to(device=args.device, non_blocking=True).clip(args.hdr_min, args.hdr_max).float() # Upscaled color in linear HDR
    lr_mv_b = lr_mv_b.to(device=args.device, non_blocking=True)               # LR dilated motion vectors
    hr_color_b = hr_color_b.to(device=args.device, non_blocking=True).clip(args.hdr_min, args.hdr_max).float() # Target color in linear HDR

    # Tonemap colors
    sr_color_b = convert_color_space(sr_color_b, "linear", args.model_cspace)
    
    # Upscale batched MV
    hr_mv_b = F.interpolate(lr_mv_b.flatten(0,1), scale_factor=2, mode='nearest').view(n,t,2,h,w)

    # Iterate over time in a batch
    predictions = torch.empty_like(hr_color_b)
    for t in range(sr_color_b.shape[1]):
        sr_color = sr_color_b[:,t]
        hr_mv = hr_mv_b[:,t]

        # Initialize internal model states at the beginning of a sequence
        if t == 0:
            reprojected_prev_pred = sr_color
            reprojected_recurrent = torch.zeros(n, model.prediction_network.recurrent_channels, h, w, device=sr_color.device)
        
        # Reproject previous prediction and recurrent channel for t>0
        else:
            reprojected_prev_pred = reprojection_obj_bicubic(prev_pred, hr_mv)
            reprojected_prev_pred = reprojected_prev_pred.clip(min=args.ldr_min) # clip beacuse of bicubic
            # note that recurrent state should be bilinear interpolated
            reprojected_recurrent = reprojection_obj_bilinear(recurrent, hr_mv)
        
        # Prepare NN inputs (KPN version)
        # x = torch.cat([
        #     (sr_color - reprojected_prev_pred).abs().amax(dim=1, keepdim=True), # Max absolute difference
        #     rgb_to_luminance(sr_color),                                         # Current frame luma
        #     rgb_to_luminance(reprojected_prev_pred),                            # History frame luma
        #     reprojected_recurrent                                               # Recurrent channel
        # ], dim=1)

        if args.graph_create:
            model = torch.cuda.make_graphed_callables(model, (x, sr_color, reprojected_prev_pred))
            args.graph_create = False


        # Prepare NN inputs (CPN version)
        x = torch.cat([
            sr_color,
            reprojected_recurrent
        ], dim=1)

        # Model prediction
        pred, new_recurrent = model(x, sr_color, reprojected_prev_pred)

        # Store prediction for loss calculation
        predictions[:,t] = pred

        # Set internal model states for a next frame
        prev_pred = pred
        recurrent = new_recurrent
    
    # Color conversion
    predictions = convert_color_space(predictions, args.model_cspace, args.loss_cspace)
    hr_color_b = convert_color_space(hr_color_b, "linear", args.loss_cspace)

    losses = {}
    for loss_name, loss_obj in loss_objs.items():
        if 'temporal' in loss_name:
            losses[loss_name] = loss_obj(predictions, hr_color_b, hr_mv_b)
        else:
            losses[loss_name] = loss_obj(predictions.flatten(0,1), hr_color_b.flatten(0,1))

    return losses, predictions, hr_color_b

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Add `--arg`` and `--no-arg`, set passed default
    def add_bool(name:str, default_value:bool):
        parser.add_argument(f'--{name}', action='store_true')
        parser.add_argument(f'--no-{name}', dest=name, action='store_false')
        parser.set_defaults(**{name: default_value})

    parser.add_argument("--model", type=str, default="CPNBaseline")
    parser.add_argument("--optimizer", type=str, default="RMSprop")
    parser.add_argument('--output_dir', type=str, default='./output')

    # Note, recommend using `--train_batch_size 8 --batch_accumulate_count 4` on 
    # local machines (assuming ~20GB). On servers (assuming > 60GB RAM) 
    # the batch size should be 32. 
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--valid_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_accumulate_count', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--train_clip_length', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    
    add_bool("train", True)
    parser.add_argument('--train_dataset', type=str, default='/proj/dataset4/dongz/18films')
    parser.add_argument('--train_seq_count', type=int, default=20000)
    

    parser.add_argument('--training_losses', default=None, nargs="*")
    parser.add_argument('--validation_metrics', default=None, nargs="*")
    parser.add_argument('--testing_metrics', default=None, nargs="*")
    

    add_bool("valid", True)
    #parser.add_argument('--valid_dataset', type=str, default=None)
    parser.add_argument('--valid_dataset', type=str, default='/proj/dataset4/Downloads/validation_sequences/')
    parser.add_argument('--valid_seq_count', type=int, default=1230)

    add_bool("test", False)
    parser.add_argument('--test_dataset', type=str, default=None)
    parser.add_argument('--test_seq_count', type=int, default=10*640//32)

    # Auto weight losses (useful for testing)
    # Uses: https://openaccess.thecvf.com/content/WACV2021/papers/Groenendijk_Multi-Loss_Weighting_With_Coefficient_of_Variations_WACV_2021_paper.pdf
    add_bool("cov_weight", False)

    parser.add_argument('--model_cspace', type=str, default="st2084")
    parser.add_argument('--loss_cspace', type=str, default="st2084")
    parser.add_argument('--metric_cspace', type=str, default="st2084")

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hdr_min', type=float, default=1e-6)
    parser.add_argument('--hdr_max', type=float, default=10000.0)
    parser.add_argument('--ldr_min', type=float, default=1e-6)
    parser.add_argument('--ldr_max', type=float, default=1.0)    # not used


    optuna_args = parser.add_argument_group("Optuna")
    optuna_args.add_argument('--optuna_trials',      type=int, default=0, help="int: number of optuna trials to do")
    optuna_args.add_argument('--optuna_study_name',  type=str, default=None, help="str: name of optuna study")
    optuna_args.add_argument('--optuna_study_path',  type=str, default=None, help="str: path to optuna db (sqlite)")
    

    optuna_args.add_argument('--criteron_metric', type=str, default=None, help="str: validation metric to use as return value from training loop, and optuna reporting")

    parser.add_argument('--out_path', type=str, default=os.getcwd())
    
    # Enable Weights and Biases
    optuna_args.add_argument('--wandb',  type=str, default=None, help="str: Name of wandb project")

    # Enable Tensorboard
    add_bool('tb_enable', True)
    add_bool('profiler_enable', False)
    add_bool('profiler_performance', True)
    add_bool('profiler_memory', False)
    add_bool('profiler_stacks', True)
    add_bool('profiler_train', True)
    add_bool('profiler_validation', False)
    
    add_bool('graph_create', False)

    args = parser.parse_args()
    assert not (args.profiler_train and args.profiler_validation), "Can only profile one side or the other."

    return args

def main(args:argparse.Namespace) -> float:
    if args.out_path != os.getcwd():
        # maybe explicitly manage out paths later if this is an issue.
        os.makedirs(args.out_path, exist_ok=True)
        os.chdir(args.out_path)


    # Log arguments for posterity
    print("---------------------")
    print("Trainer:")
    print(args)
    print("---------------------")
    if args.wandb:
        import wandb
        wandb.login()
        wandb.init(project=args.wandb, config=args)
    else:
        wandb = None


    # Set logging config
    logging.basicConfig(format="%(levelname)s: %(message)s")

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    def get_dataset(path:str, seq_count:int, split:str, arg_name:str):
        if path is None:
            logging.warning(f"Using ExampleDataset for {arg_name}")
            return ExampleDataset(num_sequences=seq_count)
        else:
            if not os.path.exists(path):
                raise ValueError(f"--{arg_name} is not a valid path: `{path}`")
            return SequenceDataset(path, split=split, num_sequences=seq_count, name=arg_name)
    
    # Avoid the final incomplete batch but on the granularity of the accumulated batch size.
    args.train_seq_count = (args.train_seq_count // (args.train_batch_size * args.batch_accumulate_count)) * (args.train_batch_size * args.batch_accumulate_count)
    train_ds = get_dataset(args.train_dataset, args.train_seq_count, split='train', arg_name="train_dataset")
    valid_ds = get_dataset(args.valid_dataset, args.valid_seq_count, split='valid', arg_name="valid_dataset")
    test_ds = get_dataset(args.test_dataset, args.test_seq_count, split='train', arg_name="test_dataset")

    if args.train and train_ds.num_sequences == 0:
        raise Exception(f"Error: train dataset cannot be found at {args.train_dataset}")
    if args.valid and valid_ds.num_sequences == 0:
        raise Exception(f"Error: validation dataset cannot be found at {args.valid_dataset}")
    if args.test and test_ds.num_sequences == 0:
        raise Exception(f"Error: test dataset cannot be found at {args.test_dataset}")


    # Create train/valid dataloaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.train_batch_size, drop_last=False, num_workers=args.num_workers, pin_memory=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=args.valid_batch_size, drop_last=False, num_workers=args.num_workers, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.test_batch_size, drop_last=False, num_workers=args.num_workers, pin_memory=True)

    # Create a model
    if args.model == "AMDNetTiny":
        model = AMDNetTiny().to(device=args.device)
    elif args.model == "KPNBaseline":
        model = KPNBaseline().to(device=args.device)
    elif args.model == "CPNBaseline":
        model = CPNBaseline().to(device=args.device)
    else:
        raise ValueError("passed --model ({args.model}) is not recognized")
    # import pdb;pdb.set_trace()
    # Create an optimizer
    if match_sub_arg_params("RMSprop", args.optimizer):
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, **parse_sub_arg_params(args.optimizer))
    elif match_sub_arg_params("AdamW", args.optimizer):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, **parse_sub_arg_params(args.optimizer))
    else:
        raise ValueError("passed --optimizer ({args.optimizer}) is not recognized")

    # Create an scheduler
    # TODO: command line drive options
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1)


    # Default values.
    if args.training_losses is None:
        args.training_losses = ["l1(weight=0.21)", "ms-ssim(weight=0.5)", "temporal-l1(weight=0.29)"]
    if args.validation_metrics is None:
        args.validation_metrics = ["lpips", "ms-ssim(data_range=1.0)", "temporal-l1"]
    if args.testing_metrics is None:
        args.testing_metrics = ["lpips"]

    args.training_loss_objs, args.training_loss_weights = instantiate_losses(args.training_losses, args.device)
    args.validation_metric_objs = instantiate_metrics(args.validation_metrics, args.device)
    args.testing_metric_objs = instantiate_metrics(args.testing_metrics, args.device)
    
    #  Training losses also need weights. Val/test metrics do not.
    if args.cov_weight:
        args.training_loss_covs = {}
        for k in args.training_loss_objs.keys():
            args.training_loss_covs[k] = mlsr_training.cov.CovModule(cov_samples_max=len(train_dl)).to(device=args.device)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    tb_summary_writer = SummaryWriter(args.output_dir) if args.tb_enable else None
    with ExitStack() as context_stack:
        profiler = None
        if args.profiler_enable:
            profiler = torch.profiler.profile(
                activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                wait=20,
                warmup=2,
                active=5,
                repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_summary_writer.log_dir),
                with_stack=args.profiler_stacks,
                record_shapes=False,
                with_flops=False,
                profile_memory=args.profiler_memory,
            )
            context_stack.push(profiler)


        for epoch in range(args.epochs):
            # Try to prevent too much memory growth by clearing memory each epoch.
            gc.collect()
            torch.cuda.empty_cache()

            torch.manual_seed(args.seed + epoch)
            random.seed(args.seed + epoch)
            np.random.seed(args.seed + epoch)

            metrics = {}
            training_loss_dict = {"combined":0.0}
            validation_metric_dict = {}
            if args.criteron_metric is not None:
                validation_metric_dict[args.criteron_metric] = 0.0
            train_dl_epoch_ticks = train_proc_epoch_ticks = 0
            valid_dl_epoch_ticks = valid_proc_epoch_ticks = 0
            # Training
            if(args.train):
                train_dl.dataset.shuffle(epoch)
                model.train()

                # These need to match the order they are placed in the tensor
                loss_names = []
                for k in args.training_loss_objs.keys():
                    loss_names.append(k)
                    loss_names.append(f"weight-{k}")
                loss_names.append("combined")

                loss_values = torch.empty((len(train_dl),len(loss_names)), device=args.device, requires_grad=False)
                optimizer.zero_grad(set_to_none=True)

                def train_inner():
                    dl_epoch_ticks = proc_epoch_ticks = 0

                    dl_start_ticks = time.monotonic()
                    for batch_index, batch in enumerate(tqdm(train_dl)):
                        proc_start_ticks = dl_end_ticks = time.monotonic()
                        dl_epoch_ticks += dl_end_ticks - dl_start_ticks

                        with torch.profiler.record_function(f"Train_{epoch}_{batch_index}"):
                            # Pick a batch seed that will work the same for any batch_accumulate_count
                            batch_seed = epoch * (len(train_dl) // args.batch_accumulate_count) + (batch_index // args.batch_accumulate_count)
                            temporal_start = random.Random(batch_seed).randint(0, batch[0].shape[1] - args.train_clip_length)
                            # truncate the data inputs to the requested clip length
                            batch_small = []
                            for i in range(len(batch)):
                                batch_small.append( batch[i][:,temporal_start:temporal_start+args.train_clip_length] )
                            batch = batch_small

                            # Prediction
                            losses, _, _ = do_batch(args, model, batch, args.training_loss_objs)


                            if args.cov_weight:
                                assert len(losses) == len(args.training_loss_objs)
                                cov_list = []
                                for k,v in losses.items():
                                    cov_list.append(args.training_loss_covs[k](v))

                                # Accumulate call updates the weights, which we can log
                                loss = mlsr_training.cov.CovAccumulate(*cov_list).squeeze()

                                # Log the weights:
                                for k in losses.keys():
                                    args.training_loss_weights[k] = args.training_loss_covs[k].weight_normalized.squeeze()
                            else:
                                # Accumulate losses with static weights
                                loss = torch.zeros((1), device=args.device)
                                
                                for k,v in losses.items():
                                    loss += args.training_loss_weights[k] * v
                            
                            # insert losses into GPU tensor so we don't sync w/ cpu each batch.
                            # put them in the same order as the loss names, this is important for reporting
                            assert len(loss_values[batch_index]) == len(losses)*2+1
                            for i,(k,v) in enumerate(losses.items()):
                                loss_values[batch_index][i*2] = v
                                loss_values[batch_index][i*2 + 1] = args.training_loss_weights[k]
                            loss_values[batch_index][-1] = loss.unsqueeze(0)
                            
                            loss /= args.batch_accumulate_count
                            loss.backward()

                            # Allow gradient accumulation so we can run a similar batch size on server parts vs local parts,
                            # without changing other parameters.
                            if (((batch_index + 1) % args.batch_accumulate_count) == 0) or batch_index + 1 == len(train_dl):
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)

                                if profiler != None and args.profiler_train:
                                    profiler.step()
                        
                        proc_end_ticks = time.monotonic()
                        
                        proc_epoch_ticks += proc_end_ticks - proc_start_ticks
                        dl_start_ticks = time.monotonic()

                    avg_loss_values = torch.mean(loss_values, dim=0).cpu()
                    loss_dict = {}
                    for k,v in zip(loss_names, avg_loss_values):
                        loss_dict[k] = v.item()
                    return loss_dict, dl_epoch_ticks, proc_epoch_ticks
                
                # Run in a function (local scope) to ensure all tensors are freed
                training_loss_dict, train_dl_epoch_ticks, train_proc_epoch_ticks = train_inner()

                # Update scheduler
                scheduler.step()
                
                for k,v in training_loss_dict.items():
                    metrics[f"training/loss-{k}"] = v
                

            # Validation
            if(args.valid):
                model.eval()
                
                def val_inner():
                    dl_epoch_ticks = proc_epoch_ticks = 0

                    with torch.no_grad():
                        # These need to match the order they are placed in the tensor
                        metric_names = []
                        for k in args.validation_metric_objs.keys():
                            metric_names.append(k)

                        metrics_values = torch.empty((len(valid_dl),len(metric_names)), device=args.device, requires_grad=False)
                        dl_start_ticks = time.monotonic()
                        for batch_index, batch in enumerate(tqdm(valid_dl)):
                            proc_start_ticks = dl_end_ticks = time.monotonic()
                            dl_epoch_ticks += dl_end_ticks - dl_start_ticks

                            with torch.profiler.record_function(f"Validation_{epoch}_{batch_index}"):
                                # Prediction
                                metrics, _,_ = do_batch(args, model, batch, args.validation_metric_objs)

                                # insert metrics into GPU tensor so we don't sync w/ cpu each batch.
                                # put them in the same order as the metric names, this is important for reporting
                                assert len(metrics_values[batch_index]) == len(metrics)
                                for i,(k,v) in enumerate(metrics.items()):
                                    metrics_values[batch_index][i] = v

                                if profiler != None and args.profiler_validation:
                                    profiler.step()
                            
                            proc_end_ticks = time.monotonic()
                            proc_epoch_ticks += proc_end_ticks - proc_start_ticks
                            dl_start_ticks = time.monotonic()

                        avg_metrics_values = torch.mean(metrics_values, dim=0).cpu()
                        metric_dict = {}
                        for k,v in zip(metric_names, avg_metrics_values):
                            metric_dict[k] = v.item()
                        return metric_dict, dl_epoch_ticks, proc_epoch_ticks
                    
                # Run in a function (local scope) to ensure all tensors are freed
                validation_metric_dict, valid_dl_epoch_ticks, valid_proc_epoch_ticks = val_inner()

                for k,v in validation_metric_dict.items():
                    metrics[f"validation/metric-{k}"] = v


            metrics[f"perf/train-dl"] = train_dl_epoch_ticks
            metrics[f"perf/train-proc"] = train_proc_epoch_ticks
            metrics[f"perf/validation-dl"] = valid_dl_epoch_ticks
            metrics[f"perf/validation-proc"] = valid_proc_epoch_ticks
            
            if tb_summary_writer is not None:
                for k,v in metrics.items():
                    tb_summary_writer.add_scalar(k, v, epoch)

            if wandb:
                wandb.log(metrics)

            print(f"Epoch: {epoch}  Loss:{training_loss_dict['combined']}  Valid Loss:{validation_metric_dict}")
            print(f"Epoch: {epoch}  Train: dl:{ticks_to_human_time(train_dl_epoch_ticks)} proc:{ticks_to_human_time(train_proc_epoch_ticks)}   Valid: dl:{ticks_to_human_time(valid_dl_epoch_ticks)} proc:{ticks_to_human_time(valid_proc_epoch_ticks)}")


            if hasattr(args, 'optuna_trial') and args.optuna_trial:
                trial = args.optuna_trial

                if args.criteron_metric is not None:
                    trial.report(validation_metric_dict[args.criteron_metric], epoch)

                # Handle pruning based on the intermediate value.
                if trial.should_prune() or not np.isfinite(training_loss_dict["combined"]):
                    raise optuna.TrialPruned()
                
            if not np.isfinite(training_loss_dict["combined"]):
                raise Exception("Error Training loss was NaN. Training has failed.")

         
        # Test
        if(args.test):
            model.eval()
            def test_inner():
                with torch.no_grad():
                    # These need to match the order they are placed in the tensor
                    metric_names = []
                    for k in args.testing_metric_objs.keys():
                        metric_names.append(k)
                    metrics_values = torch.empty((len(test_dl),len(metric_names)), device=args.device, requires_grad=False)
                    for batch_index, batch in enumerate(tqdm(test_dl)):
                        # Prediction
                        metrics, _, _ = do_batch(args, model, batch, args.testing_metric_objs)
                        
                        # insert metrics into GPU tensor so we don't sync w/ cpu each batch.
                        # put them in the same order as the metric names, this is important for reporting
                        assert len(metrics_values[batch_index]) == len(metrics)
                        for i,(k,v) in enumerate(metrics.items()):
                            metrics_values[batch_index][i] = v
                    

                    avg_metrics_values = torch.mean(metrics_values, dim=0).cpu()
                    metric_dict = {}
                    for k,v in zip(metric_names, avg_metrics_values):
                        metric_dict[k] = v.item()
                    return metric_dict
            
            test_metric_dict = test_inner()

            if tb_summary_writer is not None:
                for k,v in test_metric_dict.items():
                    tb_summary_writer.add_scalar(f"test/metric-{k}", v, epoch)

            print(f"Test Loss: {test_metric_dict}")

    if tb_summary_writer is not None:
        tb_summary_writer.flush()
        tb_summary_writer.close()


    # Save model after training
    torch.save(model.state_dict(), './model.pt')

    if args.profiler_enable: print(profiler.key_averages(group_by_stack_n=10).table(sort_by='self_cpu_time_total', row_limit=10))
    if args.profiler_enable: print(profiler.key_averages(group_by_stack_n=10).table(sort_by='self_cuda_time_total', row_limit=10))

    if args.criteron_metric is not None:
        return validation_metric_dict[args.criteron_metric]
    else:
        return None


def create_study(args: argparse.Namespace, base_trials:List[Dict[str, Any]] = None):
    # if running multi-process there can be some competition to see who creates the optuna db.
    # only one will win, but other will need to retry.
    for i in range(10):
        study = None
        if os.path.exists(args.optuna_study_path):
            study = optuna.create_study(study_name=args.optuna_study_name, load_if_exists=True, storage=f"sqlite:///{args.optuna_study_path}", sampler=sampler, pruner=pruner)
            break
        
        import tempfile
        import shutil

        # Create a temp file and then move it to the final destination. only one will win this way.
        # (prevent timing issues at study creation time)
        with tempfile.TemporaryDirectory() as tmp:
            temp_path = os.path.join(tmp, 'test.db')
            # use temp path
            temp_study = optuna.create_study(study_name=args.optuna_study_name, load_if_exists=False, storage=f"sqlite:///{temp_path}", sampler=sampler, pruner=pruner)
            
            # Seed study with initial trials, This is optional but often useful
            # make sure only the starting run adds the trials though.
            # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.enqueue_trial
            for t in base_trials:
                temp_study.enqueue_trial(t)
            
            del temp_study
            
            try:
                pathlib.Path(args.optuna_study_path).parent.mkdir(exist_ok=True, parents=True)
                shutil.move(temp_path, args.optuna_study_path)
            except:
                # it's ok if we failed, one of our siblings likely succeeded
                pass

            del temp_path

    if study is None:
        raise Exception("Unable to create or load study...")

    return study

if __name__ == "__main__":
    args = parse_args()

    # Log any additional data here into args, which will be printed/logged later.
    try:
        args.device_name = torch.cuda.get_device_name(args.device)
        args.device_memory = torch.cuda.get_device_properties(args.device).total_memory
    except:
        pass

    if args.optuna_trials < 1:
        main(args)
    else:
        if args.optuna_study_name is None:
            raise ValueError("optuna usage requires a trial name: pass --optuna_study_name [name] ")
        if args.optuna_study_path is None:
            args.optuna_study_path = os.path.join(args.out_path, "optuna.db")

        sampler = optuna.samplers.TPESampler()
        pruner = optuna.pruners.SuccessiveHalvingPruner()
        pruner = optuna.pruners.PatientPruner(wrapped_pruner=pruner, patience=2)
        
        # simple wrapper around main() which applies optuna parameters
        def objective(trial:optuna.trial.Trial):
            import copy
            args_copy = copy.deepcopy(args)
            args_copy.optuna_trial = trial
            # put each run into a separate output folder (so we can look back on the respective .pt files, etc)
            args_copy.out_path = os.path.join(args.out_path, args.optuna_study_name, f"{trial.number:04}")

            ###################################
            # override parameters as needed for study goals

            args_copy.lr = trial.suggest_float("lr", 1e-5, 0.5, log=True)

            ###################################

            return main(args_copy)
        
        study = create_study(args, [
            ###################################
            # Add explicit trials (or not) to study.
            
            { "lr": 5e-4, }  # Set the first trial to the current default value
            
            ###################################
            ])
        
        study.optimize(objective, n_trials=args.optuna_trials)

        print(study.best_params)

