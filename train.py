
import argparse
import json
import time
import numpy as np
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import optim
import torch.cuda.amp as amp
import wandb
import gc

from models.wrapper import ModelWrapper
from utils.loading_utils import get_paired_loader, get_standard_loader

from utils.misc_utils import set_seed, find_latest_checkpoint

import faiss



def main():
    parser = argparse.ArgumentParser()

    # Wrapper
    parser.add_argument('--model', type=str, choices=['vanilla', 'adassl-v', 'adassl-s', 'gt'])
    parser.add_argument('--latent_predictor', type=str, default=None, 
        help='architecture of the latent variable r predictor MLP; \
            input dim is the embedding dimension and output dim is the dimension of r, e.g., 128-256-8')
    parser.add_argument('--editor', type=str, default=None, 
        help='architecture of the editor MLP; \
            input dim is the embedding dimension + the dimension of r and output dim is the embedding dimension, e.g., 136-256-128')
    parser.add_argument('--additional_view', action='store_true', help='use additional augmented views to predict r')
    parser.add_argument('--reg_beta', type=float, default=0.5, help='coefficient of the regularization loss')
    parser.add_argument('--reg_beta_warmup_steps', type=int, default=0, help='number of linear warmup steps for reg_beta')
    
    # SSL model
    parser.add_argument('--ssl_objective', type=str, default='simclr', choices=['simclr', 'byol'])
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--ema_tau', type=float, default=0.996, help='base EMA momentum')
    parser.add_argument('--projector', type=str, help='projector MLP architecture, e.g., 512-256-128')
    parser.add_argument('--predictor', type=str, default=None,help='predictor MLP architecture, e.g., 128-256-128')
    parser.add_argument('--learnable_lambda', action='store_true', help='if using learnable dimensional weight for SimCLR')
    parser.add_argument('--tau', type=float, default=0.1, help='SimCLR temperature')

    # Dataset
    parser.add_argument('--dataset', type=str, choices=['celeba', '3dident', 'inat'])
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='Noise ratio for iNat-1M')
    parser.add_argument('--use_standard_pairing', action='store_true', help='Use standard pairs')
    parser.add_argument('--transform_type', type=str, choices=['weak', 'strong'])
    parser.add_argument('--input_size', type=int, choices=[128, 224, 64])

    # Training
    parser.add_argument('--train_steps', type=int)
    parser.add_argument('--log_steps', type=int, metavar='N', help='log frequency (in steps)')
    parser.add_argument('--save_steps', type=int, metavar='N', help='save frequency (in steps)')
    parser.add_argument('--eval_steps', type=int, metavar='N', help='eval frequency (in steps)')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # Directories
    parser.add_argument('--data_dir', type=Path, metavar='DIR')
    parser.add_argument('--save_dir', type=Path, metavar='DIR')
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint in save_dir')

    # Other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--faiss_threads', type=int, default=1)
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline'])

    args = parser.parse_args()

    if args.ssl_objective == 'byol':
        assert args.predictor is not None

    print(args)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    with open(args.save_dir / 'args.txt', 'w') as f:
        args_copy = deepcopy(args.__dict__)
        args_copy['save_dir'] = str(args.save_dir)
        args_copy['data_dir'] = str(args.data_dir)
        json.dump(args_copy, f, indent=2)

    main_worker(args)



def main_worker(args):

    set_seed(args.seed)

    wandb.init(
        project=f"ssl_from_structural_invariance_{args.dataset}",
        name=f"{args.ssl_objective}_{args.transform_type}_{args.model}",
        mode=args.wandb_mode,
        config=dict(
            model=args.model,
            ssl_objective=args.ssl_objective,
            transform_type=args.transform_type,
            dataset=args.dataset,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            train_steps=args.train_steps,
            seed=args.seed,
            noise_ratio=args.noise_ratio
        ),
        job_type='train'
    )

    torch.backends.cudnn.benchmark = True

    if args.dataset == '3dident':
        # set FAISS to single thread usage as pytorch already uses multithreading to call FAISS
        faiss.omp_set_num_threads(args.faiss_threads)

    train_loader = get_paired_loader(args)
    val_split = 'test' if args.dataset == '3dident' else 'val'
    val_loader = get_standard_loader(args, split=val_split, train=False)

    model = ModelWrapper(args).cuda()

    # Use the same learning rate for online regressor and the backbone
    param_weights = []
    param_biases = []
    for name, param in model.named_parameters():
        if param.ndim == 1: 
            param_biases.append(param)
        else: 
            param_weights.append(param)

    parameters = [{'params': param_weights, 'lr': args.learning_rate, 'weight_decay': args.weight_decay}, 
                {'params': param_biases, 'lr': args.learning_rate, 'weight_decay': 0.0}]

    optimizer = optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    ###################
    ##### Checkpoint Resumption
    ###################

    global_step = 0
    checkpoint_path = None

    if args.resume:
        # Auto-detect latest checkpoint
        checkpoint_path = find_latest_checkpoint(args.save_dir)
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cuda')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            global_step = checkpoint['step']

            # Important to copy the steps since they are not saved in the checkpoint
            if hasattr(model, 'latent_sampler') and model.latent_sampler is not None:
                model.latent_sampler.current_step = global_step
            if args.ssl_objective == 'byol':
                model.ssl_model.step = global_step

            print(f"Resumed from step {global_step}")

        else:
            checkpoint_path = None
            print(f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch.")


    # Set total_steps AFTER loading checkpoint to ensure it uses the new value
    # This is important for learning rate schedules and other step-dependent components
    model.ssl_model.total_steps = args.train_steps

    ###################
    ##### Training
    ###################

    start_time = time.time()

    running_losses = defaultdict(float)

    scaler = amp.GradScaler(enabled=args.amp)

    model.train()

    # If resuming and we're at an eval checkpoint, run eval now
    # This handles the case where the job died during eval after saving checkpoint
    if checkpoint_path and global_step > 0 and global_step % args.eval_steps == 0:
        print(f"Running eval at resumed step {global_step}")
        torch.cuda.empty_cache()
        model.eval()
        log_results = eval(args, model.ssl_model, model.evaluator, val_loader)
        wandb.log(log_results, step=global_step)
        model.train()
    
    while global_step < args.train_steps:
        for step, batch in enumerate(train_loader):

            if global_step >= args.train_steps:
                break
            
            optimizer.zero_grad()

            if args.dataset == 'inat':
                x, y, x_aug, y_aug, cat, cat_super = batch
                eval_targets = dict(cat=cat.cuda(non_blocking=True), supercat=cat_super.cuda(non_blocking=True))
                z_x2y, z_y2x = None, None
            
            else:
                x, y, x_aug, y_aug, z_x2y, z_y2x, z_x = batch
                eval_targets = dict(factors=z_x.cuda(non_blocking=True))
                z_x2y, z_y2x = z_x2y.cuda(non_blocking=True), z_y2x.cuda(non_blocking=True)
            
            with amp.autocast(enabled=args.amp):
                losses = model(
                        x.cuda(non_blocking=True), 
                        y.cuda(non_blocking=True), 
                        x_aug.cuda(non_blocking=True),
                        y_aug.cuda(non_blocking=True),
                        z_x2y,
                        z_y2x,
                        eval_targets)

            loss_total = sum(losses.values())
            for k, v in losses.items():
                running_losses[k] += losses[k].item()

            scaler.scale(loss_total).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            model.update_states()

            ###################
            ##### Logging 
            ###################

            if (global_step + 1) % args.log_steps == 0:

                gc.collect()

                loss_stats = {}
                for k, v in running_losses.items():
                    loss_stats[k] = v / args.log_steps
                    running_losses[k] = 0

                general_stats = dict(time=int(time.time() - start_time))
                eval_stats = model.evaluator.get_eval_stats()
                model_stats = model.get_states()

                wandb.log(general_stats | loss_stats | eval_stats | model_stats, step=global_step+1)

            if (global_step + 1) % args.save_steps == 0:
                save_loc = args.save_dir / (f'ckpt_{global_step+1}.pt')
                torch.save({
                    'step': global_step+1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, save_loc)

            if (global_step + 1) % args.eval_steps == 0:
                torch.cuda.empty_cache()
                model.eval()
                eval_results = eval(args, model.ssl_model, model.evaluator, val_loader)
                    
                # Log overall eval metrics to WandB
                wandb.log(eval_results, step=global_step+1)
                
                model.train()

            global_step += 1

    # final ckpt
    torch.save(model.ssl_model.backbone.state_dict(), args.save_dir / 'resnet.pt')
    torch.save(model.ssl_model.projector.state_dict(), args.save_dir / 'projector.pt')

    wandb.finish()


@torch.no_grad()
def eval(args, ssl_model, evaluator, loader):
    if args.dataset in ['celeba', '3dident']:
        return eval_celeba_ident(ssl_model, evaluator, loader)
    elif args.dataset == 'inat':
        return eval_inat(ssl_model, evaluator, loader)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")


@torch.no_grad()
def eval_celeba_ident(ssl_model, evaluator, loader):
    # For F1, we save all the predictions and compute the metrics at the end
    results = dict()
    for enc_type in evaluator.enc_types:
        for target_name in evaluator.target_dims.keys():
            predictor = getattr(evaluator, f'pred_{enc_type}_{target_name}')
            y_all = []
            out_all = []

            for x, y in loader:
                out = ssl_model.backbone(x.cuda(non_blocking=True))
                
                if enc_type == 'emb':
                    out = F.normalize(ssl_model.projector(out), p=2, dim=1)

                out = predictor(out)

                y_all.append(y.cpu().numpy())
                out_all.append(out.cpu().numpy())

            y_all = np.concatenate(y_all)
            out_all = np.concatenate(out_all)

            metric_scores = evaluator.get_scores(out_all, y_all)

            for metric_name, score in metric_scores.items():
                results[f'val_{metric_name}_{enc_type}_{target_name}'] = score

    return results


@torch.no_grad()
def eval_inat(ssl_model, evaluator, loader):
    # For larger datasets like iNat and simpler metrics like accuracy, we compute them on the fly

    total_correct = dict()
    total_count = dict()

    for enc_type in evaluator.enc_types:
        for target_name in evaluator.target_dims.keys():
            for metric_name in evaluator.metrics.keys():
                total_correct[f'{metric_name}_{enc_type}_{target_name}'] = 0
                total_count[f'{metric_name}_{enc_type}_{target_name}'] = 0

    # Single pass through the validation set
    for x, y, y_super in loader:
        repr_out = ssl_model.backbone(x.cuda(non_blocking=True))
        emb_out = F.normalize(ssl_model.projector(repr_out), p=2, dim=1)

        targets = dict(cat=y.cuda(non_blocking=True), supercat=y_super.cuda(non_blocking=True))

        for enc_type in evaluator.enc_types:
            for target_name, target in targets.items():
                out = emb_out if enc_type == 'emb' else repr_out
                predictor = getattr(evaluator, f'pred_{enc_type}_{target_name}')
                logits = predictor(out)

                metric_scores = evaluator.get_scores(logits, target)
                for metric_name, score in metric_scores.items():
                    total_correct[f'{metric_name}_{enc_type}_{target_name}'] += score * target.shape[0]
                    total_count[f'{metric_name}_{enc_type}_{target_name}'] += target.shape[0]
    
    results = dict()
    for k, v in total_correct.items():
        results[f'val_{k}'] = v / total_count[k]

    return results


if __name__ == '__main__':
    main()