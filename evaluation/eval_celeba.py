

import numpy as np
import pickle
from utils.eval_utils import minority_f1, load_pretrained_models
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc_utils import parse_sizes
from utils.loading_utils import get_standard_loader
import wandb
from collections import defaultdict
import time
from torch import optim





def run_eval_celeba(args):

    wandb.init(
        project=f"ssl_from_structural_invariance_{args.dataset}",
        name=f"{args.ssl_objective}_{args.transform_type}_{args.model}",
        config=dict(
            model=args.model,
            ssl_objective=args.ssl_objective,
            dataset=args.dataset,
            transform_type=args.transform_type,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            train_steps=args.train_steps,
            seed=args.seed
        ),
        job_type='eval'
    )

    torch.backends.cudnn.benchmark = True


    backbone, projector = load_pretrained_models(args)

    train_loader = get_standard_loader(args, split='train', train=True)
    test_loader = get_standard_loader(args, split='test', train=False)

    repr_dim = parse_sizes(args.projector)[0]
    emb_dim = parse_sizes(args.projector)[-1]
    num_factors = train_loader.dataset.num_factors
    predictor_repr = nn.Linear(repr_dim, num_factors).cuda()
    predictor_emb = nn.Linear(emb_dim, num_factors).cuda()

    # Setup optimizer for predictor heads only
    parameters = []
    for pred in [predictor_repr, predictor_emb]:
        param_weights = []
        param_biases = []
        for name, param in pred.named_parameters():
            if param.ndim <= 1: 
                param_biases.append(param)
            else: 
                param_weights.append(param)
        parameters += [{'params': param_weights, 'weight_decay': args.weight_decay}, 
                {'params': param_biases, 'weight_decay': 0.0}]
    

    optimizer = optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    train(args, backbone, projector, predictor_repr, predictor_emb, train_loader, test_loader, optimizer)
    
    # Final evaluation
    results = eval(backbone, projector, predictor_repr, predictor_emb, test_loader)
    
    # Save final results and predictors
    pickle.dump(results, open(args.save_dir / 'test_logs.pkl', 'wb'))
    torch.save(predictor_repr.state_dict(), args.save_dir / 'probe_repr.pt')
    torch.save(predictor_emb.state_dict(), args.save_dir / 'probe_emb.pt')
    
    wandb.finish()


def train(args, backbone, projector, predictor_repr, predictor_emb, train_loader, test_loader, optimizer):
    start_time = time.time()
    running_losses = defaultdict(float)
    global_step = 0
    
    while global_step < args.train_steps:
        for step, (x, y) in enumerate(train_loader):
            if global_step >= args.train_steps:
                break
                
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            
            # Get frozen representations
            with torch.no_grad():
                repr_x = backbone(x)
                emb_x = projector(repr_x)
            
            # Train predictors
            losses = dict()
            for enc_type in ['repr', 'emb']:
                x_in = repr_x if enc_type == 'repr' else F.normalize(emb_x, p=2, dim=1)
                pred = predictor_repr if enc_type == 'repr' else predictor_emb
                losses['l_'+enc_type] = F.binary_cross_entropy_with_logits(pred(x_in), y)
            
            total_loss = sum(losses.values())
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            for k, v in losses.items():
                running_losses[k] += v.item()
            
            # Logging
            if (global_step + 1) % args.log_steps == 0:
                log_dict = {'time': int(time.time() - start_time)}
                for k, v in running_losses.items():
                    log_dict[k] = v / args.log_steps
                    running_losses[k] = 0
                
                wandb.log(log_dict, step=global_step+1)
            
            # Evaluation on test set
            if (global_step + 1) % args.eval_steps == 0:
                results = eval(backbone, projector, predictor_repr, predictor_emb, test_loader)
                wandb.log(results, step=global_step+1)
            
            global_step += 1


@torch.no_grad()
def eval(backbone, projector, predictor_repr, predictor_emb, test_loader):
    # Evaluate trained predictors on test set
    
    results = {}
    
    for enc_type in ['repr', 'emb']:
        predictor = predictor_repr if enc_type == 'repr' else predictor_emb
        
        y_all = []
        out_all = []
        
        for x, y in test_loader:
            x = x.cuda(non_blocking=True)
            
            repr_x = backbone(x)
            x_input = F.normalize(projector(repr_x), p=2, dim=1) if 'emb' in enc_type else repr_x

            out = predictor(x_input)
            
            y_all.append(y.cpu().numpy())
            out_all.append(out.cpu().numpy())
        
        y_all = np.concatenate(y_all)
        out_all = (np.concatenate(out_all) > 0).astype(int)
        
        results[f'acc_test_{enc_type}'] = (out_all == y_all).astype(float).mean()
        results[f'f1_test_{enc_type}'] = minority_f1(y_all, out_all).mean()
    
    return results
