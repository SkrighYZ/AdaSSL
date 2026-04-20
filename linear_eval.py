

import argparse
from pathlib import Path
import json
from evaluation.eval_3dident import run_eval_3dident
from evaluation.eval_celeba import run_eval_celeba
from utils.misc_utils import set_seed


def main():
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--save_dir', type=Path, metavar='DIR', required=True, help='directory with pretrained model and args.txt')
    parser.add_argument('--data_dir', type=Path, metavar='DIR', required=True)

    # Training
    parser.add_argument('--train_steps', type=int)
    parser.add_argument('--eval_steps', type=int)
    parser.add_argument('--log_steps', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_batch_size', type=int)

    # System
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--seed', default=123, type=int)

    eval_args = parser.parse_args()
    
    # Load original training arguments
    args_file = eval_args.save_dir / 'args.txt'
    if not args_file.exists():
        raise FileNotFoundError(f"Could not find args.txt in {eval_args.save_dir}")
    
    with open(args_file, 'r') as f:
        train_args = json.load(f)

    args = argparse.Namespace(**train_args)
    
    # Override with eval-specific args
    args.data_dir = Path(eval_args.data_dir)
    args.save_dir = Path(eval_args.save_dir)
    args.train_steps = eval_args.train_steps
    args.eval_steps = eval_args.eval_steps
    args.log_steps = eval_args.log_steps
    args.learning_rate = eval_args.learning_rate
    args.weight_decay = eval_args.weight_decay
    args.batch_size = eval_args.batch_size
    args.eval_batch_size = eval_args.eval_batch_size
    args.num_workers = eval_args.num_workers
    args.prefetch_factor = eval_args.prefetch_factor
    args.seed = eval_args.seed

    print(args)

    set_seed(args.seed)
    
    if args.dataset == '3dident':   
        run_eval_3dident(args)
    elif args.dataset == 'celeba':
        run_eval_celeba(args)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

if __name__ == "__main__":
    main()