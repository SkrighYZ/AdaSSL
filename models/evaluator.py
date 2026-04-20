import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import defaultdict
from utils.misc_utils import parse_sizes
from utils.eval_utils import minority_f1


class OnlineEvaluator(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Dataset‑specific config
        self.dataset = args.dataset
        self._init_dataset_params(args)

        # Running stats & eval types
        self.running_stats = defaultdict(float)
        self.steps = 0

        # Build all predictors
        repr_dim = parse_sizes(args.projector)[0]
        emb_dim  = parse_sizes(args.projector)[-1]
        self._build_predictors(repr_dim, emb_dim)

    @torch.no_grad()
    def get_scores(self, out, y):
        # iNat assumes torch tensors
        # Other datasets assume numpy arrays

        if self.dataset == 'celeba':
            # Convert logits to binary predictions
            out = (out > 0).astype(int) 

        scores = {}
        for metric_name, metric_fn in self.metrics.items():
            scores[metric_name] = metric_fn(out, y)

        return scores


    def _init_dataset_params(self, args):

        if args.dataset == 'celeba':
            self.criterion    = F.binary_cross_entropy_with_logits
            self.metrics      = {
                'acc': lambda x, y: (x == y).astype(float).mean(),   
                'f1': lambda x, y: minority_f1(y, x).mean()
            }
            self.enc_types = ['repr', 'emb']
            self.target_dims = dict(factors=40)

        elif args.dataset == '3dident':
            self.criterion    = F.mse_loss
            self.metrics      = {
                # assert target is normalized
                'r2': lambda x, y: 1 - ((x - y) ** 2).mean()
            }
            self.enc_types = ['emb']
            self.target_dims = dict(factors=10)

        elif args.dataset == 'inat':
            self.criterion    = F.cross_entropy
            self.metrics      = {
                'top1': lambda x, y: (x.argmax(dim=1) == y).float().mean().item(),   
                'top5': lambda x, y: (x.topk(5, dim=1).indices == y.unsqueeze(1)).any(dim=1).float().mean().item()
            }
            self.enc_types = ['repr']
            self.target_dims = dict(cat=5000, supercat=11)

        else:
            raise ValueError(f"Unknown dataset {args.dataset}")

    def _build_predictors(self, repr_dim, emb_dim):
        for enc_type in self.enc_types:
            enc_dim = dict(repr=repr_dim, emb=emb_dim)[enc_type]
            for n, d in self.target_dims.items():
                setattr(self, f'pred_{enc_type}_{n}', nn.Linear(enc_dim, d))
        
    def get_eval_stats(self):
        # Calculate running averages of the metrics and reset them 
        stats = {}
        for k, v in self.running_stats.items():
            stats[k] = v / self.steps
            self.running_stats[k] = 0
        self.steps = 0
        return stats

    def forward(self, repr_x, emb_x, targets):
        # Train the probes for all metrics on frozen representations/embeddings
        # The gradients are detached before reaching the encoder/projector
        losses = {}
        for enc_type in self.enc_types:
            x = repr_x if enc_type == 'repr' else F.normalize(emb_x, p=2, dim=1)

            for target_name, target in targets.items():
                out = getattr(self, f'pred_{enc_type}_{target_name}')(x)
                losses[f'l_{enc_type}_{target_name}'] = self.criterion(out, target)

                out = out.detach()
                target = target.detach()

                if self.dataset != 'inat':
                    out = out.cpu().numpy()
                    target = target.cpu().numpy()

                metric_scores = self.get_scores(out, target)
                for metric_name, score in metric_scores.items():
                    self.running_stats[f'train_{metric_name}_{enc_type}_{target_name}'] += score

        self.steps += 1

        return losses
