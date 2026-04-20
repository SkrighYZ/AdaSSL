
import torch
import torch.nn as nn

import copy, math

from utils.model_utils import build_mlp
from models.resnet import resnet18
from utils.misc_utils import parse_sizes


class BYOL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = resnet18(zero_init_residual=True, norm='bn', act='relu', input_size=args.input_size)
        
        # We do not need a linear classifier for SSL
        self.backbone.fc = nn.Identity()     

        # Projector and predictor have no bias in the last layer
        # https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/utils/networks.py#L44
        
        sizes = parse_sizes(args.projector)
        self.projector = build_mlp(sizes, act='relu', norm='bn', bias=False)
        
        sizes = parse_sizes(args.predictor)
        self.predictor = build_mlp(sizes, act='relu', norm='bn', bias=False)

        # EMA
        self.step = 0
        self.total_steps = args.train_steps
        self.tau_base = args.ema_tau

        self.target_backbone = copy.deepcopy(self.backbone)
        for p in self.target_backbone.parameters():
            p.requires_grad = False

        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False

        # loss
        self.criterion = nn.CosineSimilarity(dim=1)

    def encode(self, x):
        return self.projector(self.backbone(x))

    def forward(self, x, y):
        # x (bsz, d): context
        # y (bsz, d): target

        repr_x = self.backbone(x)
        emb_x = self.projector(repr_x)
        repr_y = self.backbone(y)
        emb_y = self.projector(repr_y)

        with torch.no_grad():
            target_emb_x = self.target_projector(self.target_backbone(x))
            target_emb_y = self.target_projector(self.target_backbone(y))
        
        return repr_x, repr_y, emb_x, emb_y, target_emb_x, target_emb_y


    def calc_loss(self, x_hat, target_emb_x, y_hat, target_emb_y):
        # x_hat (bsz, d): predicted x embeddings from y
        # target_emb_x (bsz, d): target x embeddings
        # y_hat (bsz, d): predicted y embeddings from x
        # target_emb_y (bsz, d): target y embeddings

        pred_x = self.predictor(x_hat)
        pred_y = self.predictor(y_hat)

        loss = -self.criterion(pred_x, target_emb_x.detach()).mean() \
            - self.criterion(pred_y, target_emb_y.detach()).mean()

        return dict(l_ssl=loss)

    @torch.no_grad()
    def update_states(self):
        # Update momentum from self.tau_base to 1 following the cosine schedule
        
        self.step += 1
        tau = 1 - (1 - self.tau_base) * (math.cos(math.pi * self.step / self.total_steps) + 1) / 2

        for online_p, target_p in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            target_p.data += (1 - tau) * (online_p.data - target_p.data)
        for online_p, target_p in zip(self.projector.parameters(), self.target_projector.parameters()):
            target_p.data += (1 - tau) * (online_p.data - target_p.data)


