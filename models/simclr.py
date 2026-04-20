import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import build_mlp
from models.resnet import resnet18
from torchvision.models import resnet50
from utils.misc_utils import parse_sizes

def _mask_correlated_samples(bsz):
    """
    Generate a boolean mask which masks out the similarity between views of the same example in the similarity matrix
    e.g., a mask for batch size = 2 is a 4x4 matrix (due to two views)
        0  1  0  1
        1  0  1  0
        0  1  0  1  
        1  0  1  0 
    """
    N = 2 * bsz
    mask = torch.ones((N, N), dtype=bool)
    mask.fill_diagonal_(0)
    mask[:, bsz:].fill_diagonal_(0)
    mask[bsz:, :].fill_diagonal_(0)
    return mask


class NT_Xent(nn.Module):
    def __init__(self, bsz, tau=0.1):
        super().__init__()
        self.bsz = bsz
        self.tau = tau

        self.mask = _mask_correlated_samples(bsz)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")


    def forward(self, cont, targ, dimension_weight=1):
        """
        Contrastive loss on [cont, targ]
        param cont (2*bsz, d): the stacked edited views
        param target (2*bsz, d): the stacked original views
        returns the symmetric NT-Xent loss
        """

        N = cont.size(0)
        bsz = N // 2

        cont = F.normalize(cont, p=2, dim=1)
        targ = F.normalize(targ, p=2, dim=1)

        # Optionally scale the similarity by a learnable dimensional weight
        sim = cont @ targ.t() * dimension_weight / self.tau
        sim = sim.clamp(min=-20, max=20)    # stability

        positive_samples = torch.diag(sim).reshape(N, 1)

        mask = _mask_correlated_samples(bsz) if bsz != self.bsz else self.mask
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)

        return loss


class SimCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone == 'resnet18':
            self.backbone = resnet18(zero_init_residual=True, input_size=args.input_size)
        elif args.backbone == 'resnet50':
            self.backbone = resnet50(weights=None)
        else:
            raise ValueError(f'Invalid backbone: {args.backbone}')
            
        # We do not need a linear classifier for SSL
        self.backbone.fc = nn.Identity()       
            
        self.criterion = NT_Xent(args.batch_size, tau=args.tau)

        sizes = parse_sizes(args.projector)
        self.projector = build_mlp(sizes, act='relu', norm=None, bias=False)

        self.total_steps = None

    def encode(self, x):
        return self.projector(self.backbone(x))

    def forward(self, x, y):
        # x (bsz, d): context
        # y (bsz, d): target

        repr_x = self.backbone(x)
        repr_y = self.backbone(y)

        emb_x = self.projector(repr_x)
        emb_y = self.projector(repr_y)

        return repr_x, repr_y, emb_x, emb_y

    def calc_loss(self, context, target, dimension_weight=1.0):
        # The loss is always symmetric
        # param context (2*bsz, d): the stacked edited views
        # param target (2*bsz, d): the stacked original views
        loss = self.criterion(context, target, dimension_weight)
        return dict(l_ssl=loss)

    def update_states(self):
        pass

