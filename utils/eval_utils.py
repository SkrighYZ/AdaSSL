
import torch
from sklearn.metrics import f1_score
import numpy as np
from models.resnet import resnet18
from utils.model_utils import build_mlp
from utils.misc_utils import parse_sizes
import torch.nn as nn



def load_pretrained_models(args):
    # Load backbone and freeze it
    backbone = resnet18(input_size=args.input_size).cuda()
    backbone.fc = nn.Identity()  # Remove classification head
    backbone.load_state_dict(torch.load(args.save_dir / 'resnet.pt'))
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    # Load projector and freeze it
    projector_sizes = parse_sizes(args.projector)
    norm = 'bn' if args.ssl_objective == 'byol' else None
    projector = build_mlp(projector_sizes, act='relu', norm=norm, bias=False).cuda()
    projector.load_state_dict(torch.load(args.save_dir / 'projector.pt'))
    projector.eval()
    for param in projector.parameters():
        param.requires_grad = False

    return backbone, projector


def minority_f1(y, x):
    # F1 score for the minority class, calculated for each attribute
    scores = []
    for i in range(y.shape[1]):  # for each attribute
        positives = y[:, i].sum()
        negatives = len(y) - positives
        if positives < negatives:  
            # Minority class is 1
            f1 = f1_score(y[:, i], x[:, i], pos_label=1, zero_division=0)
        else:  
            # Minority class is 0
            f1 = f1_score(y[:, i], x[:, i], pos_label=0, zero_division=0)
        scores.append(f1)
    return np.array(scores)