import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from models.simclr import SimCLR
from models.byol import BYOL
from models.sampler import LatentSampler

from models.evaluator import OnlineEvaluator
from utils.model_utils import build_mlp, initialize_mlp
from utils.misc_utils import parse_sizes

ssl_models = {'simclr': SimCLR, 'byol': BYOL}


class ModelWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.ssl_objective = args.ssl_objective
        self.dataset = args.dataset
        self.model = args.model
        self.ssl_model = ssl_models[self.ssl_objective](args)
        self.additional_view = args.additional_view
        self.reg_beta = args.reg_beta
        self.learnable_lambda = args.learnable_lambda

        self.repr_dim = parse_sizes(args.projector)[0]
        self.emb_dim = parse_sizes(args.projector)[-1]

        self.evaluator = OnlineEvaluator(args)

        # AdaSSL-V: Construct p(z | x), q(z | x, y), and the MLP editor
        if self.model == 'adassl-v':
            self.latent_sampler = LatentSampler(args)
            sizes = parse_sizes(args.editor)        
            if len(sizes) == 1:
                # If args.editor is a number, we simply edit the embeddings by adding the predicted r
                self.editor = FoldAddition()
            else:
                self.editor = build_mlp(sizes, act='relu', norm='bn', bias=False)
                initialize_mlp(self.editor, act='relu')

        # AdaSSL-S: Construct the latent predictor, and the linear residual editor
        elif self.model == 'adassl-s':
            self.latent_sampler = LatentSampler(args)
            r_dim = parse_sizes(args.latent_predictor)[-1]
            self.editor_As = nn.Linear(self.emb_dim, r_dim)
            self.editor_Bs = nn.Linear(r_dim, self.emb_dim, bias=False)
            torch.nn.init.zeros_(self.editor_Bs.weight)

        # GT: Construct the MLP editor
        elif self.model == 'gt':
            sizes = parse_sizes(args.editor)    
            self.editor = build_mlp(sizes, act='relu', norm='bn', bias=False)
            initialize_mlp(self.editor, act='relu')
        
        # Learnable dimensional weight for SimCLR
        if self.ssl_objective != 'byol':
            lambda_init = math.log(math.e - 1)
            self.lambda_train = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, x, y, x_aug, y_aug, z_x2y, z_y2x, eval_targets):
        """
        param x (bsz, ...): context view (x in paper)
        param y (bsz, ...): target view (x^+ in paper)
        param x_aug (bsz, ...): another augmented view of x (x^{++} in paper)
        param y_aug (bsz, ...): another augmented view of y (since the loss is symmetric)
        param z_x2y (bsz, d_r): GT difference between the latent codes of x and y, if available (None otherwise)
        param z_y2x (bsz, d_r): GT difference between the latent codes of y and x, if available (None otherwise)
        param eval_targets (dict): dictionary of evaluation targets
        """

        # Encode the images into representations and embeddings
        if self.ssl_objective == 'simclr':
            repr_x, _, emb_x, emb_y = self.ssl_model(x, y)
        else:
            repr_x, _, emb_x, emb_y, target_emb_x, target_emb_y = self.ssl_model(x, y)

        # Predict the target views
        if self.additional_view:
            emb_x_aug = self.ssl_model.encode(x_aug)
            emb_y_aug = self.ssl_model.encode(y_aug)

            # Use f(x) and f(x^{++}) to predict r if x^{++} is available
            y_hat, losses_x2y = self.predict(emb_x, emb_y_aug, r=z_x2y)
            x_hat, losses_y2x = self.predict(emb_y, emb_x_aug, r=z_y2x)

        elif self.ssl_objective == 'simclr':
            y_hat, losses_x2y = self.predict(emb_x, emb_y, r=z_x2y)
            x_hat, losses_y2x = self.predict(emb_y, emb_x, r=z_y2x)
        
        else:
            y_hat, losses_x2y = self.predict(emb_x, target_emb_y, r=z_x2y)
            x_hat, losses_y2x = self.predict(emb_y, target_emb_x, r=z_y2x)

        # Average losses
        losses = {k: (v + losses_y2x[k]) / 2 for k, v in losses_x2y.items()}

        # Compute the SSL loss
        if self.ssl_objective == 'simclr':
            cont = torch.cat([x_hat, y_hat], dim=0)
            targ = torch.cat([emb_x, emb_y], dim=0)

            lambda_hat = F.softplus(self.lambda_train) if self.learnable_lambda else 1.0
            losses |= self.ssl_model.calc_loss(cont, targ, dimension_weight=lambda_hat)
        
        else:
            losses |= self.ssl_model.calc_loss(x_hat, target_emb_x, y_hat, target_emb_y)

        # Online evaluation
        losses |= self.evaluator(repr_x.detach(), emb_x.detach(), eval_targets)

        return losses

    def predict(self, emb_x, emb_y, r=None):
        """
        # emb_x (bsz, d): context embedding
        # emb_y (bsz, d): target embedding
        # r (bsz, d_r): GT r, if available (None otherwise)
        """

        # Store the regularization losses if any
        losses = {}

        if self.model in ['adassl-v', 'adassl-s']:
            # Sample r using the latent sampler
            r_hat, losses = self.latent_sampler(emb_x, emb_y)
            # Edit the context embedding f(x) with information of r to get the target prediction \hat{f(x^+)} (before predictor if there is one)
            y_hat = self.edit(emb_x, r_hat)

        elif self.model == 'gt' and r is not None:
            # If the GT r is available, use it to edit the context embedding f(x)  
            y_hat = self.edit(emb_x, r)

        else:
            # Vanilla prediction before predictor: \hat{f(x^+)} = f(x)
            y_hat = emb_x

        return y_hat, losses

    def edit(self, x, r):
        if self.model in ['adassl-v', 'gt']:
            y_hat = self.editor(torch.cat([x, r], dim=1))
        elif self.model == 'adassl-s':
            # Modular editors (r is sparse)
            # This is equivalent to Eq. 11 in the paper
            r = F.tanh(r)
            res = r * self.editor_As(x)
            y_hat = x + self.editor_Bs(res)
        return y_hat

    def update_states(self):
        self.ssl_model.update_states()

    def get_states(self):
        states = {}
        # Add reg_beta logging if adassl
        if hasattr(self, 'latent_sampler'):
            states['reg_beta'] = self.latent_sampler.reg_beta

        # Any additional model states can be logged here as well
        return states


class FoldAddition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xr):
        x, r = xr.chunk(2, dim=-1)
        return x + r
