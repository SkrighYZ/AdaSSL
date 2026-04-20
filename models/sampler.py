import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.misc_utils import parse_sizes
from utils.model_utils import build_mlp, initialize_mlp

from utils.sampling_utils import kl_gaussian, reparam_gaussian, reparam_bernoulli


class LatentSampler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model = args.model
        self.reg_beta = 0
        self.reg_beta_target = args.reg_beta
        self.reg_beta_warmup_steps = getattr(args, 'reg_beta_warmup_steps', 0)
        self.train_steps = args.train_steps
        self.current_step = 0


        sizes = parse_sizes(args.latent_predictor)
        self.latent_dim = sizes[-1]

        sizes[-1] *= 2  # multiply by 2 because we need both mean and variance (adassl-v) or content and mask (adassl-s)

        if self.model == 'adassl-v':
            self.prior_net = build_mlp(sizes, act='relu', norm='bn', bias=True) # p_theta(r | x)
            initialize_mlp(self.prior_net, act='relu')

        sizes[0] *= 2
        self.latent_predictor = build_mlp(sizes, act='relu', norm='bn', bias=True)
        initialize_mlp(self.latent_predictor, act='relu')

    def update_reg_beta(self):
        # Update reg_beta with optional linear warmup 

        self.current_step += 0.5    # we call this function twice at every step
        
        if self.current_step < self.reg_beta_warmup_steps:
            # Linear warmup 
            self.reg_beta = self.reg_beta_target * math.ceil(self.current_step) / self.reg_beta_warmup_steps
        else:
            # Stay at final value after warmup
            self.reg_beta = self.reg_beta_target

    def forward(self, x, y):
        self.update_reg_beta()

        if self.model == 'adassl-v':
            sample, l_reg = self._sample_gaussian(x, y)
        elif self.model == 'adassl-s':
            sample, l_reg = self._sample_sparse(x, y)

        losses = dict(l_reg=self.reg_beta * l_reg)

        return sample, losses

    def _sample_gaussian(self, x, y):
        # AdaSSL-V uses standard variational approximation for p(z | x) and q(z | x, y)
        
        # Get mean and variance logits of the Gaussian prior and posterior
        prior = self.prior_net(x)
        posterior = self.latent_predictor(torch.cat([x, y], dim=1))

        # Split into mu and variance logits
        q_mu = posterior[:, :self.latent_dim]
        q_preact = posterior[:, self.latent_dim:]
        p_mu = prior[:, :self.latent_dim]
        p_preact = prior[:, self.latent_dim:]

        # Stability tricks
        q_sigma = F.softplus(q_preact) + 1e-6
        p_sigma = F.softplus(p_preact) + 1e-6
        q_logvar = 2 * torch.log(q_sigma)
        p_logvar = 2 * torch.log(p_sigma)
    
        sample = reparam_gaussian(q_mu, q_logvar)
        l_reg = kl_gaussian(q_mu, q_logvar, p_mu, p_logvar).sum(1).mean()

        return sample, l_reg

    def _sample_sparse(self, x, y):
        # AdaSSL-S directly predicts an unmasked latent r and samples a Bernoulli mask
        
        prediction = self.latent_predictor(torch.cat([x, y], dim=1))
        logits = prediction[:, :self.latent_dim]
        r_unmasked = prediction[:, self.latent_dim:]

        # Sample a Bernoulli mask from the logits
        mask = reparam_bernoulli(logits)
        sample = mask * r_unmasked

        # This implements the L0 regularization: E_i[\|r_{i, :}\|_0]
        l_reg = torch.sigmoid(logits).sum(1).mean()

        return sample, l_reg
