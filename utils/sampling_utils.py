
import torch


def kl_gaussian(q_mu, q_logvar, p_mu, p_logvar):
    # KL divergence between two factorized Gaussians
    q_logvar = q_logvar.clamp(min=-10.0, max=10.0)
    p_logvar = p_logvar.clamp(min=-10.0, max=10.0)
    var_x = q_logvar.exp()
    var_y = p_logvar.exp()
    kl = 0.5 * (p_logvar - q_logvar + var_x / var_y + (q_mu - p_mu) ** 2 / var_y - 1)
    return kl

def reparam_gaussian(mu, logvar):
    # Reparameterization trick for Gaussian
    logvar = logvar.clamp(min=-10.0, max=10.0)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

def reparam_bernoulli(logits, tau=1.0):
    # Reparameterization trick for Bernoulli (Gumbel noise + straight-through estimator)
    u = torch.rand_like(logits).clamp_(1e-6, 1-1e-6)
    g = torch.log(u) - torch.log1p(-u)          # Logistic(0,1)
    y_soft = torch.sigmoid((logits + g) / tau)
    mask = (y_soft > 0.5).to(y_soft.dtype)
    mask = mask.detach() - y_soft.detach() + y_soft
    return mask 