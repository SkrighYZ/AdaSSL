

import numpy as np
from sklearn.linear_model import LassoCV
import pickle
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loading_utils import get_standard_loader
from tqdm import tqdm
import gc
from utils.eval_utils import load_pretrained_models

def z_score(a):
    mu = a.mean(0, keepdims=True)
    sd = a.std(0, keepdims=True)
    sd = np.maximum(sd, 1e-12) 
    return (a - mu) / sd

def dci_disentanglement(X, Y):
    # https://openreview.net/forum?id=By-7dz-AZ
    # DCI disentanglement (entropy over factors per code, weighted by code importance)

    print('Calculating importance matrix...')
    R = get_importance_matrix(X, Y)

    print('Calculating disentanglement...')
    nonzero = (R.sum(1) > 0)
    P = np.zeros_like(R)
    P[nonzero] = R[nonzero] / R.sum(1, keepdims=True)[nonzero]     # Define P_i only when valid

    D = np.zeros(R.shape[0])
    D[nonzero] = 1.0 - entropy(P[nonzero].T + 1e-12, base=R.shape[1], axis=0)

    rho = R.sum(1) / R.sum()       
    disent = (rho * D).sum()
    return disent


def get_importance_matrix(X, Y):
    # We use Lasso regressor weights as the importance matrix R

    X = z_score(X)
    Y = z_score(Y)

    X = np.asarray(X, dtype=np.float64, order='C')
    Y = np.asarray(Y, dtype=np.float64, order='C')

    coefs = np.zeros((X.shape[1], Y.shape[1]))

    for j in range(Y.shape[1]):
        lcv = LassoCV(cv=5, 
                      random_state=0, 
                      max_iter=50000, 
                      precompute=False,
                      n_jobs=1).fit(X, Y[:, j])
        coefs[:, j] = lcv.coef_

    R = np.abs(coefs)  # (d, k)

    return R



@torch.no_grad()
def get_data(loader, model):
    # Z: groud-truth factors
    # Z_emb: model's embeddings
    Z = []
    Z_emb = []
    for x, z in tqdm(loader, total=len(loader), desc='Encoding data'):
        Z.append(z.numpy())
        Z_emb.append(F.normalize(model(x.cuda(non_blocking=True)), p=2, dim=1).cpu().numpy())
    
    Z = np.concatenate(Z)
    Z_emb = np.concatenate(Z_emb)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 

    return Z_emb, Z

def get_r2(Z_emb, Z, Z_emb_test, Z_test):
    model = LinearRegression(n_jobs=-1)
    print('Fitting linear regression model...')
    model.fit(Z_emb, Z)
    print('Predicting...')
    Z_pred = model.predict(Z_emb_test)
    print('Calculating R2 score...')
    r2 = r2_score(Z_test, Z_pred, multioutput='raw_values')
    return r2


def run_eval_3dident(args):
    train_loader = get_standard_loader(args, split='train', train=True)
    test_loader = get_standard_loader(args, split='test', train=False)

    results = dict()

    backbone, projector = load_pretrained_models(args)
    model = nn.Sequential(backbone, projector)

    Z_emb, Z = get_data(train_loader, model)
    Z_emb_test, Z_test = get_data(test_loader, model)

    disent = dci_disentanglement(Z_emb_test, Z_test)
    print('dci disentanglement:', disent)
    
    r2 = get_r2(Z_emb, Z, Z_emb_test, Z_test)
    print('r2:', r2.mean())

    results['disent'] = disent
    results['r2'] = r2.mean()

    pickle.dump(results, open(args.save_dir / f'test_logs.pkl', 'wb'))







