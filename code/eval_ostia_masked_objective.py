"""Step 5: does the masked-data objective recover what the full-field objective learns?

Held-out OSTIA val days, native 0.05 deg 64x64 patches. Evaluation regime is the real inference regime:
E = all ocean (nothing removed from the diffusion state), visible = ocean minus a synthetic cloud set C,
metric on C pixels where OSTIA truth exists. The models were trained as:
  F  ostia_full    : full-field objective on complete OSTIA (reference / upper bound)
  M  ostia_masked  : never sees labels inside unknown set A (A removed from the diffusion state)
  N  ostia_naive   : INVALID control, A filled with visible mean and used as labels
PASS CRITERION (fixed before looking at results): RMSE_M(ensemble mean) <= 1.10 * RMSE_F  AND  RMSE_M < RMSE_N.
"""
import json, sys, numpy as np, torch
from data import Ostia, CloudLib
from edm import sample
from evalutil import load_model, laplace_fill, hole_metrics, summarize, boot_ratio

R = '/data1/user/lz/SST_Diffusion/runs'; dev = 'cuda'
NP, NS, STEPS, BS = 256, 4, 32, 32
torch.manual_seed(2026)
lib, ostia = CloudLib(dev), Ostia(dev, years=(2010, 2014))  # same years the pilot models were trained on
models = {k: load_model(f'{R}/pilot_ostia_{k}', dev) for k in ('full', 'masked', 'naive')}
print({k: (v[1], v[2]) for k, v in models.items()})
res = {k: [] for k in ['F', 'M', 'N', 'laplace', 'mean']}; resm = {k: [] for k in ['F', 'M', 'N']}
done = 0
while done < NP:
    x, ocean, g = ostia.batch(BS, 'test', 64)
    C = lib.crops(BS, 160, 64)
    V, H_ = ocean & ~C, ocean & C
    ok = (H_.float().sum((1, 2, 3)) >= 8) & (V.float().sum((1, 2, 3)) >= 8)
    k = ok.nonzero().flatten()
    x, V, H_, ocean = x[k], V[k], H_[k], ocean[k]; g = dict(coords=g['coords'][k], spacing=g['spacing'][k], extent=g['extent'])
    truth = torch.nan_to_num(x)
    for name, key in (('full', 'F'), ('masked', 'M'), ('naive', 'N')):
        den, sc, _ = models[name]
        s = sample(den, x, V, ocean, g, sc, NS, STEPS)                   # (B,n,1,H,W)
        res[key].append(hole_metrics(s[:, 0], truth, H_)); resm[key].append(hole_metrics(s.mean(1), truth, H_))
    res['laplace'].append(hole_metrics(laplace_fill(x, V, ocean), truth, H_))
    mu = (truth * V.float()).sum((1, 2, 3), keepdim=True) / V.float().sum((1, 2, 3), keepdim=True)
    res['mean'].append(hole_metrics(mu.expand_as(truth), truth, H_))
    done += len(k); print('patches', done, flush=True)
cat = lambda L: {kk: np.concatenate([d[kk] for d in L]) for kk in L[0]}
out = {'single_sample': {k: summarize(cat(v)) for k, v in res.items()},
       'ensemble_mean_K4': {k: summarize(cat(v)) for k, v in resm.items()}}
F_, M_, N_ = cat(resm['F']), cat(resm['M']), cat(resm['N'])
out['M_vs_F_ensmean'] = boot_ratio(M_, F_); out['N_vs_F_ensmean'] = boot_ratio(N_, F_); out['M_vs_N_ensmean'] = boot_ratio(M_, N_)
r = out['ensemble_mean_K4']
out['pass'] = bool(r['M']['rmse'] <= 1.10 * r['F']['rmse'] and r['M']['rmse'] < r['N']['rmse'])
out['n_patches'] = int(len(F_['n']))
json.dump(out, open(f'{R}/pilot_masked_objective_validation.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
