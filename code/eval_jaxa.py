"""Step 7: reconstruct artificially hidden GENUINE JAXA observations on the full 0.02 deg grid.

Frames: held-out test days only (data.split_of). For each frame K = genuine observations. A real cloud
shape P (from a different JAXA frame) hides H = K & P; the network sees V = K & ~P and generates the
complete ocean field E (true gaps included) in one sampling run. Metrics only on H (true values).
Nothing filled by KNN/proxy/FNO is used anywhere.
"""
import argparse, json, time, numpy as np, torch
from data import Jaxa, CloudLib
from edm import sample
from geo import patch_geometry
from evalutil import load_model, laplace_fill, hole_metrics, summarize, boot_ratio, select_items

p = argparse.ArgumentParser()
p.add_argument('--runs', nargs='+', required=True, help='name=run_dir, e.g. pretrain=.../ostia_full jaxa=.../jaxa_adapt')
p.add_argument('--out', required=True); p.add_argument('--n-frames', type=int, default=30)
p.add_argument('--k', type=int, default=4); p.add_argument('--steps', type=int, default=32)
a = p.parse_args()
import os; os.makedirs(a.out, exist_ok=True)
dev = 'cuda'; torch.manual_seed(7); rng = np.random.default_rng(7)
lib = CloudLib(dev); jx = Jaxa(dev, lib)
models = {kv.split('=')[0]: load_model(kv.split('=')[1], dev) for kv in a.runs}
print({k: (v[1], v[2]) for k, v in models.items()})
N, Hh, Ww = jx.x.shape
geo = patch_geometry(jx.lat[None], jx.lon[None]); ocean = ~lib.land[None, None]

# frame / cloud-shape selection (fixed seed, identical for every method; shared with plot_jaxa_frames.py)
items = select_items(jx, lib, ocean, a.n_frames, seed=7)
print('frames', len(items), 'test days of', len(jx.idx['test']), 'test frames available')

res = {k: [] for k in list(models) + ['laplace']}; resm = {k: [] for k in models}
spread = {k: [] for k in models}; runtime = {k: [] for k in models}; saved = {}
for fi, (f, K, H_, V) in enumerate(items):
    x = jx.x[f][None, None]; truth = torch.nan_to_num(x)
    res['laplace'].append(hole_metrics(laplace_fill(x, V, ocean), truth, H_))
    for name, (den, sc, _) in models.items():
        torch.cuda.synchronize(); t = time.time()
        s = sample(den, x, V, ocean, geo, sc, a.k, a.steps)[0]            # (n,1,H,W)
        torch.cuda.synchronize(); runtime[name].append(time.time() - t)
        res[name].append(hole_metrics(s[:1], truth, H_)); resm[name].append(hole_metrics(s.mean(0, keepdim=True), truth, H_))
        sd = s.std(0, keepdim=True) ; spread[name].append(float((sd ** 2 * H_.float()).sum() / H_.float().sum()))
        if fi < 6:
            d = torch.cdist(s.flatten(1), s.flatten(1)).sum(1); rep = int(d.argmin())
            saved[(name, fi)] = dict(samples=s[:, 0].cpu().numpy(), mean=s.mean(0)[0].cpu().numpy(), std=s.std(0)[0].cpu().numpy(), rep=rep)
cat = lambda L: {kk: np.concatenate([d[kk] for d in L]) for kk in L[0]}
out = dict(n_frames=len(items), k=a.k, steps=a.steps, single_sample={k: summarize(cat(v)) for k, v in res.items()},
           ensemble_mean={k: summarize(cat(v)) for k, v in resm.items()},
           rms_spread_on_hidden={k: float(np.sqrt(np.mean(v))) for k, v in spread.items()},
           seconds_per_frame={k: float(np.mean(v)) for k, v in runtime.items()})
names = list(models)
for n in names:
    out[f'{n}_ensmean_vs_laplace'] = boot_ratio(cat(resm[n]), cat(res['laplace']))
if len(names) > 1:
    out[f'{names[-1]}_vs_{names[0]}_ensmean'] = boot_ratio(cat(resm[names[-1]]), cat(resm[names[0]]))
json.dump(out, open(f'{a.out}/metrics.json', 'w'), indent=1); print(json.dumps(out, indent=1))

# figures: run plot_jaxa_frames.py (project SST map style); no plotting here on purpose
np.savez_compressed(f'{a.out}/candidates.npz', **{f'{n}_f{fi}_{k}': v for (n, fi), d in saved.items() for k, v in d.items() if k != 'rep'},
                    **{f'{n}_f{fi}_rep': np.array(d['rep']) for (n, fi), d in saved.items()})
print('saved to', a.out)
