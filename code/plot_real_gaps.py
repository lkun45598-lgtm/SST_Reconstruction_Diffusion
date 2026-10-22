"""Fill REAL gaps of JAXA frames (no artificial masking, no ground truth -> qualitative only) and draw one summary figure.
Frames: library frames on held-out days (ordinal % 12 == 0, never trained on) with 25-55% of ocean pixels observed.
V = all genuine observations, E = all ocean; adapted model, K samples, one sampling run on the full 0.02 deg grid."""
import argparse, os, json, time, datetime as dt, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from data import CACHE, CloudLib
from edm import sample
from geo import patch_geometry
from evalutil import load_model
import plotstyle as ps

p = argparse.ArgumentParser()
p.add_argument('--run', required=True); p.add_argument('--out', required=True)
p.add_argument('--n-frames', type=int, default=4); p.add_argument('--k', type=int, default=4); p.add_argument('--steps', type=int, default=32)
p.add_argument('--seed', type=int, default=0); p.add_argument('--exclude', nargs='*', default=[], help='meta.json of earlier runs; their frames are skipped')
p.add_argument('--obs-range', nargs=2, type=float, default=[0.25, 0.55])
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
dev = 'cuda'; torch.manual_seed(11 + a.seed); rng = np.random.default_rng(a.seed)
lib = CloudLib(dev); land = lib.land.cpu().numpy(); ocean = ~land
S = np.load(f'{CACHE}/jaxa/lib_sst.npy', mmap_mode='r')
T = pd.to_datetime(pd.read_csv(f'{CACHE}/jaxa/lib_time.csv').time)
ordn = np.array([d.toordinal() for d in T.dt.date])
frac = np.array([(~np.isnan(np.asarray(S[i])))[ocean].mean() for i in range(len(S))])
ok = np.where((ordn % 12 == 0) & (frac >= a.obs_range[0]) & (frac <= a.obs_range[1]))[0]
print('candidates', len(ok), 'of', len(S))
used = {m['time'] for f in a.exclude for m in json.load(open(f))}
pick, months = [], set()
for i in rng.permutation(ok):
    if str(T[i]) in used: continue
    if T[i].month not in months: pick.append(int(i)); months.add(T[i].month)
    if len(pick) == a.n_frames: break
pick.sort(key=lambda i: T[i])
den, sc, step = load_model(a.run, dev); print('model step', step, 'scale', sc)
c = np.load(f'{CACHE}/jaxa_coords.npz'); lat = torch.from_numpy(c['lat']).float().to(dev)[None]; lon = torch.from_numpy(c['lon']).float().to(dev)[None]
geo = patch_geometry(lat, lon); ocean_t = torch.from_numpy(ocean).to(dev)[None, None]
rows, meta, cand = [], [], {}
for i in pick:
    x = torch.from_numpy(np.asarray(S[i]).copy()).to(dev)[None, None]
    V = (~torch.isnan(x)) & ocean_t
    torch.cuda.synchronize(); t0 = time.time()
    s = sample(den, x, V, ocean_t, geo, sc, a.k, a.steps)[0, :, 0].cpu().numpy() - 273.15   # (K,H,W) degC
    torch.cuda.synchronize(); dt_s = time.time() - t0
    obs = x[0, 0].cpu().numpy() - 273.15; Vn = V[0, 0].cpu().numpy()
    d = ((s[:, None] - s[None]) ** 2).reshape(a.k, a.k, -1).mean(-1).sum(1); rep = int(d.argmin())
    mean, std = s.mean(0), s.std(0)
    rows.append(dict(obs=obs, V=Vn, mean=mean, rep=s[rep], std=std, t=str(T[i]), frac=float(frac[i])))
    meta.append(dict(time=str(T[i]), observed_frac_of_ocean=float(frac[i]), seconds=dt_s, rms_std_in_gaps=float(np.sqrt((std[ocean & ~Vn] ** 2).mean()))))
    cand[f'f{len(rows)-1}_samples'] = s.astype('float16'); cand[f'f{len(rows)-1}_obs'] = obs.astype('float16')
    print(meta[-1], flush=True)
np.savez_compressed(f'{a.out}/candidates.npz', **cand); json.dump(meta, open(f'{a.out}/meta.json', 'w'), indent=1)

allobs = np.concatenate([r['obs'][r['V']] for r in rows]); vmin, vmax = np.percentile(allobs, [1, 99])
smax = max(0.5, float(np.ceil(max(np.percentile(r['std'][ocean & ~r['V']], 99) for r in rows) * 10) / 10))
n = len(rows); fig = plt.figure(figsize=(15.6, 4.15 * n + 0.3))
gs = fig.add_gridspec(n, 6, width_ratios=[1, 1, 1, 0.55, 1, 0.6], wspace=0.16, hspace=0.30, left=0.05, right=0.985, top=0.965, bottom=0.04)
axs = np.empty((n, 4), dtype=object)
for r_i, r in enumerate(rows):
    keep = lambda f: np.where(r['V'], r['obs'], f)                       # observed pixels kept, gaps from the model
    panels = [(('Observed input', f'{r["t"][:16]}, {r["frac"]:.0%} observed'), np.where(r['V'], r['obs'], np.nan), ps.sst_cmap(), vmin, vmax),
              (('Reconstruction', '(mean, observed kept)'), keep(r['mean']), ps.sst_cmap(), vmin, vmax),
              (('Reconstruction', '(sample, observed kept)'), keep(r['rep']), ps.sst_cmap(), vmin, vmax),
              (('Ensemble std', '(unobserved pixels, K=4)'), np.where(~r['V'] & ocean, r['std'], np.nan), ps.err_cmap(), 0, smax)]
    for c_i, ((l1, l2), fld, cm, lo, hi) in enumerate(panels):
        ax = fig.add_subplot(gs[r_i, [0, 1, 2, 4][c_i]], sharex=axs[0, 0] if (r_i or c_i) else None, sharey=axs[0, 0] if (r_i or c_i) else None)
        axs[r_i, c_i] = ax; ps.gray_base(ax, land); im = ps.draw(ax, fld, land, cm, lo, hi)
        ps.title(ax, l1, l2, size=11.5); ps.style_axes(ax, bottom=(r_i == n - 1), left=(c_i == 0))
        if c_i == 0: im_sst = im
        if c_i == 3: im_std = im
ps.cbar_slot(fig, gs[:, 3], im_sst, 'SST (°C)', step=1.0, box=(0.12, 0.25, 0.2, 0.5))
ps.cbar_slot(fig, gs[:, 5], im_std, 'Ensemble std (°C)', step=0.1 if smax <= 0.6 else 0.2, box=(0.12, 0.25, 0.18, 0.5))
fig.savefig(f'{a.out}/realgap_summary.png', dpi=200); plt.close(fig); print('saved', a.out)
