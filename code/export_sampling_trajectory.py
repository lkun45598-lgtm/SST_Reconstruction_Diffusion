"""Export the REAL default sampling trajectory (EDM Heun, no reinject/resample) of the adapted single-frame model on one real
JAXA frame with its genuine cloud gaps (V = observed ocean pixels, E = all ocean, same setup as plot_real_gaps.py).
Writes <out>/trajectory.npz (every step) and per-panel .npy + .png (project SST style, dpi 300) for the figure panels."""
import argparse, os, json, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from data import CloudLib, CACHE
from edm import sample
from geo import patch_geometry
from evalutil import load_model
import plotstyle as ps

p = argparse.ArgumentParser()
p.add_argument('--run', default='../runs/jaxa_adapt'); p.add_argument('--out', default='../runs/sampling_trajectory')
p.add_argument('--time', default='2020-02-05 16:00:00'); p.add_argument('--steps', type=int, default=32)
p.add_argument('--show-step', type=int, default=None, help='step k for the denoiser / updated-state panels (default: first step with sigma<=1)')
p.add_argument('--seed', type=int, default=0)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
dev = 'cuda'
lib = CloudLib(dev); land = lib.land.cpu().numpy(); ocean = ~land
S = np.load(f'{CACHE}/jaxa/lib_sst.npy', mmap_mode='r'); T = pd.to_datetime(pd.read_csv(f'{CACHE}/jaxa/lib_time.csv').time)
i = int(np.where(T == pd.Timestamp(a.time))[0][0])
den, sc, ckstep = load_model(a.run, dev)
c = np.load(f'{CACHE}/jaxa_coords.npz'); geo = patch_geometry(torch.from_numpy(c['lat']).float().to(dev)[None], torch.from_numpy(c['lon']).float().to(dev)[None])
ocean_t = torch.from_numpy(ocean).to(dev)[None, None]
x = torch.from_numpy(np.asarray(S[i]).copy()).to(dev)[None, None]; V = (~torch.isnan(x)) & ocean_t
gen = torch.Generator(device=dev).manual_seed(a.seed)
out, tr = sample(den, x, V, ocean_t, geo, sc, 1, a.steps, generator=gen, return_trajectory=True, save_dir=a.out)
sig = tr['sigma']; k = a.show_step if a.show_step is not None else int(np.argmax(sig[:-1] <= 1.0))
print(f'frame {a.time} ckpt step {ckstep} scale {sc} observed {V.float().sum()/ocean_t.float().sum():.1%} of ocean; show step k={k}: sigma {sig[k]:.3f} -> {sig[k+1]:.3f}')
print('sigma schedule', np.round(sig, 3).tolist())

kel = lambda f: f * tr['scale'] + tr['mu'] - 273.15              # normalised -> degC
ki = int(np.where(tr['steps'] == k)[0][0])
Vm = tr['V']; obs = np.where(Vm, x[0, 0].cpu().numpy() - 273.15, np.nan)
panels = dict(input_sst=obs, visible_mask=Vm.astype('float32'), ocean_mask=tr['E'].astype('float32'),
              init_state=tr['x_init'], denoiser_pred=kel(tr['den'][ki]), updated_state=tr['x_next'][ki], final_sst=kel(tr['x_final']))
for n_, v in panels.items(): np.save(f'{a.out}/{n_}.npy', v)
np.save(f'{a.out}/land.npy', land)
json.dump(dict(time=a.time, ckpt=a.run, ckpt_step=int(ckstep), steps=a.steps, k=k, sigma_k=float(sig[k]), sigma_next=float(sig[k + 1]),
               sigma_max=float(sig[0]), sigma_min=float(sig[-2]), mu_K=tr['mu'], scale=tr['scale'], observed_frac=float(Vm[ocean].mean())),
          open(f'{a.out}/meta.json', 'w'), indent=1)

# per-panel PNGs (SST style). SST panels share one range; diffusion states are in normalised units with their own symmetric range.
vmin, vmax = np.percentile(obs[Vm], [1, 99])
fig_kw = dict(figsize=(4.2, 5.4), constrained_layout=True)
def save(name, fld, cmap, lo, hi, l1, l2, cblabel):
    fig, ax = plt.subplots(**fig_kw); ps.gray_base(ax, land); im = ps.draw(ax, fld, land, cmap, lo, hi)
    ps.title(ax, l1, l2, size=11); ps.style_axes(ax, True, True); fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.03, aspect=25).set_label(cblabel, fontsize=10)
    fig.savefig(f'{a.out}/{name}.png', dpi=300); plt.close(fig)
from matplotlib.colors import ListedColormap
def savemask(name, m, l1, l2):
    fig, ax = plt.subplots(**fig_kw); ax.imshow(np.where(land, np.nan, 1.0), extent=ps.EXTENT, origin='upper', interpolation='nearest', cmap=ListedColormap(['black']), vmin=0, vmax=1)
    ax.imshow(np.where(m > 0.5, 1.0, np.nan), extent=ps.EXTENT, origin='upper', interpolation='nearest', cmap=ListedColormap(['white']), vmin=0, vmax=1)
    ax.imshow(np.where(land, 1.0, np.nan), extent=ps.EXTENT, origin='upper', interpolation='nearest', cmap=ListedColormap([ps.LAND]), vmin=0, vmax=1)
    ps.title(ax, l1, l2, size=11); ps.style_axes(ax, True, True); fig.savefig(f'{a.out}/{name}.png', dpi=300); plt.close(fig)
save('input_sst', obs, ps.sst_cmap(), vmin, vmax, 'Input SST', '(observed pixels)', 'SST (°C)')
savemask('visible_mask', Vm, 'Observation mask V', '(white: observed)'); savemask('ocean_mask', tr['E'], 'Ocean domain E', '(white: generated)')
r0 = 3 * sig[0]; r1 = max(3 * sig[k + 1], 1.5)
save('init_state', np.where(tr['E'], tr['x_init'], np.nan), ps.sst_cmap(), -r0, r0, f'Initial noise state', f'(x_σ, σ={sig[0]:.0f})', 'normalised state')
save('denoiser_pred', np.where(tr['E'], panels['denoiser_pred'], np.nan), ps.sst_cmap(), vmin, vmax, f'Denoiser prediction D', f'(step {k+1}, σ={sig[k]:.2f})', 'SST (°C)')
save('updated_state', np.where(tr['E'], tr['x_next'][ki], np.nan), ps.sst_cmap(), -r1, r1, 'Heun update', f'(x_σ, σ={sig[k+1]:.2f})', 'normalised state')
save('final_sst', np.where(tr['E'], panels['final_sst'], np.nan), ps.sst_cmap(), vmin, vmax, 'Final reconstruction', '(raw network output)', 'SST (°C)')
print('saved', a.out)
