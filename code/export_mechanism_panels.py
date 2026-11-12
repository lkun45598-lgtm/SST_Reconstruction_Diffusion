"""Export every panel of the mechanism figures as its own image (real data from export_g2_mechanism.py).
<out>/clean/*.png|pdf : map only (no title/axes), 300 dpi, tight; <out>/labeled/*.png|pdf : title, lat/lon axes, colorbar; <out>/colorbars/*."""
import argparse, json, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
import plotstyle as ps

p = argparse.ArgumentParser(); p.add_argument('--dir', default='../runs/mechanism_g2'); p.add_argument('--out-dir', default='../figures/mechanism_g2/panels')
a = p.parse_args(); d = a.dir
T = np.load(f'{d}/trajectory.npz'); Hs = np.load(f'{d}/history.npz'); m = json.load(open(f'{d}/meta.json')); land = np.load(f'{d}/land.npy')
E, V, y = T['E'], T['V'], T['y']; obs = np.load(f'{d}/target_obs.npy'); K, U = m['k'], m['U']; sg = T['sigma']; s0, s1 = float(sg[K]), float(sg[K + 1])
ev = lambda nm, u=0: T[f'ev_{K}_{u}_{nm}']; kel = lambda f: f * float(T['scale']) + float(T['mu']) - 273.15
vmin, vmax = np.percentile(obs[V], [1, 99]); R = 3 * s0; LB = '#BFD8EE'
for sub in ('clean', 'labeled', 'colorbars'): os.makedirs(f'{a.out_dir}/{sub}', exist_ok=True)
SST = ('SST (°C)', vmin, vmax); ST = ('normalised state', -R, R)
lab = [f't-{11 - i}h' for i in range(11)]
P = [  # name, title l1, l2, field, kind
 ('input_sst', 'Input SST', f'{m["time"][:16]}, {m["observed_frac"]:.0%} observed', obs, 'sstraw', SST),
 ('visible_values_yV', 'Visible values y·V', 'normalised, 0 elsewhere', np.where(V, y, np.nan), 'state', ST),
 ('mask_V', 'Observation mask V', 'white: observed, black: missing', V & E, 'V', None),
 ('domain_E', 'Ocean domain E', 'light blue: generated', E, 'E', None),
 *[(f'history_{lab[i]}', f'History {lab[i]}', f'{m["hist_obs_frac"][i]:.0%} observed', Hs['hist'][i], 'sstraw', SST) for i in range(11)],
 ('initial_noise_state', 'Initial state (before re-injection)', f'sigma_0={sg[0]:.0f}', T['ev_x_init_raw'], 'stateN', ('normalised state', -3 * sg[0], 3 * sg[0])),
 ('state_x_sigma_i', 'Current state x_sigma_i', f'sigma_i={s0:.2f}', ev('x_in', 0), 'state', ST),
 ('denoised_D', 'Denoised estimate D(x_sigma_i)', 'de-normalised', kel(ev('den', 0)), 'sst', SST),
 ('euler_predictor_xtilde', 'Euler predictor x~', f'at sigma_i+1={s1:.2f}', ev('x_euler', 0), 'state', ST),
 ('denoised_D_second_eval', 'Denoised estimate D(x~)', '2nd evaluation, de-normalised', kel(ev('den2', 0)), 'sst', SST),
 ('heun_corrected_state', 'After Heun correction', f'sigma_i+1={s1:.2f}, before re-injection', ev('x_heun', 0), 'state', ST),
 ('injected_value_on_V', 'Injected value on V', 'y + sigma_i+1 * eps', np.where(V, ev('x_next', 0), np.nan), 'state', ST),
 ('state_after_reinjection', 'After re-injection', f'x_sigma_i+1, sigma_i+1={s1:.2f}', ev('x_next', 0), 'state', ST),
 ('jump_back_noise', 'Jump-back noise', 'sqrt(sigma_i^2 - sigma_i+1^2) * eps', ev('x_jump_noised', 0) - ev('x_next', 0), 'state', ST),
 ('state_after_jump_noised', 'After jump back (before re-injection)', f'sigma_i={s0:.2f}', ev('x_jump_noised', 0), 'state', ST),
 ('state_after_jump_reinjected', 'After jump back + re-injection', f'sigma_i={s0:.2f}', ev('x_jump', 0), 'state', ST),
 ('state_after_traversal_2', 'After traversal 2', f'sigma_i+1={s1:.2f}', ev('x_next', U - 1), 'state', ST),
 ('final_sst', 'Final reconstruction', 'sigma=0, all levels', kel(T['x_final']), 'sst', SST)]
def draw(ax, fld, kind, cb):
    if kind in ('V', 'E'):
        for f_, c_ in ([(np.where(~land, 1., np.nan), 'black'), (np.where(fld, 1., np.nan), 'white')] if kind == 'V' else [(np.where(fld, 1., np.nan), LB)]) + [(np.where(land, 1., np.nan), ps.LAND)]:
            ax.imshow(f_, extent=ps.EXTENT, origin='upper', interpolation='nearest', cmap=ListedColormap([c_]), vmin=0, vmax=1)
        return None
    ps.gray_base(ax, land); f_ = fld if kind == 'sstraw' else np.where(E, fld, np.nan)
    return ps.draw(ax, f_, land, ps.sst_cmap(), cb[1], cb[2])
for name, l1, l2, fld, kind, cb in P:
    fig = plt.figure(figsize=(2.6, 3.34)); ax = fig.add_axes([0, 0, 1, 1]); draw(ax, fld, kind, cb); ax.set_aspect('auto')
    ax.set_xlim(ps.EXTENT[0], ps.EXTENT[1]); ax.set_ylim(ps.EXTENT[2], ps.EXTENT[3]); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(.8)
    for e in ('png', 'pdf'): fig.savefig(f'{a.out_dir}/clean/{name}.{e}', dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.4, 5.6), constrained_layout=True); im = draw(ax, fld, kind, cb)
    ps.style_axes(ax, True, True); ax.tick_params(labelsize=9); ax.set_title(f'{l1}\n{l2}', fontsize=10.5, fontweight='bold', linespacing=1.35)
    if im is not None:
        c = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=.8, pad=.03, aspect=25); c.set_label(cb[0], fontsize=9.5)
    for e in ('png', 'pdf'): fig.savefig(f'{a.out_dir}/labeled/{name}.{e}', dpi=300)
    plt.close(fig)
for nm, (lb, lo, hi) in (('colorbar_sst_degC', SST), ('colorbar_state_normalised', ST)):
    fig, ax = plt.subplots(figsize=(3.6, .6)); sm = plt.cm.ScalarMappable(cmap=ps.sst_cmap(), norm=plt.Normalize(lo, hi)); c = fig.colorbar(sm, cax=ax, orientation='horizontal'); c.set_label(lb, fontsize=10)
    for e in ('png', 'pdf'): fig.savefig(f'{a.out_dir}/colorbars/{nm}.{e}', dpi=300, bbox_inches='tight')
    plt.close(fig)
print(len(P), 'panels ->', a.out_dir)
