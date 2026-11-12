"""Single-step sampling mechanism figure (EDM / U-FNO, Heun) built from the REAL trajectory written by
export_sampling_trajectory.py (runs/sampling_trajectory/trajectory.npz + meta.json). Nothing is synthetic.
Faithful to edm.sample default path: condition c=[y*V, V, E, coords(5)] is only fed to the network; no re-injection of observations."""
import argparse, json, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.colors import ListedColormap
import plotstyle as ps

p = argparse.ArgumentParser()
p.add_argument('--traj-dir', default='../runs/sampling_trajectory'); p.add_argument('--out-dir', default='../figures')
p.add_argument('--step', type=int, default=None, help='0-based step k (default: the k stored in meta.json)')
a = p.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
T = np.load(f'{a.traj_dir}/trajectory.npz'); m = json.load(open(f'{a.traj_dir}/meta.json')); land = np.load(f'{a.traj_dir}/land.npy')
k = m['k'] if a.step is None else a.step; ki = int(np.where(T['steps'] == k)[0][0]); sig = T['sigma']; s0, s1 = float(sig[k]), float(sig[k + 1])
assert s1 > 0, 'last step (sigma_next=0) has no 2nd evaluation; pick an earlier step'
E, V = T['E'], T['V']; kel = lambda f: f * float(T['scale']) + float(T['mu']) - 273.15
obs = np.where(V, np.load(f'{a.traj_dir}/input_sst.npy'), np.nan); vmin, vmax = np.percentile(obs[V], [1, 99])
xi, den, xn = T['x_in'][ki], kel(T['den'][ki]), T['x_next'][ki]; r = 3 * s0        # same symmetric range for x_{sigma_i} and x_{sigma_{i+1}}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
W, H = 16.0, 9.0; fig = plt.figure(figsize=(W, H)); BLUE, GREEN, ORNG = '#34588F', '#4C9A5B', '#C5672B'
fx = lambda x: x / W; fy = lambda y: y / H
def axin(x, y, w, h): return fig.add_axes([fx(x), fy(y), fx(w), fy(h)])
MW, MH = 1.75, 2.25
def sst_panel(x, y, tag, l1, l2, fld, lo, hi, cmap, cb, left=False):
    ax = axin(x, y, MW, MH); ps.gray_base(ax, land) if cb == 'SST (°C)' else ax.set_facecolor('white')
    im = ps.draw(ax, np.where(E, fld, np.nan), land, cmap, lo, hi); deco(ax, tag, l1, l2, left)
    c = fig.colorbar(im, cax=axin(x + .15, y - .42, MW - .3, .12), orientation='horizontal'); c.set_label(cb, fontsize=10, labelpad=2); c.ax.tick_params(labelsize=9)
    return ax
def deco(ax, tag, l1, l2, left):
    ps.style_axes(ax, False, left); ax.set_xlabel(''); ax.set_ylabel(''); ax.set_xticks([]); ax.tick_params(labelsize=9)
    if left: ax.set_yticks([15, 19, 23])
    else: ax.set_yticks([])
    ax.set_title(l1 + ('\n' + l2 if l2 else ''), fontsize=11.5, fontweight='bold', linespacing=1.35)
    ax.text(.03, .985, tag, transform=ax.transAxes, ha='left', va='top', fontsize=13, fontweight='bold', bbox=dict(fc='white', ec='none', pad=1.5, alpha=.85))
def mask_panel(x, y, tag, l1, l2, layers, handles, left=False):
    ax = axin(x, y, MW, MH)
    for fld, col in layers: ax.imshow(fld, extent=ps.EXTENT, origin='upper', interpolation='nearest', cmap=ListedColormap([col]), vmin=0, vmax=1)
    deco(ax, tag, l1, l2, left)
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.03), ncol=2, fontsize=9, frameon=False, handlelength=1.2, columnspacing=1, handletextpad=.4)
OCEAN_BLUE = '#BFD8EE'
n_ = lambda msk: np.where(msk, 1., np.nan)
def box(x, y, w, h, fc, ec, text, fs=11.5, tc='k', bold=True):
    fig.add_artist(FancyBboxPatch((fx(x), fy(y)), fx(w), fy(h), boxstyle='round,pad=0,rounding_size=0.012', transform=fig.transFigure, fc=fc, ec=ec, lw=1.6))
    fig.text(fx(x + w / 2), fy(y + h / 2), text, ha='center', va='center', fontsize=fs, color=tc, fontweight='bold' if bold else 'normal', linespacing=1.5)
def arrow(p0, p1, color='#222', lw=1.8, style='-|>', ls='-', rad=0):
    fig.add_artist(FancyArrowPatch((fx(p0[0]), fy(p0[1])), (fx(p1[0]), fy(p1[1])), transform=fig.transFigure, arrowstyle=style, mutation_scale=18, lw=lw, color=color, ls=ls, connectionstyle=f'arc3,rad={rad}'))
def line(xs, ys, color='#222', lw=1.8): fig.add_artist(plt.Line2D([fx(v) for v in xs], [fy(v) for v in ys], transform=fig.transFigure, color=color, lw=lw))
# ---- top: conditions
y_top = 5.8
sst_panel(.95, y_top, '(a)', 'Input SST', f'({m["observed_frac"]:.0%} of ocean observed)', obs, vmin, vmax, ps.sst_cmap(), 'SST (°C)', left=True)
mask_panel(3.15, y_top, '(b)', 'Observation mask $V$', '(visible SST pixels)', [(n_(~land), 'black'), (n_(V & E), 'white'), (n_(land), ps.LAND)],
           [Patch(fc='white', ec='k', lw=.6, label='observed'), Patch(fc='black', label='missing (ocean)'), ], )
mask_panel(5.35, y_top, '(c)', 'Ocean domain $E$', '(generation domain)', [(n_(E), OCEAN_BLUE), (n_(land), ps.LAND)],
           [Patch(fc=OCEAN_BLUE, ec='k', lw=.6, label='ocean (in $E$)'), Patch(fc=ps.LAND, ec='k', lw=.6, label='land')])
# condition node c
yc = y_top + MH / 2; bx, bw = 8.25, 2.3
fig.add_artist(FancyBboxPatch((fx(.45), fy(4.85)), fx(7.2), fy(3.8), boxstyle='round,pad=0,rounding_size=0.01', transform=fig.transFigure, fc='none', ec=BLUE, lw=1.3, ls='--'))
fig.text(fx(7.55), fy(5.05), 'Condition inputs', ha='right', va='center', fontsize=11, color=BLUE, fontweight='bold')
box(bx, yc - .8, bw, 1.6, '#EEF3FA', BLUE, 'condition  $c$\n$[\\,y\\!\\cdot\\!V,\\ V,\\ E,\\ \\mathrm{coords}\\,]$\n(8 channels)', fs=11.5, tc=BLUE)
arrow((7.65, yc), (bx - .02, yc), BLUE)
# ---- bottom: one Heun step
y_bot = 1.6; yb = y_bot + MH / 2
sst_panel(.95, y_bot, '(d)', 'Current diffusion state', f'$x_{{\\sigma_i}}$, $\\sigma_i={s0:.2f}$', xi, -r, r, ps.sst_cmap(), 'normalised state (dimensionless)', left=True)
box(3.5, yb - .6, 1.7, 1.2, '#FBEBDD', ORNG, 'U-FNO\nDenoiser', fs=13, tc=ORNG)
fig.text(fx(4.35), fy(yb - .85), 'inputs: $x_{\\sigma_i},\\ \\sigma_i,\\ c$', ha='center', va='center', fontsize=10.5)
arrow((2.78, yb), (3.47, yb)); arrow((5.22, yb), (5.87, yb))
sst_panel(5.9, y_bot, '(e)', 'Denoiser prediction', f'$D(x_{{\\sigma_i}},\\sigma_i;c)$', den, vmin, vmax, ps.sst_cmap(), 'SST (°C), de-normalised')
box(8.5, yb - .95, 2.4, 1.9, '#EAF5EC', GREEN, 'Heun\nPredictor–Corrector', fs=12.5, tc=GREEN)
fig.text(fx(9.7), fy(yb - .6), '① predictor: Euler step to $\\widetilde{x}$\n② 2nd call $D(\\widetilde{x},\\sigma_{i+1};c)$\n③ corrector: average slopes', ha='center', va='center', fontsize=9.5, linespacing=1.6)
fig.text(fx(9.7), fy(yb - 1.25), '2 network evaluations / step', ha='center', va='center', fontsize=10.5)
arrow((7.68, yb), (8.47, yb)); arrow((10.93, yb), (11.72, yb))
sst_panel(11.75, y_bot, '(f)', 'Next diffusion state', f'$x_{{\\sigma_{{i+1}}}}$, $\\sigma_{{i+1}}={s1:.2f}$', xn, -r, r, ps.sst_cmap(), 'normalised state (dimensionless)')
# condition -> denoiser (and, by the 2nd call, also at x~): enters from the top
line([bx + bw / 2, bx + bw / 2], [yc - .8, 4.5], BLUE); line([bx + bw / 2, 4.35], [4.5, 4.5], BLUE); arrow((4.35, 4.5), (4.35, yb + .62), BLUE)
fig.text(fx(4.5), fy(4.56), 'condition $c$ (same at every call)', ha='left', va='bottom', fontsize=10, color=BLUE)
# loop back
line([12.625, 12.625], [.72, .5], '#777'); line([12.625, 1.825], [.5, .5], '#777'); arrow((1.825, .5), (1.825, .72), '#777')
fig.text(fx(7.2), fy(.42), 'Repeat until the final noise level ($\\sigma_{i+1}\\!\\to\\!0$): $x_{\\sigma_{i+1}}$ is the input state of the next step', ha='center', va='top', fontsize=11, color='#555')
# ---- formulas panel (top right)
fx0, fw = 11.0, 4.8
fig.add_artist(FancyBboxPatch((fx(fx0), fy(5.0)), fx(fw), fy(3.5), boxstyle='round,pad=0,rounding_size=0.01', transform=fig.transFigure, fc='white', ec=GREEN, lw=1.4))
fig.text(fx(fx0 + fw / 2), fy(8.2), 'Heun update (EDM)', ha='center', va='center', fontsize=12.5, fontweight='bold', color=GREEN)
eqs = [(7.45, r'$d_i=\dfrac{x_{\sigma_i}-D(x_{\sigma_i},\sigma_i;c)}{\sigma_i}$'), (6.65, r'$\widetilde{x}=x_{\sigma_i}+(\sigma_{i+1}-\sigma_i)\,d_i$'),
       (5.8, r'$x_{\sigma_{i+1}}=x_{\sigma_i}+\dfrac{\sigma_{i+1}-\sigma_i}{2}\left[d_i+\dfrac{\widetilde{x}-D(\widetilde{x},\sigma_{i+1};c)}{\sigma_{i+1}}\right]$')]
for yy, tx in eqs: fig.text(fx(fx0 + .2), fy(yy), tx, ha='left', va='center', fontsize=12)
fig.text(fx(fx0 + .2), fy(5.25), '$D$: EDM-preconditioned U-FNO ($c_{\\rm skip},c_{\\rm out},c_{\\rm in}$)', ha='left', va='center', fontsize=9.5)
fig.text(.5, .992, 'Single-step sampling mechanism of the EDM / U-FNO conditional denoiser (deterministic Heun solver)', ha='center', va='top', fontsize=14, fontweight='bold')
fig.text(.5, .012, f'Real model outputs (adapted U-FNO, step {k+1}/{m["steps"]}, {m["time"][:16]}). Observed SST enters only through $c$; it is never pasted back into the state. Final step ($\\sigma_{{i+1}}=0$) uses the Euler step only (no 2nd evaluation). '
         f'Map extent 111–118°E, 15–24°N.', ha='center', va='bottom', fontsize=9)
for ext in ('png', 'pdf', 'svg'):
    if ext == 'svg': plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(f'{a.out_dir}/sampling_mechanism.{ext}', dpi=300)
print('saved', a.out_dir)
