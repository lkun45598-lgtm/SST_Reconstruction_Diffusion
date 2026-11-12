"""Compose the paper figure of the real sampling trajectory from the arrays written by export_sampling_trajectory.py.
Everything drawn is a real intermediate of edm.sample (default path: EDM Heun, no re-injection of observations)."""
import argparse, json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import ListedColormap
import plotstyle as ps

p = argparse.ArgumentParser(); p.add_argument('--dir', default='../runs/sampling_trajectory'); p.add_argument('--out', default=None)
a = p.parse_args(); d = a.dir; out = a.out or f'{d}/sampling_process'
L = lambda n: np.load(f'{d}/{n}.npy'); m = json.load(open(f'{d}/meta.json'))
land, E, Vm = L('land'), L('ocean_mask') > .5, L('visible_mask') > .5
obs, init, den, upd, fin = L('input_sst'), L('init_state'), L('denoiser_pred'), L('updated_state'), L('final_sst')
vmin, vmax = np.percentile(obs[Vm], [1, 99]); k, s0, s1, sN = m['k'], m['sigma_k'], m['sigma_next'], m['sigma_max']
r0, r1 = 3 * sN, max(3 * s1, 1.5)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})

fig = plt.figure(figsize=(15.5, 10.6))
gs = fig.add_gridspec(2, 7, width_ratios=[1, .34, 1, .34, 1, .34, 1], left=.045, right=.99, top=.905, bottom=.15, wspace=.05, hspace=.62)
cols = [0, 2, 4, 6]; axs = {}
def panel(r, c, tag, l1, l2, fld, cmap, lo, hi, cb, bottom, left):
    ax = fig.add_subplot(gs[r, c]); axs[tag] = ax; ps.gray_base(ax, land); im = ps.draw(ax, np.where(E, fld, np.nan), land, cmap, lo, hi)
    ps.title(ax, l1, l2, size=11.5); ps.style_axes(ax, bottom, left); ax.tick_params(labelsize=8.5)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.text(.03, .985, tag, transform=ax.transAxes, ha='left', va='top', fontsize=13, fontweight='bold', bbox=dict(fc='white', ec='none', pad=1.5, alpha=.85))
    if cb:
        cax = ax.inset_axes([.1, -.2 if bottom else -.09, .8, .028]); c_ = fig.colorbar(im, cax=cax, orientation='horizontal'); c_.set_label(cb, fontsize=9.5, labelpad=2); c_.ax.tick_params(labelsize=8.5)
    return ax
def mask(r, c, tag, l1, l2, mk, bottom, left):
    ax = fig.add_subplot(gs[r, c]); axs[tag] = ax
    for fld, col in ((np.where(land, np.nan, 1.), 'black'), (np.where(mk, 1., np.nan), 'white'), (np.where(land, 1., np.nan), ps.LAND)):
        ax.imshow(fld, extent=ps.EXTENT, origin='upper', interpolation='nearest', cmap=ListedColormap([col]), vmin=0, vmax=1)
    ps.title(ax, l1, l2, size=11.5); ps.style_axes(ax, bottom, left); ax.tick_params(labelsize=8.5); ax.set_xlabel(''); ax.set_ylabel('')
    ax.text(.03, .985, tag, transform=ax.transAxes, ha='left', va='top', fontsize=13, fontweight='bold', bbox=dict(fc='white', ec='none', pad=1.5, alpha=.85))
sst, kw = ps.sst_cmap(), dict()
panel(0, 0, '(a)', 'Input SST', f'({m["observed_frac"]:.0%} of ocean observed)', obs, sst, vmin, vmax, 'SST (°C)', False, True)
mask(0, 2, '(b)', 'Observation mask $V$', '(white: observed)', Vm & E, False, False)
mask(0, 4, '(c)', 'Ocean domain $E$', '(white: generated)', E, False, False)
panel(1, 0, '(d)', 'Initial noise state', rf'$x_{{\sigma_0}}\sim\sigma_0\,\mathcal{{N}}(0,I)$ on $E$, $\sigma_0={sN:.0f}$', init, sst, -r0, r0, 'normalised state (dimensionless)', True, True)
panel(1, 2, '(e)', f'Denoiser prediction (step {k+1}/{m["steps"]})', rf'$D(x_{{\sigma_i}},\sigma_i)$, $\sigma_i={s0:.2f}$', den, sst, vmin, vmax, 'SST (°C)', True, False)
panel(1, 4, '(f)', f'Heun update (step {k+1}/{m["steps"]})', rf'$x_{{\sigma_{{i+1}}}}$, $\sigma_{{i+1}}={s1:.2f}$', upd, sst, -r1, r1, 'normalised state (dimensionless)', True, False)
panel(1, 6, '(g)', 'Final reconstruction', rf'$D\!\left(x_{{\sigma_{{N-1}}}}\right)$ at $\sigma_{{\min}}={m["sigma_min"]:.3f}$, $x_{{\sigma_N=0}}$', fin, sst, vmin, vmax, 'SST (°C)', True, False)

# condition box (row 0, col 6)
bx = fig.add_subplot(gs[0, 6]); bx.axis('off'); bx.set_xlim(0, 1); bx.set_ylim(0, 1)
bx.add_patch(FancyBboxPatch((0, .06), 1, .86, boxstyle='round,pad=0.0,rounding_size=0.04', fc='#EEF3FA', ec='#34588F', lw=1.2))
bx.text(.5, .84, 'Condition inputs\n(same at every step)', ha='center', va='center', fontsize=11, fontweight='bold', color='#34588F')
bx.text(.06, .68, r'$c=[\,y\!\cdot\!V,\;V,\;E,\;\mathrm{coords}\,]$  (8 ch)', fontsize=10.5, va='center')
bx.text(.06, .56, r'$y=(x_{\rm obs}-\mu_V)/s$ on $V$, else 0' + '\n' + r'$\mu_V$: mean of visible SST, $s=%.1f$' % m['scale'], fontsize=9.5, va='top', linespacing=1.5)
bx.text(.06, .32, 'coords: x/y (km), latitude, local' + '\n' + 'grid spacing (5 ch)', fontsize=9.5, va='top', linespacing=1.5)
bx.text(.06, .14, r'noise level $\sigma_i$ enters via $c_{\rm noise}=\ln\sigma_i/4$', fontsize=9.5, va='center')

def arrow(p0, p1, **k_):
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle='-|>', mutation_scale=18, lw=1.8, color=k_.pop('color', '#222'), **k_))
def ctr(tag, side):
    b = axs[tag].get_position(); return {'r': (b.x1, (b.y0 + b.y1) / 2), 'l': (b.x0, (b.y0 + b.y1) / 2), 'b': ((b.x0 + b.x1) / 2, b.y0), 't': ((b.x0 + b.x1) / 2, b.y1)}[side]
def label(x, y, txt, **k_): fig.text(x, y, txt, ha='center', va='center', fontsize=k_.pop('fontsize', 9.5), linespacing=1.4, **k_)
yr = ctr('(d)', 'r')[1]; gap = lambda t0, t1: ((ctr(t0, 'r')[0]), ctr(t1, 'l')[0])
for t0, t1, txt in (('(d)', '(e)', 'U-FNO\ndenoiser\n$D(x_{\\sigma_i},\\sigma_i;c)$'), ('(e)', '(f)', 'Heun step\n(2 network\nevaluations)'), ('(f)', '(g)', f'repeat\nsteps {k+2}…{m["steps"]}\n(σ → 0)')):
    x0, x1 = gap(t0, t1); arrow((x0 + .004, yr), (x1 - .004, yr)); label((x0 + x1) / 2, yr + .075, txt)
# condition inputs feed every network call: bracket from row 1 down to the denoiser arrow
xa = [ctr(t, 'b')[0] for t in ('(a)', '(b)', '(c)')]; yb = ctr('(a)', 'b')[1] - .07
xd0, xd1 = gap('(d)', '(e)'); xm = (xd0 + xd1) / 2
fig.add_artist(plt.Line2D([xa[0], xa[2]], [yb, yb], transform=fig.transFigure, color='#34588F', lw=1.6))
for x_ in xa: fig.add_artist(plt.Line2D([x_, x_], [yb, ctr('(a)', 'b')[1] - .005], transform=fig.transFigure, color='#34588F', lw=1.6))
fig.add_artist(plt.Line2D([xa[1], xa[1]], [yb, yb - .012], transform=fig.transFigure, color='#34588F', lw=1.6))
fig.add_artist(plt.Line2D([xa[1], xm], [yb - .012, yb - .012], transform=fig.transFigure, color='#34588F', lw=1.6))
arrow((xm, yb - .012), (xm, yr + .165), color='#34588F')
label((xa[0] + xa[1]) / 2, yb - .022, 'Condition inputs', color='#34588F', fontweight='bold', fontsize=9.5)
fig.text(.5, .985, 'Real sampling trajectory of the EDM / U-FNO conditional denoiser (deterministic Heun solver, no observation re-injection)', ha='center', va='top', fontsize=14, fontweight='bold')
fig.text(.5, .012, rf'Heun step: $d_i=(x_{{\sigma_i}}-D(x_{{\sigma_i}},\sigma_i))/\sigma_i$;  $\tilde x=x_{{\sigma_i}}+(\sigma_{{i+1}}-\sigma_i)d_i$;  '
         rf'$x_{{\sigma_{{i+1}}}}=x_{{\sigma_i}}+(\sigma_{{i+1}}-\sigma_i)\,\frac{{1}}{{2}}\,[\,d_i+(\tilde x-D(\tilde x,\sigma_{{i+1}}))/\sigma_{{i+1}}\,]$.' '\n'
         f'{m["steps"]} steps, σ: {sN:.0f} → {m["sigma_min"]:.3f} → 0 (ρ=7).  {m["time"][:16]}, 0.02° grid.  Observed values are NOT pasted back.', ha='center', va='bottom', fontsize=9)
for ext in ('png', 'pdf'): fig.savefig(f'{out}.{ext}', dpi=300)
print('saved', out + '.png/.pdf')
