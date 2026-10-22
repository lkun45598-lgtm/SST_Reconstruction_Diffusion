"""Redraw eval_jaxa candidates in the project SST map style (plotstyle.py). Uses saved candidates.npz; truth, hidden set H and
visible set V are re-derived with evalutil.select_items (same seed/n_frames as the eval run -> identical frames and clouds)."""
import argparse, os, numpy as np, torch, matplotlib.pyplot as plt
from data import Jaxa, CloudLib
from evalutil import laplace_fill, select_items
import plotstyle as ps

p = argparse.ArgumentParser()
p.add_argument('--eval-dir', required=True); p.add_argument('--n-frames', type=int, default=60)
p.add_argument('--n-plot', type=int, default=6); p.add_argument('--expect', nargs='*', default=[],
               help='sanity check: "idx=YYYY-MM-DD HH" that the re-derived frame must match')
a = p.parse_args()
dev = 'cuda'
lib = CloudLib(dev); jx = Jaxa(dev, lib); ocean_t = ~lib.land[None, None]
items = select_items(jx, lib, ocean_t, a.n_frames, seed=7, n_build=a.n_plot)
cand = np.load(f'{a.eval_dir}/candidates.npz')
land = lib.land.cpu().numpy(); ocean = ~land
out = f'{a.eval_dir}/figs'; os.makedirs(out, exist_ok=True)
for kv in a.expect:
    i, t = kv.split('='); got = str(jx.times[items[int(i)][0]])
    assert got.startswith(t), f'frame {i}: expected {t}, re-derived {got}'
C = lambda k: k - 273.15

for fi, (f, K, H_, V) in enumerate(items):
    x = jx.x[f][None, None]; truth = C(x[0, 0].cpu().numpy())
    Hm, Vm = H_[0, 0].cpu().numpy(), V[0, 0].cpu().numpy()
    lap = C(laplace_fill(x, V, ocean_t)[0, 0].cpu().numpy())
    g = lambda n, k: C(cand[f'{n}_f{fi}_{k}'])
    rep = lambda n: g(n, 'samples')[int(cand[f'{n}_f{fi}_rep'])]
    t = str(jx.times[f]).replace(':', '').replace(' ', '_')
    vmin, vmax = np.nanpercentile(truth[ocean], [1, 99])
    show = lambda fld, valid=None: np.where(valid, fld, np.nan) if valid is not None else fld
    panels = [
        (('Reference SST', '(ground truth)'),            show(truth, ~np.isnan(truth))),
        (('Masked input', '(observed pixels withheld)'), show(truth, Vm)),
        (('Withheld observations', '(scored pixels)'),   show(truth, Hm)),
        (('Laplace interpolation', '(baseline)'),         lap),
        (('OSTIA-pretrained only', '(representative sample)'), rep('pretrain')),
        (('OSTIA-pretrained only', '(ensemble mean, K=4)'),    g('pretrain', 'mean')),
        (('OSTIA + JAXA adapted', '(representative sample)'),  rep('jaxa')),
        (('OSTIA + JAXA adapted', '(ensemble mean, K=4)'),     g('jaxa', 'mean')),
    ]
    fig, axs = plt.subplots(4, 2, figsize=(9.6, 18.6), sharex=True, sharey=True, constrained_layout=True)
    for k, ((l1, l2), fld) in enumerate(panels):
        ax = axs.flat[k]; r, c = divmod(k, 2)
        ps.gray_base(ax, land); im = ps.draw(ax, fld, land, ps.sst_cmap(), vmin, vmax)
        ps.title(ax, l1, l2); ps.style_axes(ax, bottom=(r == 3), left=(c == 0))
    ps.shared_colorbar(fig, axs, im, 'SST (°C)')
    fig.savefig(f'{out}/frame{fi}_{t}_fields.png', dpi=200); plt.close(fig)

    # |error| on the scored pixels only (everything else gray / tan), vertical stack, one colorbar
    eh = lambda pred: np.where(Hm, np.abs(pred - truth), np.nan)
    ep = [(('|Error|, Laplace', '(withheld pixels)'), eh(lap)),
          (('|Error|, ensemble mean', '(OSTIA-pretrained only)'), eh(g('pretrain', 'mean'))),
          (('|Error|, ensemble mean', '(OSTIA + JAXA adapted)'), eh(g('jaxa', 'mean')))]
    def stack(panels, vmax, label, name, step):
        fig, axs = plt.subplots(len(panels), 1, figsize=(5.6, 5.0 * len(panels) + 0.6), sharex=True, constrained_layout=True)
        for k, ((l1, l2), fld) in enumerate(panels):
            ax = axs[k]; ps.gray_base(ax, land); im = ps.draw(ax, fld, land, ps.err_cmap(), 0, vmax)
            ps.title(ax, l1, l2); ps.style_axes(ax, bottom=(k == len(panels) - 1), left=True)
        ps.shared_colorbar(fig, axs, im, label, step=step, shrink=0.55)
        fig.savefig(f'{out}/frame{fi}_{t}_{name}.png', dpi=200); plt.close(fig)
    stack(ep, 1.5, 'Absolute error (°C)', 'errors', 0.5)
    # spread vs skill: same colour scale for error and ensemble std so they can be compared
    sp = [(('|Error|, ensemble mean', '(OSTIA + JAXA adapted)'), eh(g('jaxa', 'mean'))),
          (('Ensemble std', '(OSTIA + JAXA adapted, K=4)'), np.where(Hm, g('jaxa', 'std'), np.nan))]
    stack(sp, 1.0, 'Error or std (°C)', 'spread', 0.2)
    # combined summary: rows 1-2 = SST fields (shared SST colorbar), row 3 = |error| of the same three methods (own colorbar)
    fl = [(('Reference SST', '(ground truth)'), show(truth, ~np.isnan(truth))),
          (('Masked input', '(observed pixels withheld)'), show(truth, Vm)),
          (('Withheld observations', '(scored pixels)'), show(truth, Hm)),
          (('Laplace interpolation', '(baseline)'), lap),
          (('OSTIA-pretrained only', '(ensemble mean, K=4)'), g('pretrain', 'mean')),
          (('OSTIA + JAXA adapted', '(ensemble mean, K=4)'), g('jaxa', 'mean'))]
    er = [(('|Error|, Laplace', '(withheld pixels)'), eh(lap)),
          (('|Error|, OSTIA-pretrained only', '(withheld pixels)'), eh(g('pretrain', 'mean'))),
          (('|Error|, OSTIA + JAXA adapted', '(withheld pixels)'), eh(g('jaxa', 'mean')))]
    fig, axs = plt.subplots(3, 3, figsize=(13.2, 16.4), sharex=True, sharey=True, constrained_layout=True)
    for k, ((l1, l2), fld) in enumerate(fl):
        ax = axs.flat[k]; ps.gray_base(ax, land); im1 = ps.draw(ax, fld, land, ps.sst_cmap(), vmin, vmax)
        ps.title(ax, l1, l2); ps.style_axes(ax, bottom=False, left=(k % 3 == 0))
    for k, ((l1, l2), fld) in enumerate(er):
        ax = axs[2, k]; ps.gray_base(ax, land); im2 = ps.draw(ax, fld, land, ps.err_cmap(), 0, 1.5)
        ps.title(ax, l1, l2); ps.style_axes(ax, bottom=True, left=(k == 0))
    ps.shared_colorbar(fig, axs[:2, :], im1, 'SST (°C)', shrink=0.7)
    ps.shared_colorbar(fig, axs[2, :], im2, 'Absolute error (°C)', shrink=0.9)
    fig.savefig(f'{out}/frame{fi}_{t}_summary.png', dpi=200); plt.close(fig)
    m = Hm & ~np.isnan(truth)
    print(f'frame{fi} {t} hidden {m.sum()} px ({m.sum()/(~np.isnan(truth)).sum():.0%} of obs); '
          f'rmse hidden: laplace {np.sqrt(np.nanmean((lap-truth)[m]**2)):.3f} '
          f'pretrain-mean {np.sqrt(np.mean((g("pretrain","mean")-truth)[m]**2)):.3f} '
          f'adapted-mean {np.sqrt(np.mean((g("jaxa","mean")-truth)[m]**2)):.3f}', flush=True)
print('saved to', out)
