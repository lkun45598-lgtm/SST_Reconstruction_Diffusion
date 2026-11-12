"""Companion figure: what the G2 (RePaint-like) part does INSIDE one noise level, for the same four cases as the progressive figure.
Columns: observation mask V | state after Heun step | after re-injection of V | after jump-back noise | re-injected at sigma_i | after 2nd traversal.
All states are REAL recorded states of the sampler at one level (default: middle stage), shown with the display scaling x/sqrt(sigma^2+sigma_data^2)
(display only). Same style/grid as plot_progressive_figure.py."""
import argparse, glob, json, os, numpy as np, matplotlib.pyplot as plt, cmocean
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager as fm

p = argparse.ArgumentParser(); p.add_argument('--config', default='progressive_config.json'); p.add_argument('--out', default='../figures/progressive_cases'); p.add_argument('--stage', default='middle')
a = p.parse_args(); cfg = json.load(open(a.config)); fs_ = sorted(glob.glob(f'{cfg["out"]}/case*.npz'), key=lambda s: int(s.split('case')[-1].split('_')[0]))
Z = [np.load(f, allow_pickle=True) for f in fs_]; n = len(Z); SD = cfg['sigma_data']; R = cfg['state_range_scaled']
have = {f.name for f in fm.fontManager.ttflist}; FONT = next(f for f in ('Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans', 'DejaVu Sans') if f in have)
plt.rcParams.update({'font.family': FONT, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
edges = lambda c: np.concatenate([[c[0] - (c[1] - c[0]) / 2], (c[1:] + c[:-1]) / 2, [c[-1] + (c[-1] - c[-2]) / 2]])
lon, lat = Z[0]['lon'], Z[0]['lat']; LE, AE = edges(lon), edges(lat); X0, X1, Y0, Y1 = LE[0], LE[-1], min(AE[0], AE[-1]), max(AE[0], AE[-1])
names = [str(s) for s in Z[0]['stage_names']]; si = names.index(a.stage); k = int(Z[0]['stage_levels'][si]); sg = Z[0]['sigma_schedule']; s0, s1 = float(sg[k]), float(sg[k + 1])
cm = cmocean.cm.balance.copy()
PW = 1.5; PH = PW * (Y1 - Y0) / (X1 - X0); GAP = .3; LM, RM = .78, 1.15; GAPV = .66; TOP = 1.55; BOT = 1.15; ncol = 6
W = LM + ncol * PW + (ncol - 1) * GAP + RM; H = TOP + n * PH + (n - 1) * GAPV + BOT
fig = plt.figure(figsize=(W, H)); X = lambda v: v / W; Yf = lambda v: v / H; ptop = lambda r: H - TOP - r * (PH + GAPV)
FS = dict(title=17, col=12.5, sub=11, case=11.5, time=11, aux=10.5, tick=10, foot=9.5)
cols = [('Observation mask $V$', 'white: observed', None, None), ('After Heun step', f'σ = {s1:.3g}', 'heun', s1), ('After re-injection', f'σ = {s1:.3g}', 'reinj', s1),
        ('After jump-back noise', f'σ = {s0:.3g}', 'jump_noised', s0), ('Re-injection at σ', f'σ = {s0:.3g}', 'jump_reinj', s0), ('After 2nd traversal', f'σ = {s1:.3g}', 'trav_last', s1)]
cols[0] = ('Observation mask', 'white: observed', None, None)
for r, z in enumerate(Z):
    land, E, V = z['land'], z['E'], z['V']
    for c, (_, _, key, sg_) in enumerate(cols):
        ax = fig.add_axes([X(LM + c * (PW + GAP)), Yf(ptop(r) - PH), X(PW), Yf(PH)]); ax.set_facecolor('#B8B8B8')
        if key is None:
            for f_, col in ((np.where(~land, 1., np.nan), 'black'), (np.where(V & E, 1., np.nan), 'white'), (np.where(land, 1., np.nan), '#D6C6A2')):
                ax.pcolormesh(LE, AE, f_, cmap=ListedColormap([col]), vmin=0, vmax=1, rasterized=True)
        else:
            fld = np.where(E, z[f'rp_{a.stage}_{key}'] / np.sqrt(sg_ ** 2 + SD ** 2), np.nan)
            ax.pcolormesh(LE, AE, np.where(land, 1., np.nan), cmap=ListedColormap(['#D6C6A2']), vmin=0, vmax=1, rasterized=True)
            im = ax.pcolormesh(LE, AE, np.where(land, np.nan, fld), cmap=cm, vmin=-R, vmax=R, rasterized=True)
            if key in ('reinj', 'jump_reinj'): ax.contour(lon, lat, V.astype(float), levels=[.5], colors='k', linewidths=.35)   # where V was overwritten
        ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect('equal'); ax.set_xticks([111, 113, 115, 117]); ax.set_yticks([15, 17, 19, 21, 23])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}')); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}°N'))
        ax.tick_params(direction='out', length=3, width=.7, labelsize=FS['tick'], pad=1.5, labelbottom=(r == n - 1), labelleft=(c == 0))
        for sp in ax.spines.values(): sp.set_linewidth(.6)
        if r == n - 1: ax.set_xlabel('Longitude (°E)', fontsize=FS['aux'], labelpad=1.5)
    x_l = LM; y_c = ptop(r) + .10; rend = fig.canvas.get_renderer(); xx = x_l
    for txt, kw in ((f'Case {r + 1}', dict(fontsize=FS['case'], fontweight='bold')), (f'·  {str(z["time_utc"])[:16]} UTC', dict(fontsize=FS['time'])), (f'·  Observed: {z["obs_frac"]:.0%}', dict(fontsize=FS['time'], color='#444444'))):
        t_ = fig.text(X(xx), Yf(y_c), txt, ha='left', va='bottom', **kw); xx += t_.get_window_extent(rend).width / fig.dpi + .12
x_r = LM + ncol * PW + (ncol - 1) * GAP
for c, (t1, t2, _, _) in enumerate(cols):
    xc = LM + c * (PW + GAP) + PW / 2
    fig.text(X(xc), Yf(ptop(0) + .72), t1, fontsize=FS['col'], fontweight='semibold', ha='center', va='bottom'); fig.text(X(xc), Yf(ptop(0) + .53), t2, fontsize=FS['sub'], ha='center', va='bottom', color='#333333')
fig.add_artist(plt.Line2D([X(LM), X(x_r)], [Yf(ptop(0) + .42)] * 2, transform=fig.transFigure, color='#999999', lw=.6))
cax = fig.add_axes([X(x_r + .16), Yf(ptop(n - 1) - PH + (ptop(0) - ptop(n - 1) + PH) * .25), X(.115), Yf((ptop(0) - ptop(n - 1) + PH) * .5)]); cb = fig.colorbar(im, cax=cax)
cb.set_ticks([-4, -2, 0, 2, 4]); cb.ax.tick_params(labelsize=FS['tick'], length=2.5, width=.5); cb.outline.set_linewidth(.5)
fig.text(X(x_r + .16 + .115 + .7), Yf((ptop(0) + ptop(n - 1) - PH) / 2), 'Diffusion state, $x/\\sqrt{\\sigma^2+\\sigma_{data}^2}$ (display scaling)', rotation=90, fontsize=FS['aux'], ha='center', va='center')
fig.text(.5, Yf(H - .12), 'Observation Re-injection and Resampling Inside One Noise Level', ha='center', va='top', fontsize=FS['title'], fontweight='bold')
fig.text(.5, Yf(.12), f'Real sampler states at level {k + 1}/{int(Z[0]["steps"])} ({a.stage} stage): Heun step, re-injection of visible pixels V (outlined) with y + σε, jump-back noise on the ocean domain, re-injection at σ, and the 2nd traversal (U = {int(Z[0]["U"])}).\nAll panels share one diverging scale; the scaling is for display only. Gray: missing ocean; tan: land.',
         ha='center', va='bottom', fontsize=FS['foot'], linespacing=1.5)
os.makedirs(a.out, exist_ok=True)
for e, kw in (('png', dict(dpi=600)), ('pdf', {}), ('svg', {})): fig.savefig(f'{a.out}/repaint_mechanism_levelwise.{e}', **kw)
print('saved', a.out, 'level', k, 'sigma', s0, '->', s1)
