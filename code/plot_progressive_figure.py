"""Progressive SST reconstruction figure (final typography). rows = test cases, columns = Input SST | denoiser estimates x0_hat at 4 noise levels | Final SST.
Reads the NPZ files of export_progressive_cases.py. Per-row SST colour range = 1-99th percentile of that case's observed SST (no ground truth for real gaps),
shared by the six panels of the row. Real lon/lat arrays (pcolormesh on cell edges), north up. Writes the figure (png 600 dpi / pdf / svg), figure_caption.txt, per_case_range_stats.json."""
import argparse, glob, json, os, numpy as np, matplotlib.pyplot as plt, cmocean
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from homeyer_rainbow import homeyer_rainbow, homeyer_rainbow_soft, muted_sst, vivid_sst

p = argparse.ArgumentParser(); p.add_argument('--config', default='progressive_config.json'); p.add_argument('--out', default='../figures/progressive_cases'); p.add_argument('--cmap', default=None); p.add_argument('--suffix', default='')
a = p.parse_args(); cfg = json.load(open(a.config)); fs = sorted(glob.glob(f'{cfg["out"]}/case*.npz'), key=lambda s: int(s.split('case')[-1].split('_')[0]))
Z = [np.load(f, allow_pickle=True) for f in fs]; n = len(Z)
from matplotlib import font_manager as fm
have = {f.name for f in fm.fontManager.ttflist}; FONT = next(f for f in ('Times New Roman', 'TeX Gyre Termes', 'Nimbus Roman', 'Nimbus Roman No9 L', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif') if f in have)   # Times New Roman if installed, else metric-compatible Times clone
plt.rcParams.update({'font.family': FONT, 'mathtext.fontset': 'custom', 'mathtext.rm': FONT, 'mathtext.it': FONT + ':italic', 'mathtext.bf': FONT + ':bold', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
edges = lambda c: np.concatenate([[c[0] - (c[1] - c[0]) / 2], (c[1:] + c[:-1]) / 2, [c[-1] + (c[-1] - c[-2]) / 2]])
lon, lat = Z[0]['lon'], Z[0]['lat']; assert all(np.array_equal(z['lon'], lon) and np.array_equal(z['lat'], lat) for z in Z)
LE, AE = edges(lon), edges(lat); X0, X1, Y0, Y1 = LE[0], LE[-1], min(AE[0], AE[-1]), max(AE[0], AE[-1])
LAND_COLOR = cfg.get('land_color', '#D8CCBC'); MISS_COLOR = '#8E959B'   # land = brown-gray, missing ocean = light neutral gray (distinct hue and lightness)
from matplotlib.colors import LinearSegmentedColormap
SST_ANCHORS = [(0.000, '#31356D'), (0.105, '#5482B1'), (0.221, '#93C3D8'), (0.337, '#D4EAEF'), (0.453, '#FDFDC4'), (0.558, '#F5D98C'), (0.674, '#E8995B'), (0.791, '#CF4338'), (0.895, '#9B242B'), (1.000, '#842225')]   # approximate picks from ref. Fig. 4 (user-supplied), not official values
ref_fig4_sst = lambda: LinearSegmentedColormap.from_list('SST_Reference_Figure4', SST_ANCHORS, N=256)
cm_sst = {'thermal': lambda: cmocean.cm.thermal.copy(), 'homeyer': homeyer_rainbow, 'homeyer_soft': homeyer_rainbow_soft, 'muted': muted_sst, 'vivid': vivid_sst, 'weather': ref_fig4_sst}[a.cmap or cfg.get('sst_cmap', 'weather')]()
def case_range(z):   # no complete ground truth for real cloud gaps -> robust percentiles of the valid OBSERVED SST of this case only
    o = z['input_sst_C'][np.isfinite(z['input_sst_C'])]; lo_, hi_ = np.percentile(o, cfg.get('range_percentiles', [1, 99])); return round(float(lo_), 1), round(float(hi_), 1)
RNG = [case_range(z) for z in Z]
from scipy import ndimage as ndi
def land_display(land, max_px=10):   # display only: ocean specks (<= max_px px) enclosed in the coast/land are painted as land so no stray dots/rings appear; data and mask files untouched
    lab, k = ndi.label(~land); sz = ndi.sum(~land, lab, range(1, k + 1)); return land | np.isin(lab, [i + 1 for i, v in enumerate(sz) if v <= max_px])
stages = [str(s) for s in Z[0]['stage_names']]; TT = {'initial': 'Initial estimate', 'early': 'Early estimate', 'middle': 'Middle estimate', 'late': 'Late estimate'}
cols = ['input'] + stages + ['final']; sig = {nm: float(Z[0]['stage_sigma'][i]) for i, nm in enumerate(stages)}
steps_, lv_ = int(Z[0]['steps']), [int(v) for v in Z[0]['stage_levels']]   # progress = completed levels / total levels of the actually saved states (not a sigma conversion)
PCT = [100 * l_ // steps_ for l_ in lv_] + [100]; assert PCT == [0, 25, 50, 75, 100], PCT
COLT = ['Input SST'] + [f'Denoising {p_}%' for p_ in PCT[:-1]] + ['Final SST']; COLS = [''] * 6
# ---- layout in inches
PW = PH = 1.7; GAP = .75; LM, RM = 1.5, 1.3; GAPV = .46; TOP = .55; BOT = .7; CBGAP, CBW = .16, .115   # square panels
W = LM + len(cols) * PW + (len(cols) - 1) * GAP + RM; H = TOP + n * PH + (n - 1) * GAPV + BOT
fig = plt.figure(figsize=(W, H)); X = lambda v: v / W; Yf = lambda v: v / H
ptop = lambda r: H - TOP - r * (PH + GAPV)        # top edge (inches from bottom) of row r
FS = dict(col=15, sub=11.5, case=15, time=12.5, cov=12, aux=12, tick=10.5, cbtick=11.5)
ims = []; axs = {}
for r, z in enumerate(Z):
    land, E = land_display(z['land']), z['E']; rlo, rhi = RNG[r]
    for c, nm in enumerate(cols):
        ax = fig.add_axes([X(LM + c * (PW + GAP)), Yf(ptop(r) - PH), X(PW), Yf(PH)]); axs[r, c] = ax; ax.set_facecolor(MISS_COLOR)
        fld = z['input_sst_C'] if nm == 'input' else np.where(E, z['final_sst_C'] if nm == 'final' else z[f'est_sst_C_{nm}'], np.nan)
        ax.pcolormesh(LE, AE, np.where(land, 1., np.nan), cmap=ListedColormap([LAND_COLOR]), vmin=0, vmax=1, rasterized=True)
        im = ax.pcolormesh(LE, AE, np.where(land, np.nan, fld), cmap=cm_sst, vmin=rlo, vmax=rhi, rasterized=True)
        if nm == 'input': ims.append(im)
        ax.contour(lon, lat, land.astype(float), levels=[.5], colors='black', linewidths=.9, zorder=5)   # coastline
        ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect('auto'); ax.set_box_aspect(1); ax.set_xticks([111, 113, 115, 117]); ax.set_yticks([15, 17, 19, 21, 23])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}°E')); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}°N'))
        ax.tick_params(direction='out', length=3, width=.7, labelsize=FS['tick'], pad=3.5, labelbottom=True, labelleft=True); plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), fontweight='bold')
        for sp in ax.spines.values(): sp.set_linewidth(.6)
    # left info column: Case and date/time rotated 90 deg (separate columns), missing fraction horizontal; all centred on the row's panels
    ts_ = __import__('pandas').to_datetime(str(z['time_utc'])); yc = ptop(r) - PH / 2; mf = 100 * (1 - float(z['obs_frac']))
    fig.text(X(.52), Yf(yc), ts_.strftime('%d %b %Y %H:%M UTC'), rotation=90, ha='center', va='center', fontsize=FS['time'], fontweight='bold')
    fig.text(X(.80), Yf(yc), f'Missing fraction: {mf:.0f}%', rotation=90, ha='center', va='center', fontsize=FS['cov'], fontweight='bold')
    x_l, x_r = LM, LM + len(cols) * PW + (len(cols) - 1) * GAP
    # per-row VERTICAL colorbar to the right of the sixth panel, same height as the panels
    cax = fig.add_axes([X(x_r + CBGAP), Yf(ptop(r) - PH), X(CBW), Yf(PH)]); cb = fig.colorbar(ims[r], cax=cax, orientation='vertical')
    cb.set_ticks(np.round(np.linspace(rlo, rhi, 5), 1)); cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}')); cb.ax.yaxis.set_ticks_position('right')
    cb.ax.tick_params(labelsize=FS['cbtick'], length=2.5, width=.5, pad=2); cb.outline.set_linewidth(.5)
    cb.set_label('SST (°C)', fontsize=FS['aux'], fontweight='bold', labelpad=5); cb.ax.yaxis.set_label_position('right')
# column headers, once, above the first row's case label
for c in range(len(cols)):
    xc = LM + c * (PW + GAP) + PW / 2
    fig.text(X(xc), Yf(ptop(0) + .14), COLT[c], fontsize=FS['col'], fontweight='bold', ha='center', va='bottom')
    if COLS[c]: fig.text(X(xc), Yf(ptop(0) + .40 - .21), COLS[c], fontsize=FS['sub'], ha='center', va='bottom')
os.makedirs(a.out, exist_ok=True)
for e, kw in (('png', dict(dpi=600)), ('pdf', {})): fig.savefig(f'{a.out}/progressive_sst_reconstruction_final{a.suffix}.{e}', **kw)
# ---- per-case statistics (unchanged definition) and the full technical caption
stats = {}
for r, z in enumerate(Z):
    lo_, hi_ = RNG[r]; E = z['E']; o = z['input_sst_C'][np.isfinite(z['input_sst_C'])]; d = dict(vmin=lo_, vmax=hi_, basis='percentiles of valid observed SST of this case', n_observed_px=int(o.size),
        observed_percentiles={f'p{q}': float(np.percentile(o, q)) for q in (1, 5, 25, 50, 75, 95, 99)}, panels={})
    for nm in ['input'] + stages + ['final']:
        f_ = o if nm == 'input' else (z['final_sst_C'] if nm == 'final' else z[f'est_sst_C_{nm}'])[E]
        d['panels'][nm] = dict(frac_below=float((f_ < lo_).mean()), frac_above=float((f_ > hi_).mean()), min=float(f_.min()), max=float(f_.max()), p1=float(np.percentile(f_, 1)), p50=float(np.percentile(f_, 50)), p99=float(np.percentile(f_, 99)))
    stats[f'case{r + 1}_{str(z["time_utc"])[:10]}'] = d
if a.suffix: raise SystemExit('saved comparison variant ' + a.suffix)
json.dump(stats, open(f'{cfg["out"]}/per_case_range_stats.json', 'w'), indent=1)
s_ = float(Z[0]['scale']); lv = [int(v) for v in Z[0]['stage_levels']]
cap = f"""Figure caption (full version for the manuscript)

Progressive SST reconstruction during EDM sampling for four real cloud-contaminated test cases (rows). Column 1: input SST (observed pixels only; medium gray = missing ocean (mask), beige = land, black line = coast). Columns 2-5: the denoiser's clean-field estimate x0_hat = D(x_sigma, sigma; c) at four noise levels of the 32-level EDM schedule (denoising progress 0/25/50/75% = sigma {', '.join(f'{sig[s]:.3g}' for s in stages)}; schedule levels {lv[1:] if lv[0] == 0 and len(lv) > len(stages) else lv}), taken from the first denoiser evaluation of each level. Column 6: the final SST after the full sampling process (Temporal-12 U-FNO denoiser, EDM Heun solver, observation re-injection with U = {int(Z[0]['U'])} resampling traversals per level).

Normalisation. The sampler works in normalised space x_norm = (x_SST - mu_V) / s, with s = {s_} (fixed, from the checkpoint) and mu_V the mean of the visible SST of each case. The denoiser output D (after EDM preconditioning, c_skip x + c_out F_theta) is converted back for display only: x_SST = s * x0_hat + mu_V, in kelvin, then minus 273.15 to get degC. mu_V is the case's own value used during sampling (not recomputed per stage). Sampling states and the final output are not modified by plotting.

Colour scales. Each row has its own SST colour range, identical for its six panels: the 1st-99th percentile of that case's observed SST (no complete ground truth exists for real cloud gaps), rounded to 0.1 degC. Values outside the range saturate at the end colours. Colours are comparable within a row, not across cases. Ranges (degC): """ + '; '.join(f"Row {i + 1} {RNG[i][0]:.1f}-{RNG[i][1]:.1f}" for i in range(n)) + """.
Share of ocean pixels outside the range (below + above), per panel input/initial/early/middle/late/final:
""" + '\n'.join(f"  Row {i + 1}: " + ' / '.join(f"{100 * (q['frac_below'] + q['frac_above']):.1f}%" for q in list(stats.values())[i]['panels'].values()) for i in range(n)) + f"""

Colormap: custom colormap SST_Reference_Figure4 (approximate colours picked from Fig. 4 of Chao et al. 2025, not the paper's official values), 256-level linear interpolation between the anchors #31356D, #5482B1, #93C3D8, #D4EAEF, #FDFDC4, #F5D98C, #E8995B, #CF4338, #9B242B, #842225 (cold to warm); one vertical colorbar per row.

Cases (UTC, missing fraction of ocean): """ + '; '.join(f"Row {i + 1}: {str(Z[i]['time_utc'])[:16]}, {1 - Z[i]['obs_frac']:.0%}" for i in range(n)) + f""".
Geographic extent: {lon.min():.0f}-{lon.max():.0f} E, {lat.min():.0f}-{lat.max():.0f} N, 0.02 degree grid. Figure font: {FONT}.
"""
open(f'{a.out}/figure_caption.txt', 'w').write(cap)
for k_, v_ in stats.items(): print(k_, 'range', v_['vmin'], v_['vmax'], {nm: round(100 * (q['frac_below'] + q['frac_above']), 1) for nm, q in v_['panels'].items()})
print('saved', a.out, 'font', FONT, 'size', round(W, 2), 'x', round(H, 2), 'in')
