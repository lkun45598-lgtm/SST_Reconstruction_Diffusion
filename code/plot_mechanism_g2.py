"""Component figures + full figure of the CURRENT mechanism (Temporal-12 U-FNO denoiser, EDM Heun, G2 = reinject V + resample jumps),
built only from the real trajectory exported by export_g2_mechanism.py. Writes figures/mechanism_g2/{1_conditions,...,7_full}.{png,pdf,svg}."""
import argparse, json, os, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.colors import ListedColormap
import plotstyle as ps

p = argparse.ArgumentParser(); p.add_argument('--dir', default='../runs/mechanism_g2'); p.add_argument('--out-dir', default='../figures/mechanism_g2')
a = p.parse_args(); os.makedirs(a.out_dir, exist_ok=True); d = a.dir
T = np.load(f'{d}/trajectory.npz'); Hs = np.load(f'{d}/history.npz'); m = json.load(open(f'{d}/meta.json')); land = np.load(f'{d}/land.npy'); tb = np.load(f'{d}/time_bias.npy')
E, V, y = T['E'], T['V'], T['y']; obs = np.load(f'{d}/target_obs.npy'); K, U = m['k'], m['U']; sg = T['sigma']; s0, s1 = float(sg[K]), float(sg[K + 1])
ev = lambda nm, u=0: T[f'ev_{K}_{u}_{nm}']; kel = lambda f: f * float(T['scale']) + float(T['mu']) - 273.15
vmin, vmax = np.percentile(obs[V], [1, 99]); R = 3 * s0
BLUE, GREEN, ORNG, PURP, RED, GRY = '#34588F', '#4C9A5B', '#C5672B', '#7A4FA3', '#B03A2E', '#777777'
LB = '#BFD8EE'; plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
n_ = lambda mk: np.where(mk, 1., np.nan)

class F:
    def __init__(s, W, H, title=None, foot=None):
        s.W, s.H = W, H; s.fig = plt.figure(figsize=(W, H)); s.ax = {}
        if title: s.fig.text(.5, .995, title, ha='center', va='top', fontsize=14, fontweight='bold')
        if foot: s.fig.text(.5, .01, foot, ha='center', va='bottom', fontsize=9)
    def A(s, x, y, w, h): return s.fig.add_axes([x / s.W, y / s.H, w / s.W, h / s.H])
    def txt(s, x, y, t, fs=10.5, c='k', ha='center', va='center', bold=False, **k): s.fig.text(x / s.W, y / s.H, t, ha=ha, va=va, fontsize=fs, color=c, fontweight='bold' if bold else 'normal', linespacing=1.45, **k)
    def box(s, x, y, w, h, t, ec, fc, fs=11.5, bold=True, dash=False):
        s.fig.add_artist(FancyBboxPatch((x / s.W, y / s.H), w / s.W, h / s.H, boxstyle='round,pad=0,rounding_size=0.01', transform=s.fig.transFigure, fc=fc, ec=ec, lw=1.5, ls='--' if dash else '-'))
        if t: s.txt(x + w / 2, y + h / 2, t, fs=fs, c=ec, bold=bold)
    def arr(s, p0, p1, c='#222', ls='-', lw=1.8, rad=0):
        s.fig.add_artist(FancyArrowPatch((p0[0] / s.W, p0[1] / s.H), (p1[0] / s.W, p1[1] / s.H), transform=s.fig.transFigure, arrowstyle='-|>', mutation_scale=17, lw=lw, color=c, ls=ls, connectionstyle=f'arc3,rad={rad}'))
    def ln(s, xs, ys, c='#222', lw=1.8, ls='-'): s.fig.add_artist(plt.Line2D([v / s.W for v in xs], [v / s.H for v in ys], transform=s.fig.transFigure, color=c, lw=lw, ls=ls))
    def mp(s, x, y, w, h, tag, l1, l2, fld, kind, cb=None, left=False, outline=None, w_=None):
        ax = s.A(x, y, w, h)
        if kind in ('sst', 'state', 'sstraw'):
            lo, hi, cm = (vmin, vmax, ps.sst_cmap()) if kind != 'state' else (-R, R, ps.sst_cmap())
            ps.gray_base(ax, land); im = ps.draw(ax, np.where(E, fld, np.nan) if kind != 'sstraw' else fld, land, cm, lo, hi)
        else:
            col = {'V': [(n_(~land), 'black'), (n_(fld), 'white')], 'E': [(n_(fld), LB)]}[kind]
            for f_, c_ in col + [(n_(land), ps.LAND)]: ax.imshow(f_, extent=ps.EXTENT, origin='upper', interpolation='nearest', cmap=ListedColormap([c_]), vmin=0, vmax=1)
            if kind == 'E': ax.imshow(n_(land), extent=ps.EXTENT, origin='upper', interpolation='nearest', cmap=ListedColormap([ps.LAND]), vmin=0, vmax=1)
        if outline is not None: ax.contour(np.flipud(outline.astype(float)), levels=[.5], colors='k', linewidths=.3, extent=ps.EXTENT)
        ps.style_axes(ax, False, left); ax.set_xlabel(''); ax.set_ylabel(''); ax.set_xticks([]); ax.set_yticks([15, 19, 23] if left else []); ax.tick_params(labelsize=9)
        ax.set_title(l1 + ('\n' + l2 if l2 else ''), fontsize=11, fontweight='bold', linespacing=1.35)
        if tag: ax.text(.03, .985, tag, transform=ax.transAxes, ha='left', va='top', fontsize=12.5, fontweight='bold', bbox=dict(fc='white', ec='none', pad=1.5, alpha=.85))
        if kind == 'V': ax.legend(handles=[Patch(fc='white', ec='k', lw=.6, label='observed'), Patch(fc='black', label='missing')], loc='upper center', bbox_to_anchor=(.5, -.03), ncol=2, fontsize=8.5, frameon=False, handlelength=1.1, columnspacing=.8, handletextpad=.4)
        if kind == 'E': ax.legend(handles=[Patch(fc=LB, ec='k', lw=.6, label='ocean'), Patch(fc=ps.LAND, ec='k', lw=.6, label='land')], loc='upper center', bbox_to_anchor=(.5, -.03), ncol=2, fontsize=8.5, frameon=False, handlelength=1.1, columnspacing=.8, handletextpad=.4)
        if cb:
            c = s.fig.colorbar(im, cax=s.A(x + .12 * w, y - .4, .76 * w, .1), orientation='horizontal'); c.set_label(cb, fontsize=9, labelpad=2); c.ax.tick_params(labelsize=8.5)
        return ax
    def save(s, name):
        for e in ('png', 'pdf', 'svg'):
            if e == 'svg': plt.rcParams['svg.fonttype'] = 'none'
            s.fig.savefig(f'{a.out_dir}/{name}.{e}', dpi=300)
        plt.close(s.fig); print('saved', name)
CB_S, CB_N = 'SST (°C)', 'normalised state'
SIG = lambda s_: f'$\\sigma_i={s0:.2f}$' if s_ == 0 else f'$\\sigma_{{i+1}}={s1:.2f}$'
FOOT = f'Real model outputs: Temporal-12 U-FNO, target hour {m["time"][:16]}, real cloud gaps ({m["observed_frac"]:.0%} of ocean observed), noise level index i={K} of {m["steps"]}, G2 with U={U} traversals. Map extent 111–118°E, 15–24°N.'
MW, MH = 1.75, 2.25

# ---------------------------------------------------------------- 1 conditions
f = F(15, 6.2, 'Component 1 — Condition inputs $c$ (fixed during sampling)', FOOT)
yy = 2.0
f.mp(.9, yy, MW, MH, '(a)', 'Input SST $x_{\\rm obs}$', '(real JAXA, target hour)', obs, 'sstraw', CB_S, left=True)
f.mp(3.4, yy, MW, MH, '(b)', 'Visible values $y\\cdot V$', '(normalised, 0 elsewhere)', np.where(V, y, np.nan), 'state', 'normalised ($x_{\\rm obs}-\\mu_V)/s$')
f.mp(5.9, yy, MW, MH, '(c)', 'Observation mask $V$', '(visible pixels)', V & E, 'V')
f.mp(8.4, yy, MW, MH, '(d)', 'Ocean domain $E$', '(generation domain)', E, 'E')
f.box(10.85, yy + .55, 1.85, 1.15, 'coords (5 ch)', BLUE, '#F3F6FB', fs=11); f.txt(11.775, yy + .1, '$x_{\\rm km}/100,\\ y_{\\rm km}/100,$\n$(\\mathrm{lat}-20)/10,$\n$\\ln dx_{\\rm loc}/3,\\ \\ln dy/3$', fs=9)
f.box(13.2, yy + .35, 1.6, 1.55, 'condition\n$c$ (8 ch)', BLUE, '#EEF3FA', fs=12)
for xr in (2.65, 5.15, 7.65, 10.15): pass
f.ln([2.8, 3.1], [yy + MH / 2] * 2, BLUE)
for x0 in (5.3, 7.8): f.ln([x0, x0 + .1], [yy + MH / 2] * 2, BLUE)
f.arr((12.75, yy + 1.12), (13.18, yy + 1.12), BLUE); f.ln([10.2, 10.6], [yy + MH / 2] * 2, BLUE)
f.txt(7.5, 5.55, 'c = [ y·V , V , E , coords ]  — concatenated channel-wise;   $y=(x_{\\rm obs}-\\mu_V)/s$ on $V$ ($\\mu_V$: mean of visible SST, $s=1.6$)', fs=11.5, c=BLUE)
f.save('1_conditions')

# ---------------------------------------------------------------- 2 temporal
f = F(16, 7.6, 'Component 2 — Temporal-12 branch: 11 past hours as a fixed condition (causal, never noised)', FOOT)
hi, lab = [0, 5, 9, 10], ['$t-11$ h', '$t-6$ h', '$t-2$ h', '$t-1$ h']; yh = 4.55
for j, (i_, lb) in enumerate(zip(hi, lab)):
    f.mp(.8 + j * 2.1, yh, 1.6, 2.05, f'({"abcd"[j]})', f'history {lb}', f'({m["hist_obs_frac"][i_]:.0%} observed)', Hs['hist'][i_], 'sstraw', left=(j == 0))
f.txt(4.95, 8.0 - .02, '', fs=1); f.txt(4.95, yh - .55, '… 11 frames in total ($t-11$ h … $t-1$ h), real clouds differ in every hour', fs=10, c=GRY)
f.box(9.2, yh + .25, 2.4, 1.5, 'Shared frame encoder\n$[{\\rm SST\\cdot valid},\\ {\\rm valid}]$ (2 ch) → 32 ch\nstride-2 conv + 2 res. blocks', BLUE, '#EEF3FA', fs=10)
f.arr((7.2, yh + 1.0), (9.18, yh + 1.0))
f.txt(13.35, yh + 1.75, 'learned time bias $b(\\mathrm{head},\\ \\mathrm{lag})$', fs=10.5, bold=True)
ax = f.A(12.2, yh + .1, 3.4, 1.35)
for hh in range(tb.shape[0]): ax.plot(np.arange(-11, 0), tb[hh], marker='o', ms=3, lw=1.2, label=f'head {hh + 1}')
ax.axhline(0, c='#aaa', lw=.6); ax.set_xticks([-11, -6, -1]); ax.set_xticklabels(['$t-11$h', '$t-6$h', '$t-1$h'], fontsize=8.5); ax.tick_params(labelsize=8.5); ax.legend(fontsize=7.5, ncol=2, frameon=False)
yb = 1.15
f.mp(.8, yb, 1.6, 2.05, '(e)', 'Current hour', '(query source)', obs, 'sstraw', left=True)
f.box(3.1, yb + .45, 2.5, 1.2, 'Stem conv 3×3\n$[x_\\sigma,\\,y\\!\\cdot\\!V,\\,V,\\,E,\\,\\mathrm{coords}]$ → 48 ch', ORNG, '#FBEBDD', fs=10)
f.arr((2.45, yb + 1.05), (3.08, yb + 1.05))
f.box(6.5, yb + .1, 3.7, 2.0, 'Per-pixel temporal attention\n(half resolution, 4 heads)\nquery: current stem features\nkey / value: 11 history frames\n+ learned time bias', GREEN, '#EAF5EC', fs=10.2)
f.arr((5.62, yb + 1.05), (6.48, yb + 1.05), ORNG); f.txt(6.05, yb + 1.35, 'query', fs=9, c=ORNG)
f.arr((10.4, yb + 2.7 + .1 + 1.4 - 1.4 + 0.65), (9.3, yb + 2.1), BLUE, rad=0) if False else None
f.ln([10.4, 10.4], [yh + .25, yb + 1.9], BLUE); f.arr((10.4, yb + 1.9), (10.2, yb + 1.9) if False else (9.0, yb + 1.9) if False else (10.22, yb + 1.9), BLUE) if False else None
f.ln([10.4, 10.4], [yh + .25, yb + 1.4], BLUE); f.arr((10.4, yb + 1.4), (10.2, yb + 1.4), BLUE)
f.txt(10.95, yb + 2.2, 'key / value', fs=9, c=BLUE)
f.box(11.4, yb + .35, 4.0, 1.5, 'zero-initialised 1×1 conv, then\n$h \\leftarrow h + \\Phi(h,\\ \\mathrm{history})$\n(single-frame net unchanged at init)', PURP, '#F4EEF9', fs=10)
f.arr((10.22, yb + 1.1), (11.38, yb + 1.1), GREEN)
f.txt(8.0, .62, 'History: normalised with the TARGET’s $\\mu_V, s$; invalid pixels are exactly 0 with mask 0; frames with no valid pixel are not attended.  Same history at every network call.', fs=9.5, c='#444')
f.save('2_temporal')

# ---------------------------------------------------------------- 3 denoiser
f = F(16, 9.0, 'Component 3 — EDM-preconditioned U-FNO denoiser $D(x_\\sigma,\\sigma;c,\\mathrm{history})$', FOOT)
f.mp(.7, 6.1, 1.6, 2.05, '(a)', 'Input state', f'$x_{{\\sigma_i}}$, {SIG(0)}', ev('x_in', 0), 'state', CB_N, left=True)
f.box(3.1, 6.65, 1.9, .95, '$c_{\\rm in}\\,x_\\sigma\\cdot E$\n(input scaling)', GRY, '#F2F2F2', fs=10.5); f.arr((2.35, 7.12), (3.08, 7.12))
f.box(5.6, 6.35, 2.6, 1.55, 'concat with $c$ (8 ch) → 9 ch\nstem conv 3×3 → 48 ch\n$+$ temporal branch\n(Component 2)', ORNG, '#FBEBDD', fs=9.8); f.arr((5.02, 7.12), (5.58, 7.12))
f.box(8.9, 6.65, 3.2, .95, 'FiLM embedding of\n$c_{\\rm noise}=\\ln\\sigma/4$ and grid spacing', PURP, '#F4EEF9', fs=9.5); f.ln([12.1, 12.5], [7.12, 7.12], PURP, lw=1.2, ls=':'); f.txt(13.9, 7.12, 'modulates every\nFiLM-Res / FNO block', fs=9, c=PURP)
f.box(8.9, 5.5, 6.3, .8, 'FNO block: spectral conv (12 modes, physical wavenumbers) ‖ 1×1 conv, + MLP', GRY, '#F7F7F7', fs=8.8, bold=False)
rb = [(1.0, 'level 0 (48 ch)\nFiLM-Res + FNO'), (3.95, '↓conv s2 → level 1 (96)\nFiLM-Res + FNO'), (6.9, 'level 2 mid (96 ch)\nFiLM-Res + FNO'), (9.85, 'decoder 1 (96→48)\nup + skip + FNO'), (12.8, 'decoder 0 (48 ch)\nup + skip + FNO')]
yB = 3.4
for k_, (x_, t_) in enumerate(rb): f.box(x_, yB, 2.3, .85, t_, RED if k_ == 2 else ORNG, '#FBEAE8' if k_ == 2 else '#FBEBDD', fs=9)
for k_ in range(4): f.arr((rb[k_][0] + 2.32, yB + .42), (rb[k_ + 1][0] - .02, yB + .42), GRY)
pass; f.txt(5.05, yB + 1.0 - .02, '', fs=1)
f.ln([6.9 - .5, 6.9 - .5], [yB + .42, yB + .42], GRY)
f.ln([2.15, 2.15, 13.95], [yB + .87, 4.95, 4.95], GRY, lw=1.3, ls='--'); f.arr((13.95, 4.95), (13.95, yB + .87), GRY, ls='--', lw=1.3)
f.ln([5.1, 5.1, 11.0], [yB + .87, 4.6, 4.6], GRY, lw=1.3, ls='--'); f.arr((11.0, 4.6), (11.0, yB + .87), GRY, ls='--', lw=1.3); f.txt(8.0, 4.52, 'skip connections', fs=8.5, c=GRY, va='top')
f.ln([6.9, 6.9, .45, .45], [6.35, 5.25, 5.25, yB + .42], ORNG); f.arr((.45, yB + .42), (.98, yB + .42), ORNG)
f.box(12.8, 1.7, 2.3, .9, 'head: GN–SiLU–\nconv 3×3 → $F_\\theta$', ORNG, '#FBEBDD', fs=9.5); f.arr((13.95, yB - .02), (13.95, 2.62), GRY)
f.box(5.7, 1.3, 5.9, 1.7, '', BLUE, '#EEF3FA'); f.txt(8.65, 2.55, '$D=(c_{\\rm skip}\\,x_\\sigma+c_{\\rm out}\\,F_\\theta)\\cdot E$', fs=12, c=BLUE)
f.txt(8.65, 1.75, '$c_{\\rm skip}=\\dfrac{\\sigma_d^2}{\\sigma^2+\\sigma_d^2},\\ c_{\\rm out}=\\dfrac{\\sigma\\sigma_d}{\\sqrt{\\sigma^2+\\sigma_d^2}},\\ c_{\\rm in}=\\dfrac{1}{\\sqrt{\\sigma^2+\\sigma_d^2}}$', fs=10.5, c=BLUE)
f.arr((12.78, 2.15), (11.62, 2.15), GRY)
f.mp(2.9, 1.15, 1.6, 1.75, '(b)', 'Denoised estimate', f'$D(x_{{\\sigma_i}},\\sigma_i;c)$', kel(ev('den', 0)), 'sst', 'SST (°C), de-normalised'); f.arr((5.68, 2.15), (4.55, 2.15))
f.save('3_denoiser')

# ---------------------------------------------------------------- 4 heun
f = F(16, 6.6, 'Component 4 — Heun predictor–corrector step (two network evaluations)', FOOT)
yy = 2.35; xs = [.8, 3.65, 6.5, 9.35, 12.2]
f.mp(xs[0], yy, MW, MH, '(a)', 'Current state', f'$x_{{\\sigma_i}}$, {SIG(0)}', ev('x_in', 0), 'state', CB_N, left=True)
f.mp(xs[1], yy, MW, MH, '(b)', '1st evaluation', f'$D(x_{{\\sigma_i}},\\sigma_i;c)$', kel(ev('den', 0)), 'sst', CB_S)
f.mp(xs[2], yy, MW, MH, '(c)', 'Predictor (Euler)', f'$\\widetilde{{x}}$, at {SIG(1)}', ev('x_euler', 0), 'state', CB_N)
f.mp(xs[3], yy, MW, MH, '(d)', '2nd evaluation', f'$D(\\widetilde{{x}},\\sigma_{{i+1}};c)$', kel(ev('den2', 0)), 'sst', CB_S)
f.mp(xs[4], yy, MW, MH, '(e)', 'Corrector (Heun)', f'$x_{{\\sigma_{{i+1}}}}$ (before re-injection)', ev('x_heun', 0), 'state', CB_N)
yc = yy + MH / 2
for j, lb in enumerate(['$d_i=\\dfrac{x_{\\sigma_i}-D}{\\sigma_i}$', 'Euler step\n$\\widetilde{x}=x_{\\sigma_i}+(\\sigma_{i+1}-\\sigma_i)d_i$', 'U-FNO\n(2nd call)', 'average the two slopes']):
    f.arr((xs[j] + MW + .1, yc), (xs[j + 1] - .1, yc))
f.txt(xs[0] + MW + .55, yc + .45, 'U-FNO\n(1st call)', fs=9.5, c=ORNG); f.txt(xs[1] + MW + .55, yc + .55, 'Euler\nstep', fs=9.5, c=GREEN); f.txt(xs[2] + MW + .55, yc + .45, 'U-FNO\n(2nd call)', fs=9.5, c=ORNG); f.txt(xs[3] + MW + .55, yc + .55, 'slope\naverage', fs=9.5, c=GREEN)
f.txt(8.0, 1.3, '$d_i=\\dfrac{x_{\\sigma_i}-D(x_{\\sigma_i},\\sigma_i;c)}{\\sigma_i},\\quad \\widetilde{x}=x_{\\sigma_i}+(\\sigma_{i+1}-\\sigma_i)\\,d_i,\\quad x_{\\sigma_{i+1}}=x_{\\sigma_i}+\\dfrac{\\sigma_{i+1}-\\sigma_i}{2}[d_i+\\dfrac{\\widetilde{x}-D(\\widetilde{x},\\sigma_{i+1};c)}{\\sigma_{i+1}}]$', fs=12.5)
f.txt(8.0, .55, 'At the last level ($\\sigma_{i+1}=0$) only the Euler step is used (no 2nd evaluation). Deterministic: no noise is added inside the Heun step.', fs=9.5, c='#444')
f.save('4_heun')

# ---------------------------------------------------------------- 5 reinject
f = F(15, 6.6, 'Component 5 — Re-injection of the visible observations (G2 / “reinject”)', FOOT)
yy = 2.3
f.mp(.9, yy, MW, MH, '(a)', 'After Heun step', f'$x_{{\\sigma_{{i+1}}}}$ ({"model on all of $E$"})', ev('x_heun', 0), 'state', CB_N, left=True)
f.mp(3.9, yy, MW, MH, '(b)', 'Visible mask $V$', '(where the state is replaced)', V & E, 'V')
f.mp(6.9, yy, MW, MH, '(c)', 'Injected value on $V$', f'$y+\\sigma_{{i+1}}\\,\\varepsilon$, fresh $\\varepsilon\\!\\sim\\!\\mathcal{{N}}(0,I)$', np.where(V, ev('x_next', 0), np.nan), 'state', CB_N)
f.mp(10.9, yy, MW, MH, '(d)', 'After re-injection', f'$x_{{\\sigma_{{i+1}}}}$ (outline = $V$)', ev('x_next', 0), 'state', CB_N, outline=V)
yc = yy + MH / 2
f.arr((2.75, yc), (3.85, yc)); f.txt(3.3, yc + .3, '$\\odot(1-V)$', fs=9.5); f.arr((5.7, yc), (6.85, yc)); f.arr((8.7, yc), (10.85, yc)); f.txt(9.78, yc + .3, '$+\\ V\\odot$', fs=10)
f.txt(7.5, 1.35, '$x_{\\sigma_{i+1}}\\ \\leftarrow\\ (1-V)\\odot x_{\\sigma_{i+1}}\\ +\\ V\\odot(y+\\sigma_{i+1}\\,\\varepsilon)$', fs=13.5)
f.txt(7.5, .6, 'Hidden pixels and true cloud gaps keep the model’s state; only $V$ is overwritten, with the noised observation at the NEXT noise level (training-time state).\nAt $\\sigma_{i+1}=0$ this makes the output equal the observation on $V$.', fs=9.5, c='#444')
f.save('5_reinject')

# ---------------------------------------------------------------- 6 jump
f = F(15, 6.9, f'Component 6 — Resampling jump (G2, U={U} traversals per noise level)', FOOT)
yy = 2.45; xs = [.9, 3.9, 6.9, 9.9]
f.mp(xs[0], yy, MW, MH, '(a)', f'After traversal 1', f'$x_{{\\sigma_{{i+1}}}}$, {SIG(1)}', ev('x_next', 0), 'state', CB_N, left=True)
f.mp(xs[1], yy, MW, MH, '(b)', 'Jump-back noise on $E$', f'$\\sqrt{{\\sigma_i^2-\\sigma_{{i+1}}^2}}\\,\\varepsilon$', ev('x_jump_noised', 0) - ev('x_next', 0), 'state', CB_N)
f.mp(xs[2], yy, MW, MH, '(c)', 'Back at $\\sigma_i$, $V$ re-injected', f'$x_{{\\sigma_i}}$, {SIG(0)} (outline = $V$)', ev('x_jump', 0), 'state', CB_N, outline=V)
f.mp(xs[3], yy, MW, MH, '(d)', f'After traversal 2', f'$x_{{\\sigma_{{i+1}}}}$, {SIG(1)}', ev('x_next', U - 1), 'state', CB_N)
yc = yy + MH / 2
f.arr((xs[0] + MW + .1, yc), (xs[1] - .1, yc)); f.txt(xs[0] + MW + .62, yc + .55, '$+$ noise\n(only on $E$)', fs=9.5, c=RED)
f.arr((xs[1] + MW + .1, yc), (xs[2] - .1, yc)); f.txt(xs[1] + MW + .62, yc + .55, 're-inject $V$\nat $\\sigma_i$', fs=9.5, c=BLUE)
f.arr((xs[2] + MW + .1, yc), (xs[3] - .1, yc)); f.txt(xs[2] + MW + .62, yc + .6, 'repeat\nHeun step\n(Comp. 3–5)', fs=9.5, c=GREEN)
f.txt(7.5, 1.5, '$x\\leftarrow x+\\sqrt{\\sigma_i^2-\\sigma_{i+1}^2}\;\\varepsilon\\ \\ (\\text{on }E),\\qquad x\\leftarrow(1-V)\\odot x+V\\odot(y+\\sigma_i\\varepsilon)$', fs=13)
f.txt(7.5, .65, f'Done for traversals u = 1 … U−1 at every noise level except σ=0; the last traversal moves on to the next level. U={U} was the best setting in the test-set comparison (U≥5 hurt).\nThe jump only re-noises (so it can be viewed as stochastic resampling); the memory notes leave open whether the gain comes from this or from RePaint-style harmonisation.', fs=9, c='#444')
f.save('6_jump')

# ---------------------------------------------------------------- 7 full
f = F(16, 9.4, 'Current sampling mechanism: Temporal-12 U-FNO + EDM Heun + G2 (visible re-injection and resampling jumps)', FOOT)
yt = 6.1; w_ = 1.6; h_ = 2.05
f.mp(.7, yt, w_, h_, '(a)', 'Input SST', f'({m["observed_frac"]:.0%} observed)', obs, 'sstraw', left=True)
f.mp(2.75, yt, w_, h_, '(b)', 'Mask $V$', '', V & E, 'V'); f.mp(4.8, yt, w_, h_, '(c)', 'Domain $E$', '', E, 'E')
f.mp(6.85, yt, w_, h_, '(d)', 'History', '($t-1$ h; 11 past hours)', Hs['hist'][10], 'sstraw')
f.fig.add_artist(FancyBboxPatch((.35 / 16, 5.45 / 9.4), 8.35 / 16, 3.55 / 9.4, boxstyle='round,pad=0,rounding_size=0.01', transform=f.fig.transFigure, fc='none', ec=BLUE, lw=1.2, ls='--'))
f.txt(8.55, 5.62, 'Condition inputs (fixed)', fs=10, c=BLUE, ha='right', bold=True)
f.box(9.3, yt + .3, 2.0, 1.5, 'condition $c$\n$[y\\!\\cdot\\!V,V,E,$ coords$]$\n+ 11-hour history', BLUE, '#EEF3FA', fs=10); f.arr((8.75, yt + 1.05), (9.28, yt + 1.05), BLUE)
f.box(11.8, 5.55, 4.0, 3.4, '', GREEN, 'white', fs=9)
f.txt(13.8, 8.7, 'Per noise level $\\sigma_i\\to\\sigma_{i+1}$', fs=11.5, c=GREEN, bold=True)
f.txt(11.95, 7.75, '1. $d_i=(x_{\\sigma_i}-D(x_{\\sigma_i},\\sigma_i;c))/\\sigma_i$,  $\\widetilde{x}=x_{\\sigma_i}+(\\sigma_{i+1}-\\sigma_i)d_i$', fs=9.3, ha='left')
f.txt(11.95, 7.15, '2. $x\\leftarrow x_{\\sigma_i}+\\frac{\\sigma_{i+1}-\\sigma_i}{2}[d_i+\\frac{\\widetilde{x}-D(\\widetilde{x},\\sigma_{i+1};c)}{\\sigma_{i+1}}]$', fs=9.3, ha='left')
f.txt(11.95, 6.55, '3. $x\\leftarrow(1-V)\\odot x+V\\odot(y+\\sigma_{i+1}\\varepsilon)$', fs=9.3, ha='left')
f.txt(11.95, 5.95, f'4. if traversal $<U$={U}: $x\\leftarrow x+\\sqrt{{\\sigma_i^2-\\sigma_{{i+1}}^2}}\\,\\varepsilon$ on $E$,\n    re-inject $V$ at $\\sigma_i$, go to 1', fs=9.3, ha='left')
yb_ = 1.75; P = [.6, 3.95, 7.3, 10.65]; yc = yb_ + h_ / 2; BW = 1.2
f.mp(P[0], yb_, w_, h_, '(e)', 'State $x_{\\sigma_i}$', f'{SIG(0)}', ev('x_in', 0), 'state', CB_N, left=True)
f.mp(P[1], yb_, w_, h_, '(f)', 'Denoised $D(\\cdot)$', '', kel(ev('den', 0)), 'sst', CB_S)
f.mp(P[2], yb_, w_, h_, '(g)', 'After Heun step', f'{SIG(1)}', ev('x_heun', 0), 'state', CB_N)
f.mp(P[3], yb_, w_, h_, '(h)', 'After re-injection', f'$x_{{\\sigma_{{i+1}}}}$, {SIG(1)}', ev('x_next', 0), 'state', CB_N, outline=V)
bxs = [P[j] + w_ + .27 for j in range(3)]; names = [('Temporal\nU-FNO', ORNG, '#FBEBDD'), ('Heun\npred.–corr.', GREEN, '#EAF5EC'), ('Re-inject\n$V$', BLUE, '#EEF3FA')]
for j in range(3):
    f.box(bxs[j], yc - .55, BW, 1.1, names[j][0], names[j][1], names[j][2], fs=10); f.arr((P[j] + w_ + .03, yc), (bxs[j] - .02, yc)); f.arr((bxs[j] + BW + .02, yc), (P[j + 1] - .03, yc))
xu, xr = bxs[0] + BW / 2, bxs[2] + BW / 2
f.ln([10.3, 10.3], [yt + .3, 4.85], BLUE); f.ln([xu, 10.3], [4.85, 4.85], BLUE); f.arr((xu, 4.85), (xu, yc + .57), BLUE)
f.ln([10.3, xr], [4.85, 4.85], BLUE); f.arr((xr, 4.85), (xr, yc + .57), BLUE); f.txt(6.5, 4.98, 'condition $c$ and history at every network call', fs=9.5, c=BLUE, va='bottom')
xh, xe = P[3] + 1.52, P[0] + 1.52
f.ln([xh, xh], [yb_ - .02, .78], GRY); f.ln([xh, xe], [.78, .78], GRY); f.arr((xe, .78), (xe, yb_ - .02), GRY)
f.txt(7.0, .56, f'jump back (traversal 1 → 2): add noise to $\\sigma_i$ on $E$, re-inject $V$, repeat the step (U={U} traversals); then next level $\\sigma_{{i+1}}\\to\\sigma_{{i+2}}$ …', fs=10, c='#555', va='top')
f.mp(13.95, yb_, 1.65, h_, '(i)', 'Final result', '($\\sigma=0$, all levels done)', kel(T['x_final']), 'sst', CB_S)
f.arr((P[3] + w_ + .1, yc), (13.9, yc)); f.txt(13.1, yc + .5, 'remaining\nnoise levels', fs=9, c=GRY)
f.save('7_full')
