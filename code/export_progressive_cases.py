"""Record the REAL diffusion states x_sigma of the current sampler (Temporal-12 U-FNO, EDM Heun, G2 reinject + resample jumps) for several test cases.
Also saves the denoiser clean-field estimate D(x_sigma_k, sigma_k; c) of the same first call (already computed by the sampler, no extra forward pass), inverse-normalised with the case's own mu_V and the checkpoint scale.
Recorded state of level k = the tensor entering the FIRST network call of that level (= output of level k-1 after Heun, re-injection and all G2
traversals). Initial = level 0 (after the initial draw and re-injection). Final = the actual `sample()` output. No sampling code is changed.
Config: progressive_config.json ("rows" = JaxaTemporal window rows to pin the cases, "stage_levels" = explicit level indices).
Checks: consistency (recording on == off, bitwise), oracle (saved states == tensors seen by the denoiser via a forward pre-hook)."""
import argparse, json, os, numpy as np, torch
from data import CloudLib, JaxaTemporal
from edm import sample
from geo import patch_geometry
from evalutil import load_model

p = argparse.ArgumentParser(); p.add_argument('--config', default='progressive_config.json'); p.add_argument('--n-cases', type=int, default=None)
a = p.parse_args(); cfg = json.load(open(a.config)); dev = 'cuda'; os.makedirs(cfg['out'], exist_ok=True)
lib = CloudLib(dev); jt = JaxaTemporal(dev, lib); land = lib.land.cpu().numpy(); ocean_t = (~lib.land)[None, None]; of = float(ocean_t.float().mean())
den, sc, ckstep = load_model(cfg['run'], dev); assert den.net.temporal_enabled
S, U = cfg['steps'], cfg['U']

# ---- case selection (fixed seed, deterministic; or pinned rows from the config)
if cfg['rows']: rows = list(cfg['rows'])
else:
    cand = jt.idx['test']; fr = jt.finite_frac[jt.win[cand, -1]] / of; lo, hi = cfg['obs_frac_range']
    cand, fr = cand[(fr >= lo) & (fr <= hi)], fr[(fr >= lo) & (fr <= hi)]
    sub = np.random.default_rng(cfg['selection_seed']).choice(len(cand), min(400, len(cand)), replace=False)
    cand, fr = cand[sub], fr[sub]; o = np.argsort(fr); n = a.n_cases or cfg['n_cases']
    rows, months = [], set()
    for q in np.linspace(0.05, 0.95, n):                       # spread over cloud cover, prefer different months
        i0 = int(q * (len(o) - 1))
        for d in sorted(range(-40, 41), key=abs):
            j = o[min(max(i0 + d, 0), len(o) - 1)]; mth = jt.times[int(jt.win[cand[j], -1])].month
            if mth not in months: months.add(mth); rows.append(int(cand[j])); break
if a.n_cases: rows = rows[:a.n_cases]
print('case rows', rows)
lvl = cfg['stage_levels'] or {nm: int(round(pr * S)) for nm, pr in cfg['stage_progress'].items()}   # nearest valid level index (0..S-1)
lvl = {'initial': 0, **{k: min(max(v, 0), S - 1) for k, v in lvl.items()}}
levels = sorted(set(lvl.values())); print('stage levels', lvl)
geo = patch_geometry(jt.lat[None], jt.lon[None]); lat, lon = jt.lat.cpu().numpy(), jt.lon.cpu().numpy()
report = dict(rows=rows, stage_levels=lvl, checks=[])
for ci, row in enumerate(rows):
    hx, hm = jt.history_full(row); tgt = int(jt.win[row, -1]); t = str(jt.times[tgt])
    x = torch.from_numpy(np.asarray(jt.frames[tgt])).to(dev)[None, None]; V = (~torch.isnan(x)) & ocean_t
    kw = dict(history_x=hx, history_mask=hm, reinject=True, resample=U)
    gen = lambda: torch.Generator(device=dev).manual_seed(cfg['seed'] + ci)
    ref = sample(den, x, V, ocean_t, geo, sc, 1, S, generator=gen(), **kw)                 # recording OFF
    cap = {}; capd = {}; cnt = [0]; want = {2 * U * k: k for k in levels}
    def hook(mod, args):
        if cnt[0] in want: cap[want[cnt[0]]] = (args[0][0, 0].detach().cpu().numpy().copy(), float(args[1][0]))
        cnt[0] += 1
    def hookd(mod, args, outp):
        if cnt[0] - 1 in want: capd[want[cnt[0] - 1]] = outp[0, 0].detach().cpu().numpy().copy()
    h = den.register_forward_pre_hook(hook); h2 = den.register_forward_hook(hookd)
    out, tr = sample(den, x, V, ocean_t, geo, sc, 1, S, generator=gen(), return_trajectory=True, trajectory_steps=levels, **kw)   # recording ON
    h.remove(); h2.remove()
    sig = tr['sigma']; consistent = bool(torch.equal(out, ref))
    states = {k: tr[f'ev_{k}_0_x_in'] for k in levels}
    oracle = {k: bool(np.array_equal(states[k], cap[k][0]) and abs(cap[k][1] - sig[k]) < 1e-6) for k in levels}
    est = {k: tr[f'ev_{k}_0_den'] for k in levels}; oracle_den = {k: bool(np.array_equal(est[k], capd[k])) for k in levels}
    est_C = {k: (est[k] * float(sc) + tr['mu'] - 273.15).astype('float32') for k in levels}
    assert all(np.isfinite(v[tr['E']]).all() for v in est_C.values()), 'non-finite estimate'
    fin = out[0, 0, 0].cpu().numpy() - 273.15; fin_ok = bool(np.array_equal(fin, (tr['x_final'] * sc + tr['mu']) - 273.15)) or float(np.abs(fin - ((tr['x_final'] * sc + tr['mu']) - 273.15)).max()) < 1e-3
    save = dict(row=row, time_utc=t, target_index=tgt, lat=lat, lon=lon, land=land, V=V[0, 0].cpu().numpy(), E=tr['E'], y=tr['y'], mu_K=tr['mu'], scale=sc,
                input_sst_C=np.where(V[0, 0].cpu().numpy(), x[0, 0].cpu().numpy() - 273.15, np.nan).astype('float32'), final_sst_C=fin.astype('float32'),
                sigma_schedule=sig, stage_names=np.array(list(lvl)), stage_levels=np.array(list(lvl.values())), stage_sigma=np.array([sig[k] for k in lvl.values()]),
                obs_frac=float(V.float().sum() / ocean_t.float().sum()), steps=S, U=U, seed=cfg['seed'] + ci, ckpt=cfg['run'], ckpt_step=int(ckstep),
                **{f'state_{nm}': states[k].astype('float32') for nm, k in lvl.items()},
                **{f'rp_{nm}_{key}': tr[f'ev_{k}_{u_}_{src}'].astype('float32') for nm, k in lvl.items() for key, u_, src in (('heun', 0, 'x_heun'), ('reinj', 0, 'x_next'), ('jump_noised', 0, 'x_jump_noised'), ('jump_reinj', 0, 'x_jump'), ('trav_last', U - 1, 'x_next')) if f'ev_{k}_{u_}_{src}' in tr},
                **{f'est_norm_{nm}': est[k].astype('float32') for nm, k in lvl.items()}, **{f'est_sst_C_{nm}': est_C[k] for nm, k in lvl.items()})
    np.savez_compressed(f'{cfg["out"]}/case{ci + 1}_row{row}.npz', **save)
    chk = dict(case=ci + 1, row=row, time=t, obs_frac=save['obs_frac'], consistency_on_vs_off_bitwise=consistent, oracle_states_equal=oracle, oracle_denoiser_output_equal=oracle_den, final_matches_state_x_final=fin_ok)
    report['checks'].append(chk); print(chk, flush=True)
    assert consistent and all(oracle.values()) and all(oracle_den.values()), 'CHECK FAILED'
json.dump(report, open(f'{cfg["out"]}/checks.json', 'w'), indent=1); print('saved', cfg['out'])
