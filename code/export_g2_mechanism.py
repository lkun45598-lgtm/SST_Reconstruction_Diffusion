"""Export the REAL Temporal-12 + G2 (reinject + resample jumps) sampling trajectory on one real JAXA target hour with its genuine cloud gaps
(V = observed ocean pixels of the target hour, E = all ocean, history = the 11 previous hours). Same setup as the real-gap checks.
Verifies that trajectory recording does not change the sampled field. Output: <out>/{trajectory.npz,history.npz,land.npy,time_bias.npy,meta.json}."""
import argparse, os, json, numpy as np, torch
from data import CloudLib, JaxaTemporal
from edm import sample
from geo import patch_geometry
from evalutil import load_model

p = argparse.ArgumentParser()
p.add_argument('--run', default='../runs/jaxa_temporal12'); p.add_argument('--out', default='../runs/mechanism_g2')
p.add_argument('--row', type=int, default=None, help='window row index in JaxaTemporal (default: first test window with 30-45%% of ocean observed, fixed seed)')
p.add_argument('--steps', type=int, default=32); p.add_argument('--U', type=int, default=2); p.add_argument('--show-step', type=int, default=15); p.add_argument('--seed', type=int, default=0)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True); dev = 'cuda'
lib = CloudLib(dev); jt = JaxaTemporal(dev, lib); land = lib.land.cpu().numpy(); ocean_t = (~lib.land)[None, None]
den, sc, ckstep = load_model(a.run, dev); assert den.net.temporal_enabled
if a.row is None:
    cand = jt.idx['test']; fr = jt.finite_frac[jt.win[cand, -1]] / float(ocean_t.float().mean())
    hfr = np.stack([jt.finite_frac[jt.win[cand, j]] for j in range(11)], 1).mean(1) / float(ocean_t.float().mean())
    ok = cand[(fr >= .30) & (fr <= .45) & (hfr >= .35)]; a.row = int(np.random.default_rng(3).choice(ok))
hx, hm = jt.history_full(a.row); tgt = int(jt.win[a.row, -1])
x = torch.from_numpy(np.asarray(jt.frames[tgt])).to(dev)[None, None]; V = (~torch.isnan(x)) & ocean_t
geo = patch_geometry(jt.lat[None], jt.lon[None])
g = lambda: torch.Generator(device=dev).manual_seed(a.seed)
kw = dict(history_x=hx, history_mask=hm, reinject=True, resample=a.U)
ref = sample(den, x, V, ocean_t, geo, sc, 1, a.steps, generator=g(), **kw)
out, tr = sample(den, x, V, ocean_t, geo, sc, 1, a.steps, generator=g(), return_trajectory=True, trajectory_steps=[a.show_step], save_dir=a.out, **kw)
print('trajectory-on vs off max |diff| (G2):', float((out - ref).abs().max()))
t = jt.times[tgt]; print('target', t, 'row', a.row, 'observed', float(V.float().sum() / ocean_t.float().sum()), 'sigma', tr['sigma'][a.show_step], '->', tr['sigma'][a.show_step + 1])
hk = hx[0, :, 0].cpu().numpy() - 273.15
np.savez_compressed(f'{a.out}/history.npz', hist=hk.astype('float32'), hist_mask=hm[0, :, 0].cpu().numpy(), times=np.array([str(jt.times[int(i)]) for i in jt.win[a.row]]))
np.save(f'{a.out}/land.npy', land); np.save(f'{a.out}/target_obs.npy', np.where(V[0, 0].cpu().numpy(), x[0, 0].cpu().numpy() - 273.15, np.nan))
sd = torch.load(f'{a.run}/ckpt.pt', map_location='cpu')['model']; np.save(f'{a.out}/time_bias.npy', sd['net.temporal.time_bias'].numpy())
json.dump(dict(time=str(t), row=a.row, ckpt=a.run, ckpt_step=int(ckstep), steps=a.steps, U=a.U, k=a.show_step, observed_frac=float(V.float().sum() / ocean_t.float().sum()),
               hist_obs_frac=[float(hm[0, j, 0].float().sum() / ocean_t.float().sum()) for j in range(hm.shape[1])], scale=float(sc), mu_K=float(tr['mu'])), open(f'{a.out}/meta.json', 'w'), indent=1)
print('saved', a.out)
