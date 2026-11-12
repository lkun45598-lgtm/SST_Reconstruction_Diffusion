"""EDM denoiser wrapper, masked-data objective and Heun sampler.

Pixel roles in one sample (all on the model grid):
  E  "exists"  : pixels that take part in the diffusion state (ocean and, for JAXA training,
                 genuinely observed). Outside E the noisy state is zero and nothing is supervised.
  V  visible   : subset of E whose clean value is given to the network as condition y.
  H  hidden    : E and not V; the loss is computed ONLY here (their clean values are real data).
Pixels that are neither in E (land, true JAXA gaps) are never used as labels.
Field normalisation: x_norm = (x_K - mu_V) / scale, mu_V = mean of the visible values of the sample.
"""
import numpy as np
import torch
import torch.nn.functional as F
from ufno import pad_to_div

P_MEAN, P_STD = -1.2, 1.2
SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80.0, 7.0


def visible_mean(x, V):
    """x (B,1,H,W) kelvin (NaN allowed outside V), V bool."""
    Vf = V.float()
    return (torch.nan_to_num(x) * Vf).sum((1, 2, 3), keepdim=True) / Vf.sum((1, 2, 3), keepdim=True).clamp_min(1)


def build_history_cond(history_x, history_mask, mu, scale):
    """Past frames in the SAME normalisation as the current target (its visible mean mu and scale; no per-frame mean).
    history_x (B,T,1,H,W) kelvin (NaN allowed), history_mask (B,T,1,H,W) bool genuine-observation mask, mu (B,1,1,1).
    -> (B,T,2,H,W) = [normalised SST * valid, valid]. Invalid pixels are exactly 0 with mask 0."""
    m = history_mask.float()
    hn = torch.nan_to_num((history_x - mu.unsqueeze(1)) / scale) * m
    return torch.cat([hn, m], 2)


class Denoiser(torch.nn.Module):
    def __init__(self, net, sigma_data=0.5):
        super().__init__()
        self.net, self.sd = net, sigma_data

    def forward(self, xt, sigma, y, V, E, geo, history=None):
        """xt (B,1,H,W) noisy normalised state (zero outside E); y normalised visible values.
        history: optional fixed (never noised) (B,T,2,H,W) condition from build_history_cond.
        Returns the denoised estimate of the clean normalised field (zero outside E)."""
        sd = self.sd
        s = sigma.view(-1, 1, 1, 1)
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / (s ** 2 + sd ** 2).sqrt()
        c_in = 1 / (s ** 2 + sd ** 2).sqrt()
        Vf, Ef = V.float(), E.float()
        cond = torch.cat([y * Vf, Vf, Ef, geo['coords']], 1)
        x_in = c_in * xt * Ef
        div = self.net.div
        x_in, (H, W) = pad_to_div(x_in, div)
        cond, _ = pad_to_div(cond, div)
        if history is not None: history, _ = pad_to_div(history, div)   # same top-left alignment as x_in/cond
        Hp, Wp = x_in.shape[-2:]
        extent = geo['extent'] * torch.tensor([Hp / H, Wp / W], device=x_in.device)
        f = self.net(x_in, sigma.log().flatten() / 4, cond, geo['spacing'], extent, history=history)[..., :H, :W]
        return (c_skip * xt + c_out * f) * Ef


def masked_loss(den, x0, V, H_, E, geo, scale, history_x=None, history_mask=None):
    """x0 kelvin (B,1,H,W); V,H_,E bool. Loss only on hidden pixels H_ (genuine values).
    history_x/history_mask (B,T,1,H,W): past frames as fixed condition only (never a label, never noised)."""
    B = x0.shape[0]
    mu = visible_mean(x0, V)
    xn = torch.nan_to_num((x0 - mu) / scale) * E.float()
    sigma = (torch.randn(B, device=x0.device) * P_STD + P_MEAN).exp()
    s = sigma.view(-1, 1, 1, 1)
    xt = (xn + s * torch.randn_like(xn)) * E.float()
    y = xn * V.float()
    hc = None if history_x is None else build_history_cond(history_x, history_mask, mu, scale)
    d = den(xt, sigma, y, V, E, geo, history=hc)
    sd = getattr(den, 'module', den).sd
    w = (s ** 2 + sd ** 2) / (s * sd) ** 2
    Hf = H_.float()
    err = ((d - xn) ** 2 * Hf * w).sum((1, 2, 3)) / Hf.sum((1, 2, 3)).clamp_min(1)
    return err.mean(), dict(sigma=sigma.detach())


@torch.no_grad()
def sample(den, x_obs, V, E, geo, scale, n_samples=4, steps=32, generator=None, history_x=None, history_mask=None, reinject=False, resample=1, return_trajectory=False, trajectory_steps=None, save_dir=None):
    """Heun sampler. x_obs (B,1,H,W) kelvin (values used only where V), V/E bool (B,1,H,W).
    Returns kelvin samples (B,n,1,H,W). E is the whole generation domain (e.g. all ocean).
    history_x/history_mask (B,T,1,H,W): fixed past-frame condition, identical for all n samples and every Heun step.
    reinject (G1): after the initial draw and after every Heun step set the state on V to y + sigma_next*eps (fresh eps), i.e. the
    training-time state xt=(xn+sigma*eps)*E on visible pixels. Only V is used (y is already zero outside V); hidden/true-gap pixels untouched.
    resample (G2, RePaint jumps; needs reinject): every noise level is traversed `resample` times; between traversals the state is
    re-noised from sigma_next back to sigma_cur (x += sqrt(s0^2-s1^2)*eps on E) and V is re-injected at sigma_cur. resample=1: no jumps.
    return_trajectory (figure export, default off, sampling unchanged): also return a dict of numpy arrays for sample 0 of the batch,
    all in the NORMALISED space of the network (x_norm = (x_K - mu)/scale, zero outside E), plus mu/scale/y/V/E:
      sigma (steps+1,), x_init (state before step 0), and per recorded step k (trajectory_steps=None -> all):
      x_in[k]=x_{sigma_k}, den[k]=D(x_{sigma_k},sigma_k) (first Heun evaluation), x_euler[k], den2[k] (2nd evaluation, NaN at sigma=0),
      x_next[k]=x_{sigma_k+1} (after the Heun correction and, if reinject, the re-injection of V).
      ev_<k>_<u>_<name> (every traversal u of a recorded step; name in x_in, den, x_euler, den2, x_heun (before re-injection), x_next,
      x_jump_noised (after the jump-back noise), x_jump (after re-injection at sigma_k)); ev_x_init_raw = initial draw before re-injection. save_dir: also dump the dict as save_dir/trajectory.npz.
    """
    assert resample == 1 or reinject, 'resample>1 (G2) requires reinject (G1)'
    dev = x_obs.device
    B = x_obs.shape[0]
    mu = visible_mean(x_obs, V)
    y = torch.nan_to_num((x_obs - mu) / scale) * V.float()
    rep = lambda t: t.repeat_interleave(n_samples, 0)
    y, Vn, En = rep(y), rep(V), rep(E)
    hc = None if history_x is None else rep(build_history_cond(history_x, history_mask, mu, scale))
    g = dict(coords=rep(geo['coords']), spacing=rep(geo['spacing']), extent=geo['extent'])
    i = torch.arange(steps, device=dev, dtype=torch.float64)
    ts = (SIGMA_MAX ** (1 / RHO) + i / (steps - 1) * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO
    ts = torch.cat([ts, torch.zeros(1, device=dev, dtype=torch.float64)]).float()
    N = B * n_samples
    x = torch.randn(N, 1, *x_obs.shape[-2:], device=dev, generator=generator) * ts[0] * En.float()
    Vf = Vn.float()
    def _reinj(x, sig):
        if not reinject: return x
        return x * (1 - Vf) + (y + sig * torch.randn(x.shape, device=dev, generator=generator)) * Vf
    x_raw = x
    x = _reinj(x, ts[0])
    Ef = En.float()
    rec = {} if return_trajectory else None
    keep = set(range(steps)) if trajectory_steps is None else set(int(t) for t in trajectory_steps)
    ev = {}
    if rec is not None: rec['x_init'] = x[0, 0].cpu().numpy(); ev['x_init_raw'] = x_raw[0, 0].cpu().numpy()
    for k in range(steps):
        t0, t1 = ts[k], ts[k + 1]
        for u in range(resample):
            d = den(x, torch.full((N,), float(t0), device=dev), y, Vn, En, g, history=hc); dx = (x - d) / t0
            xn = x + (t1 - t0) * dx
            xe, d2 = xn, None
            if t1 > 0:
                d2 = den(xn, torch.full((N,), float(t1), device=dev), y, Vn, En, g, history=hc)
                xn = x + (t1 - t0) * 0.5 * (dx + (xn - d2) / t1)
            x_prev, x_heun = x, xn * Ef; x = _reinj(x_heun, t1)
            if rec is not None and k in keep:
                for nm, v in (('x_in', x_prev), ('den', d), ('x_euler', xe), ('den2', d2), ('x_heun', x_heun), ('x_next', x)):
                    if v is not None: ev[f'{k}_{u}_{nm}'] = v[0, 0].cpu().numpy()
            if rec is not None and k in keep and u == resample - 1:
                nan = torch.full_like(xe, float('nan'))
                for nm, v in (('x_in', x_prev), ('den', d), ('x_euler', xe), ('den2', nan if d2 is None else d2), ('x_next', x)):
                    rec.setdefault(nm, {})[k] = v[0, 0].cpu().numpy()
            if u < resample - 1 and t1 > 0:   # jump back to sigma_cur
                x = x + (t0 ** 2 - t1 ** 2).sqrt() * torch.randn(x.shape, device=dev, generator=generator) * Ef
                if rec is not None and k in keep: ev[f'{k}_{u}_x_jump_noised'] = x[0, 0].cpu().numpy()
                x = _reinj(x, t0)
                if rec is not None and k in keep: ev[f'{k}_{u}_x_jump'] = x[0, 0].cpu().numpy()
    out = x * scale + rep(mu)
    out = out.view(B, n_samples, 1, *x_obs.shape[-2:])
    if not return_trajectory: return out
    ks = sorted(rec['x_in'])
    traj = dict(sigma=ts.cpu().numpy(), steps=np.array(ks), x_init=rec['x_init'], mu=float(mu[0]), scale=float(scale),
                y=y[0, 0].cpu().numpy(), V=Vn[0, 0].cpu().numpy(), E=En[0, 0].cpu().numpy(), x_final=x[0, 0].cpu().numpy(),
                **{nm: np.stack([rec[nm][k] for k in ks]) for nm in ('x_in', 'den', 'x_euler', 'den2', 'x_next')}, resample=resample, reinject=reinject,
                **{f'ev_{kk}': v for kk, v in ev.items()})
    if save_dir is not None:
        import os; os.makedirs(save_dir, exist_ok=True); np.savez_compressed(f'{save_dir}/trajectory.npz', **traj)
    return out, traj
