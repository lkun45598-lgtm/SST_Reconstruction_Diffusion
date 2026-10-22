"""EDM denoiser wrapper, masked-data objective and Heun sampler.

Pixel roles in one sample (all on the model grid):
  E  "exists"  : pixels that take part in the diffusion state (ocean and, for JAXA training,
                 genuinely observed). Outside E the noisy state is zero and nothing is supervised.
  V  visible   : subset of E whose clean value is given to the network as condition y.
  H  hidden    : E and not V; the loss is computed ONLY here (their clean values are real data).
Pixels that are neither in E (land, true JAXA gaps) are never used as labels.
Field normalisation: x_norm = (x_K - mu_V) / scale, mu_V = mean of the visible values of the sample.
"""
import torch
import torch.nn.functional as F
from ufno import pad_to_div

P_MEAN, P_STD = -1.2, 1.2
SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80.0, 7.0


def visible_mean(x, V):
    """x (B,1,H,W) kelvin (NaN allowed outside V), V bool."""
    Vf = V.float()
    return (torch.nan_to_num(x) * Vf).sum((1, 2, 3), keepdim=True) / Vf.sum((1, 2, 3), keepdim=True).clamp_min(1)


class Denoiser(torch.nn.Module):
    def __init__(self, net, sigma_data=0.5):
        super().__init__()
        self.net, self.sd = net, sigma_data

    def forward(self, xt, sigma, y, V, E, geo):
        """xt (B,1,H,W) noisy normalised state (zero outside E); y normalised visible values.
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
        Hp, Wp = x_in.shape[-2:]
        extent = geo['extent'] * torch.tensor([Hp / H, Wp / W], device=x_in.device)
        f = self.net(x_in, sigma.log().flatten() / 4, cond, geo['spacing'], extent)[..., :H, :W]
        return (c_skip * xt + c_out * f) * Ef


def masked_loss(den, x0, V, H_, E, geo, scale):
    """x0 kelvin (B,1,H,W); V,H_,E bool. Loss only on hidden pixels H_ (genuine values)."""
    B = x0.shape[0]
    mu = visible_mean(x0, V)
    xn = torch.nan_to_num((x0 - mu) / scale) * E.float()
    sigma = (torch.randn(B, device=x0.device) * P_STD + P_MEAN).exp()
    s = sigma.view(-1, 1, 1, 1)
    xt = (xn + s * torch.randn_like(xn)) * E.float()
    y = xn * V.float()
    d = den(xt, sigma, y, V, E, geo)
    sd = getattr(den, 'module', den).sd
    w = (s ** 2 + sd ** 2) / (s * sd) ** 2
    Hf = H_.float()
    err = ((d - xn) ** 2 * Hf * w).sum((1, 2, 3)) / Hf.sum((1, 2, 3)).clamp_min(1)
    return err.mean(), dict(sigma=sigma.detach())


@torch.no_grad()
def sample(den, x_obs, V, E, geo, scale, n_samples=4, steps=32, generator=None):
    """Heun sampler. x_obs (B,1,H,W) kelvin (values used only where V), V/E bool (B,1,H,W).
    Returns kelvin samples (B,n,1,H,W). E is the whole generation domain (e.g. all ocean)."""
    dev = x_obs.device
    B = x_obs.shape[0]
    mu = visible_mean(x_obs, V)
    y = torch.nan_to_num((x_obs - mu) / scale) * V.float()
    rep = lambda t: t.repeat_interleave(n_samples, 0)
    y, Vn, En = rep(y), rep(V), rep(E)
    g = dict(coords=rep(geo['coords']), spacing=rep(geo['spacing']), extent=geo['extent'])
    i = torch.arange(steps, device=dev, dtype=torch.float64)
    ts = (SIGMA_MAX ** (1 / RHO) + i / (steps - 1) * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO
    ts = torch.cat([ts, torch.zeros(1, device=dev, dtype=torch.float64)]).float()
    N = B * n_samples
    x = torch.randn(N, 1, *x_obs.shape[-2:], device=dev, generator=generator) * ts[0] * En.float()
    for k in range(steps):
        t0, t1 = ts[k], ts[k + 1]
        d = den(x, torch.full((N,), float(t0), device=dev), y, Vn, En, g); dx = (x - d) / t0
        xn = x + (t1 - t0) * dx
        if t1 > 0:
            d2 = den(xn, torch.full((N,), float(t1), device=dev), y, Vn, En, g)
            xn = x + (t1 - t0) * 0.5 * (dx + (xn - d2) / t1)
        x = xn * En.float()
    out = x * scale + rep(mu)
    return out.view(B, n_samples, 1, *x_obs.shape[-2:])
