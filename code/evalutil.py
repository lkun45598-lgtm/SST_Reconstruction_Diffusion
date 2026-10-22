import json, numpy as np, torch, torch.nn.functional as F
from ufno import UFNO
from edm import Denoiser


def load_model(run_dir, dev='cuda'):
    ck = torch.load(f'{run_dir}/ckpt.pt', map_location=dev)
    net = UFNO().to(dev); den = Denoiser(net).to(dev); den.load_state_dict(ck['model']); den.eval()
    return den, ck['scale'], ck['step']


def laplace_fill(x, V, domain, iters=1500):
    """Baseline: harmonic (Laplace) interpolation inside `domain`, visible values V held fixed."""
    Vf, D = V.float(), domain.float()
    mu = (torch.nan_to_num(x) * Vf).sum((1, 2, 3), keepdim=True) / Vf.sum((1, 2, 3), keepdim=True).clamp_min(1)
    f = torch.where(V, torch.nan_to_num(x), mu.expand_as(x)) * D
    k = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], device=x.device, dtype=x.dtype)[None, None]
    cnt = F.conv2d(F.pad(D, (1, 1, 1, 1)), k).clamp_min(1)
    for _ in range(iters):
        nb = F.conv2d(F.pad(f * D, (1, 1, 1, 1)), k) / cnt
        f = torch.where(V, torch.nan_to_num(x), nb) * D
    return f


def grad_mag(f):
    gx = f[..., :, 1:] - f[..., :, :-1]; gy = f[..., 1:, :] - f[..., :-1, :]
    return gx, gy


def hole_metrics(pred, truth, H_, ens_axis=None):
    """pred/truth (B,1,H,W) kelvin; H_ bool hidden mask (truth genuine there). Per-patch sums so that
    pooled and bootstrapped statistics can be built. Returns dict of per-patch arrays."""
    Hf = H_.float()
    n = Hf.sum((1, 2, 3))
    e = (pred - truth) * Hf
    # interior-of-hole gradient magnitude ratio (both pixels of each difference inside the hole)
    px = pred[..., :, 1:] - pred[..., :, :-1]; tx = truth[..., :, 1:] - truth[..., :, :-1]
    mx = Hf[..., :, 1:] * Hf[..., :, :-1]
    py = pred[..., 1:, :] - pred[..., :-1, :]; ty = truth[..., 1:, :] - truth[..., :-1, :]
    my = Hf[..., 1:, :] * Hf[..., :-1, :]
    return dict(n=n.cpu().numpy(), se=(e ** 2).sum((1, 2, 3)).cpu().numpy(), ae=e.abs().sum((1, 2, 3)).cpu().numpy(),
                bias=e.sum((1, 2, 3)).cpu().numpy(),
                gp=((px.abs() * mx).sum((1, 2, 3)) + (py.abs() * my).sum((1, 2, 3))).cpu().numpy(),
                gt=((tx.abs() * mx).sum((1, 2, 3)) + (ty.abs() * my).sum((1, 2, 3))).cpu().numpy(),
                gn=(mx.sum((1, 2, 3)) + my.sum((1, 2, 3))).cpu().numpy())


def summarize(m):
    n = m['n'].sum()
    return dict(rmse=float(np.sqrt(m['se'].sum() / n)), mae=float(m['ae'].sum() / n), bias=float(m['bias'].sum() / n),
                grad_ratio=float(m['gp'].sum() / max(m['gt'].sum(), 1e-9)))


def boot_ratio(ma, mb, nb=2000, seed=0):
    """paired bootstrap over patches of rmse_a - rmse_b and rmse_a / rmse_b"""
    rng = np.random.default_rng(seed); N = len(ma['n']); d, r = [], []
    for _ in range(nb):
        i = rng.integers(0, N, N)
        a = np.sqrt(ma['se'][i].sum() / ma['n'][i].sum()); b = np.sqrt(mb['se'][i].sum() / mb['n'][i].sum())
        d.append(a - b); r.append(a / b)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(diff=float(np.mean(d)), diff_ci=q(d), ratio=float(np.mean(r)), ratio_ci=q(r))


def select_items(jx, lib, ocean, n_frames, seed=7, n_build=None):
    """Test frames + real cloud shapes. Returns [(frame_idx, K, H, V)]: K genuine obs, H = K & cloud (hidden,
    scored), V = K & ~cloud (visible). Deterministic given seed; items[i] does not depend on items[j>i]."""
    rng = np.random.default_rng(seed)
    cand = jx.idx['test']; obs = (~torch.isnan(jx.x[cand])).float().mean((1, 2)) / ocean.float().mean()
    cand = cand[(obs >= 0.6).cpu().numpy()]
    frames = rng.choice(cand, min(n_frames, len(cand)), replace=False)
    items = []
    for f in frames[:n_build]:
        K = (~torch.isnan(jx.x[f]))[None, None]
        for _ in range(200):
            P = lib.pat[int(rng.integers(0, lib.pat.shape[0]))][None, None]
            if rng.random() < .5: P = P.flip(3)
            if rng.random() < .5: P = P.flip(2)
            H_ = K & P; fr = H_.float().sum() / K.float().sum()
            if 0.10 <= fr <= 0.40: break
        items.append((int(f), K, H_, K & ~P))
    return items
