"""Step 2/4 smoke tests: forward/backward, physical-extent spectral consistency, variable resolution."""
import torch, time
from ufno import UFNO, SpectralConv2d
from edm import Denoiser, masked_loss, sample
from geo import patch_geometry

dev = 'cuda'
net = UFNO().to(dev); den = Denoiser(net).to(dev)
print('params (M):', sum(p.numel() for p in net.parameters()) / 1e6)

def make_geo(h, w, step, lat0=18.0, lon0=113.0, B=2):
    lat = (lat0 - step * torch.arange(h, device=dev))[None].repeat(B, 1)
    lon = (lon0 + step * torch.arange(w, device=dev))[None].repeat(B, 1)
    return patch_geometry(lat, lon)

# 1. same 3.2 deg extent, different pixel counts
for name, n, step in [('OSTIA 0.05deg', 64, 0.05), ('JAXA 0.02deg', 160, 0.02)]:
    g = make_geo(n, n, step)
    x0 = 300 + torch.randn(2, 1, n, n, device=dev) * 0.5
    E = torch.ones_like(x0, dtype=torch.bool); V = torch.rand_like(x0) > 0.5
    loss, _ = masked_loss(den, x0, V, E & ~V, E, g, 0.5)
    loss.backward()
    gn = sum(p.grad.norm() ** 2 for p in net.parameters() if p.grad is not None).sqrt()
    print(f'{name}: grid {n}x{n} extent_km {g["extent"].tolist()} loss {loss.item():.3f} grad_norm {gn:.3f}')
    net.zero_grad()

# 2. spectral conv: same smooth physical field sampled at two resolutions -> consistent output
sc = SpectralConv2d(4, 12).to(dev).eval()
def field(n, L):
    t = torch.linspace(0, 1, n + 1, device=dev)[:-1]
    X, Y = torch.meshgrid(t, t, indexing='ij')
    f = torch.stack([torch.sin(2 * 3.14159 * 2 * X) * torch.cos(2 * 3.14159 * 3 * Y),
                     torch.cos(2 * 3.14159 * X), torch.sin(2 * 3.14159 * 4 * Y), X * 0 + 1])[None]
    return f
with torch.no_grad():
    a = sc(field(64, 350), (350., 350.)); b = sc(field(128, 350), (350., 350.))[..., ::2, ::2]
    print('spectral conv 64 vs 128 (same extent, integer ratio, subsampled) rel diff:', ((a - b).norm() / a.norm()).item())
    # 160/64 = 2.5 is non-integer: compare against exact pointwise samples at the shared points (every 5th of 160 vs every 2nd of 64)
    c160 = sc(field(160, 350), (350., 350.))[..., ::5, ::5]; a32 = sc(field(64, 350), (350., 350.))[..., ::2, ::2]
    print('spectral conv 160 vs 64 at shared points rel diff:', ((c160 - a32).norm() / a32.norm()).item())
    # larger extent keeps physical wavenumber: field with a wavelength of 350 km on a 700 km domain
    c = sc(field(64, 700)[..., :], (700., 700.))
    print('extent 700km runs, out std', c.std().item())

# 3. variable resolution inference with identical weights (full JAXA grid is not /8)
net.eval()
for name, h, w, step in [('OSTIA patch 64', 64, 64, 0.05), ('JAXA patch 160', 160, 160, 0.02),
                         ('JAXA full 451x351', 451, 351, 0.02), ('OSTIA full-ish 180x140', 180, 140, 0.05)]:
    g = make_geo(h, w, step, B=1)
    x = 300 + torch.randn(1, 1, h, w, device=dev) * 0.5
    E = torch.ones_like(x, dtype=torch.bool); V = torch.rand_like(x) > 0.6
    t = time.time()
    s = sample(den, x, V, E, g, 0.5, n_samples=2, steps=4)
    torch.cuda.synchronize()
    print(f'{name}: out {tuple(s.shape)} finite {bool(torch.isfinite(s).all())} {time.time()-t:.1f}s')
print('ALL OK')
