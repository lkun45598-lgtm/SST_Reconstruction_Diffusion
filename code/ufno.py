"""Resolution-aware U-shaped FNO denoiser.

* U-shaped conv encoder/decoder (local, multiscale) with FiLM from [noise level, grid spacing].
* FNO blocks at every scale (global dependencies). Spectral weights are defined on PHYSICAL
  wavenumbers: weight index i corresponds to wavenumber i / L_REF, and for a field of physical
  extent L the integer frequency j (wavenumber j / L) reads the weights by linear interpolation at
  index j * L_REF / L. Same weights therefore keep the same physical meaning on a 64x64 / 0.05deg
  patch, a 160x160 / 0.02deg patch of the same extent, or the full 9x7 deg JAXA grid.
* Grid spacing (log dx, log dy, per scale) enters every local block through FiLM, plus coordinate
  channels built in km (cos-latitude aware); nothing assumes a conv kernel has a fixed physical size.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

L_REF_KM = (350.0, 350.0)   # reference extent of the spectral weights (~3.2 deg x 3.2 deg)
N_COORD = 5


def fourier_feats(v, n=8):
    """v: (B,) -> (B, 2n)."""
    freqs = 2.0 ** torch.arange(n, device=v.device, dtype=v.dtype) * 0.5
    a = v[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class Embed(nn.Module):
    """noise level + grid spacing at a given pyramid level -> FiLM source vector."""
    def __init__(self, d):
        super().__init__()
        self.t = nn.Sequential(nn.Linear(16, d), nn.SiLU(), nn.Linear(d, d))
        self.s = nn.Sequential(nn.Linear(32, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, c_noise, spacing, level):
        sp = spacing + level * math.log(2.0)  # spacing doubles with every 2x downsample
        s = torch.cat([fourier_feats(sp[:, 0]), fourier_feats(sp[:, 1])], 1)
        return self.t(fourier_feats(c_noise)) + self.s(s)


def _interp_matrix(n_out, m, scale, device):
    """(n_out, m): row j interpolates reference weights at index j*scale (scale = L_REF/L)."""
    pos = torch.arange(n_out, device=device, dtype=torch.float32) * scale
    pos = pos.clamp(max=m - 1)
    i0 = pos.floor().long().clamp(max=m - 2) if m > 1 else torch.zeros_like(pos).long()
    f = (pos - i0.float()) if m > 1 else torch.zeros_like(pos)
    P = torch.zeros(n_out, m, device=device)
    P[torch.arange(n_out), i0] += 1 - f
    if m > 1:
        P[torch.arange(n_out), i0 + 1] += f
    return P


class SpectralConv2d(nn.Module):
    def __init__(self, c, modes=12):
        super().__init__()
        self.c, self.m = c, modes
        s = 1.0 / (c * c)
        self.w1 = nn.Parameter(s * torch.randn(c, c, modes, modes, 2))
        self.w2 = nn.Parameter(s * torch.randn(c, c, modes, modes, 2))

    def _eff(self, w, Py, Px):
        wc = torch.view_as_complex(w.contiguous())
        # Py: (n1,m) Px: (n2,m)  -> (c,c,n1,n2)
        wr = torch.einsum('ja,ioab,kb->iojk', Py, wc.real, Px)
        wi = torch.einsum('ja,ioab,kb->iojk', Py, wc.imag, Px)
        return torch.complex(wr, wi)

    def forward(self, x, extent):
        B, C, H, W = x.shape
        Ly, Lx = float(extent[0]), float(extent[1])
        # highest integer frequency whose physical wavenumber is inside the reference band
        n1 = min(int((self.m - 1) * Ly / L_REF_KM[0]), (H - 1) // 2)
        n2 = min(int((self.m - 1) * Lx / L_REF_KM[1]), W // 2)
        xf = torch.fft.rfft2(x.float(), norm='ortho')
        out = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        dev = x.device
        Py = _interp_matrix(n1 + 1, self.m, L_REF_KM[0] / Ly, dev)
        Px = _interp_matrix(n2 + 1, self.m, L_REF_KM[1] / Lx, dev)
        w1 = self._eff(self.w1, Py, Px)
        out[:, :, :n1 + 1, :n2 + 1] = torch.einsum('bixy,ioxy->boxy', xf[:, :, :n1 + 1, :n2 + 1], w1)
        if n1 > 0:
            w2 = self._eff(self.w2, Py[1:], Px).flip(2)  # rows ordered by descending |j| to match H-n1..H-1
            out[:, :, H - n1:, :n2 + 1] = torch.einsum('bixy,ioxy->boxy', xf[:, :, H - n1:, :n2 + 1], w2)
        return torch.fft.irfft2(out, s=(H, W), norm='ortho').to(x.dtype)


class FiLMRes(nn.Module):
    """Local residual conv block, FiLM-modulated by [noise, spacing]."""
    def __init__(self, cin, cout, d):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin); self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout); self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(d, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.film(e)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + sc) + sh
        return self.skip(x) + self.c2(F.silu(h))


class FNOBlock(nn.Module):
    def __init__(self, c, d, modes):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.spec = SpectralConv2d(c, modes)
        self.pw = nn.Conv2d(c, c, 1)
        self.mlp = nn.Sequential(nn.Conv2d(c, 2 * c, 1), nn.GELU(), nn.Conv2d(2 * c, c, 1))
        self.film = nn.Linear(d, 2 * c)

    def forward(self, x, e, extent):
        h = self.norm(x)
        sc, sh = self.film(e)[:, :, None, None].chunk(2, 1)
        h = h * (1 + sc) + sh
        x = x + F.gelu(self.spec(h, extent) + self.pw(h))
        return x + self.mlp(x)


class UFNO(nn.Module):
    def __init__(self, width=48, mults=(1, 2, 2), modes=12, emb=128):
        super().__init__()
        self.mults = mults
        cs = [width * m for m in mults]
        self.emb = Embed(emb)
        cin = 4 + N_COORD  # x_t, y*M, M, E, coords
        self.stem = nn.Conv2d(cin, cs[0], 3, padding=1)
        self.enc_res, self.enc_fno, self.down = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i, c in enumerate(cs):
            self.enc_res.append(FiLMRes(c, c, emb)); self.enc_fno.append(FNOBlock(c, emb, modes))
            if i < len(cs) - 1:
                self.down.append(nn.Conv2d(c, cs[i + 1], 3, stride=2, padding=1))
        self.mid_res = FiLMRes(cs[-1], cs[-1], emb)
        self.mid_fno = FNOBlock(cs[-1], emb, modes)
        self.up, self.dec_res, self.dec_fno = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(len(cs) - 2, -1, -1):
            self.up.append(nn.Conv2d(cs[i + 1], cs[i], 3, padding=1))
            self.dec_res.append(FiLMRes(2 * cs[i], cs[i], emb)); self.dec_fno.append(FNOBlock(cs[i], emb, modes))
        self.out = nn.Sequential(nn.GroupNorm(8, cs[0]), nn.SiLU(), nn.Conv2d(cs[0], 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    @property
    def div(self):
        return 2 ** (len(self.mults) - 1)

    def forward(self, x, c_noise, cond, spacing, extent):
        """x (B,1,H,W) preconditioned noisy field; cond (B,3+N_COORD,H,W) = [y*M, M, E, coords].
        H, W must be divisible by self.div (use `pad_to_div`)."""
        h = self.stem(torch.cat([x, cond], 1))
        skips = []
        for i in range(len(self.mults)):
            e = self.emb(c_noise, spacing, i)
            h = self.enc_fno[i](self.enc_res[i](h, e), e, extent)
            skips.append(h)
            if i < len(self.mults) - 1:
                h = self.down[i](h)
        e = self.emb(c_noise, spacing, len(self.mults) - 1)
        h = self.mid_fno(self.mid_res(h, e), e, extent)
        for k, i in enumerate(range(len(self.mults) - 2, -1, -1)):
            h = self.up[k](F.interpolate(h, scale_factor=2, mode='nearest'))
            e = self.emb(c_noise, spacing, i)
            h = self.dec_fno[k](self.dec_res[k](torch.cat([h, skips[i]], 1), e), e, extent)
        return self.out(h)


def pad_to_div(t, div, value=0.0):
    H, W = t.shape[-2:]
    ph, pw = (-H) % div, (-W) % div
    return F.pad(t, (0, pw, 0, ph), value=value), (H, W)
