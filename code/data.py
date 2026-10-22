"""GPU-resident samplers. Everything is native resolution; OSTIA is never interpolated to 0.02 deg.

Orientation: all fields are north-up (row 0 = northernmost). OSTIA is flipped on load (flip is an
index reversal, not resampling). Held-out days: ordinal%12==0 -> test, ==6 -> val, train excludes
test/val days and their +-1 neighbours.
"""
import glob, datetime as dt, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from geo import patch_geometry

import os
CACHE = os.environ.get('FNO_CACHE', '/data1/user/lz/SST_Diffusion/cache')
SPLIT_MOD = 12


def split_of(ordinals):
    o = np.asarray(ordinals)
    test, val = (o % SPLIT_MOD == 0), (o % SPLIT_MOD == 6)
    near = np.zeros_like(test)
    for d in (-1, 0, 1):
        near |= np.isin(o + d, o[test | val])
    return dict(test=np.where(test)[0], val=np.where(val)[0], train=np.where(~near)[0])


class CloudLib:
    """Real JAXA cloud-gap shapes (NaN & not land) from the library frames."""
    def __init__(self, dev):
        s = np.load(f'{CACHE}/jaxa/lib_sst.npy', mmap_mode='r')
        nanm = np.isnan(np.asarray(s))
        land = nanm.mean(0) > 0.99
        self.land = torch.from_numpy(land).to(dev)
        self.pat = torch.from_numpy(nanm & ~land).to(dev)  # (N,451,351)
        self.dev = dev

    def crops(self, B, px, out_px=None, frac=(0.1, 0.8), max_try=6):
        """B cloud patterns, each from a random px x px window (flips applied). If out_px != px the
        pattern is area-resampled to out_px (e.g. 160 -> 64 gives native-OSTIA-scale gaps)."""
        N, H, W = self.pat.shape
        out, got = [], 0
        for _ in range(max_try):
            n = B * 3
            f = torch.randint(0, N, (n,), device=self.dev)
            i0 = torch.randint(0, H - px + 1, (n,), device=self.dev); j0 = torch.randint(0, W - px + 1, (n,), device=self.dev)
            ar = torch.arange(px, device=self.dev)
            c = self.pat[f[:, None, None], (i0[:, None] + ar)[:, :, None], (j0[:, None] + ar)[:, None, :]]
            nl = ~self.land[(i0[:, None] + ar)[:, :, None], (j0[:, None] + ar)[:, None, :]]
            fr = c.float().sum((1, 2)) / nl.float().sum((1, 2)).clamp_min(1)
            keep = (fr >= frac[0]) & (fr <= frac[1])
            c = c[keep]
            flip_h = torch.rand(c.shape[0], device=self.dev) < 0.5; flip_v = torch.rand(c.shape[0], device=self.dev) < 0.5
            c = torch.where(flip_h[:, None, None], c.flip(2), c); c = torch.where(flip_v[:, None, None], c.flip(1), c)
            out.append(c); got += c.shape[0]
            if got >= B: break
        c = torch.cat(out)[:B]
        if c.shape[0] < B: c = c[torch.randint(0, c.shape[0], (B,), device=self.dev)]
        c = c[:, None]
        if out_px and out_px != px:
            c = F.interpolate(c.float(), size=(out_px, out_px), mode='area') > 0.5
        return c


class Ostia:
    def __init__(self, dev, years=None):
        fs = sorted(glob.glob(f'{CACHE}/ostia/*.npy'))
        if years: fs = [f for f in fs if years[0] <= int(f.split('/')[-1][:4]) <= years[1]]
        arrs, ords = [], []
        for f in fs:
            a = np.load(f); ym = f.split('/')[-1][:-4]
            d0 = dt.date(int(ym[:4]), int(ym[5:7]), 1)
            arrs.append(a); ords += [(d0 + dt.timedelta(i)).toordinal() for i in range(a.shape[0])]
        c = np.load(f'{CACHE}/ostia_coords.npz')
        self.x = torch.from_numpy(np.concatenate(arrs)[:, ::-1].copy()).to(dev)   # (T,500,600) north-up, kelvin, land NaN
        self.lat = torch.from_numpy(c['lat'][::-1].copy()).float().to(dev); self.lon = torch.from_numpy(c['lon']).float().to(dev)
        self.ord = np.array(ords); self.idx = split_of(self.ord); self.dev = dev

    def batch(self, B, split='train', px=64, min_ocean=0.7, gen=None):
        T, H, W = self.x.shape
        ids = torch.from_numpy(self.idx[split]).to(self.dev)
        n = B * 3
        d = ids[torch.randint(0, len(ids), (n,), device=self.dev)]
        i0 = torch.randint(0, H - px + 1, (n,), device=self.dev); j0 = torch.randint(0, W - px + 1, (n,), device=self.dev)
        ar = torch.arange(px, device=self.dev)
        rows, cols = i0[:, None] + ar, j0[:, None] + ar
        x = self.x[d[:, None, None], rows[:, :, None], cols[:, None, :]]
        ocean = ~torch.isnan(x)
        keep = ocean.float().mean((1, 2)) >= min_ocean
        k = keep.nonzero().flatten()[:B]
        if len(k) < B: k = torch.cat([k, k[torch.randint(0, len(k), (B - len(k),), device=self.dev)]])
        x, rows, cols = x[k][:, None], rows[k], cols[k]
        geo = patch_geometry(self.lat[rows], self.lon[cols])
        return x, ~torch.isnan(x), geo


class Jaxa:
    def __init__(self, dev, lib: CloudLib):
        self.x = torch.from_numpy(np.load(f'{CACHE}/jaxa/adapt_sst.npy')).to(dev)  # (N,451,351), true gaps NaN
        t = pd.read_csv(f'{CACHE}/jaxa/adapt_time.csv').time.astype('datetime64[ns]')
        self.times = pd.to_datetime(t).reset_index(drop=True)
        self.ord = np.array([x.toordinal() for x in self.times.dt.date])
        self.idx = split_of(self.ord)
        c = np.load(f'{CACHE}/jaxa_coords.npz')
        self.lat = torch.from_numpy(c['lat']).float().to(dev); self.lon = torch.from_numpy(c['lon']).float().to(dev)
        self.land, self.dev = lib.land, dev

    def batch(self, B, split='train', px=160, min_obs=0.5):
        N, H, W = self.x.shape
        ids = torch.from_numpy(self.idx[split]).to(self.dev)
        n = B * 3
        f = ids[torch.randint(0, len(ids), (n,), device=self.dev)]
        i0 = torch.randint(0, H - px + 1, (n,), device=self.dev); j0 = torch.randint(0, W - px + 1, (n,), device=self.dev)
        ar = torch.arange(px, device=self.dev)
        rows, cols = i0[:, None] + ar, j0[:, None] + ar
        x = self.x[f[:, None, None], rows[:, :, None], cols[:, None, :]]
        ocean = ~self.land[rows[:, :, None], cols[:, None, :]]
        obs = (~torch.isnan(x)).float().sum((1, 2)) / ocean.float().sum((1, 2)).clamp_min(1)
        k = ((obs >= min_obs) & (ocean.float().mean((1, 2)) >= 0.6)).nonzero().flatten()[:B]
        if len(k) < B: k = torch.cat([k, k[torch.randint(0, len(k), (B - len(k),), device=self.dev)]])
        x, rows, cols, ocean = x[k][:, None], rows[k], cols[k], ocean[k][:, None]
        geo = patch_geometry(self.lat[rows], self.lon[cols])
        return x, ocean, geo


def roles_complete(x, ocean, P):
    """Complete field (OSTIA): E=ocean, hidden = ocean & cloud pattern, visible = rest."""
    H_ = ocean & P; V = ocean & ~P
    return V, H_, ocean


def roles_masked(x, ocean, P, A=None):
    """Genuinely observed K = ocean & ~A (A = true/unknown gaps, JAXA: NaN pixels). E=K, hidden = K & P."""
    K = ocean if A is None else ocean & ~A
    return K & ~P, K & P, K
