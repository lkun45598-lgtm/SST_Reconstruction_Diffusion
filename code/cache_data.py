"""Cache native-resolution OSTIA (no interpolation) and genuine JAXA L3 frames.

OSTIA: native 0.05 deg, box lat 5-30N / lon 100-130E, years given by --ostia-years (default 2010-2014,
       i.e. before the JAXA period, so no temporal overlap with JAXA test days). Kelvin, land = NaN.
JAXA : frames chosen by observed fraction from jaxa_processed_data diagnostics; values are the raw L3
       observations (NaN stay NaN, nothing is filled). Kelvin.
"""
import argparse, glob, os, sys, numpy as np, pandas as pd, xarray as xr
from multiprocessing import Pool

ROOT = '/data/sst_data/sst_missing_value_imputation'
OUT = '/data1/user/lz/SST_Diffusion/cache'
LAT = (5.0, 30.0); LON = (100.0, 130.0)

def ostia_month(ym):
    f = f'{ROOT}/copernicus_data/copernicus_data_monthly/sst_{ym}.nc'
    out = f'{OUT}/ostia/{ym}.npy'
    if os.path.exists(out) or not os.path.exists(f): return ym, 0
    d = xr.open_dataset(f)
    s = d.sel(latitude=slice(*LAT), longitude=slice(*LON))
    a = s.analysed_sst.values.astype('float32')
    np.save(out, a)
    if not os.path.exists(f'{OUT}/ostia_coords.npz'):
        np.savez(f'{OUT}/ostia_coords.npz', lat=s.latitude.values, lon=s.longitude.values)
    return ym, a.shape[0]

def jaxa_frame(ts):
    t = pd.Timestamp(ts)
    f = f'{ROOT}/jaxa_data/jaxa_extract_L3/{t:%Y%m}/{t:%d}/{t:%Y%m%d%H}0000.nc'
    try:
        return xr.open_dataset(f).sea_surface_temperature.values[0].astype('float32')
    except Exception as e:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ostia-years', nargs=2, type=int, default=[2010, 2014])
    p.add_argument('--n-adapt', type=int, default=4000)
    p.add_argument('--n-lib', type=int, default=1500)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--skip-jaxa', action='store_true')
    a = p.parse_args()
    with Pool(a.workers) as pool:
        yms = [f'{y}-{m:02d}' for y in range(a.ostia_years[0], a.ostia_years[1] + 1) for m in range(1, 13)]
        for ym, n in pool.imap_unordered(ostia_month, yms): print('ostia', ym, n, flush=True)
        if a.skip_jaxa: return
        # JAXA frame selection from diagnostics (raw_fraction = real observed fraction)
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{ROOT}/jaxa_processed_data/diagnostics/series_0?.csv'))])
        d['t'] = pd.to_datetime(d.timestamp); d = d[(d.t.dt.year >= 2018) & (d.t.dt.year <= 2023)]
        rng = np.random.default_rng(0)
        hi = d[d.raw_fraction >= 0.6]
        per_m = max(1, a.n_adapt // 12)
        sel = pd.concat([g.sample(min(len(g), per_m), random_state=0) for _, g in hi.groupby(hi.t.dt.month)])
        lib = d[(d.raw_fraction >= 0.1) & (d.raw_fraction <= 0.9)].sample(a.n_lib, random_state=1)
        for name, df in (('adapt', sel), ('lib', lib)):
            ts = [str(x) for x in df.t.sort_values()]
            fr = pool.map(jaxa_frame, ts)
            ok = [(t, f) for t, f in zip(ts, fr) if f is not None and f.shape == (451, 351)]
            np.save(f'{OUT}/jaxa/{name}_sst.npy', np.stack([f for _, f in ok]))
            pd.Series([t for t, _ in ok]).to_csv(f'{OUT}/jaxa/{name}_time.csv', index=False, header=['time'])
            print('jaxa', name, len(ok), 'of', len(ts), flush=True)
    d0 = xr.open_dataset(glob.glob(f'{ROOT}/jaxa_data/jaxa_extract_L3/201808/01/*.nc')[0])
    np.savez(f'{OUT}/jaxa_coords.npz', lat=d0.lat.values, lon=d0.lon.values)

if __name__ == '__main__': main()
