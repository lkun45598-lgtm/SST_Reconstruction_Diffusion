"""Physical coordinates / grid spacing for lat-lon patches (north-up, row 0 = northernmost)."""
import math
import torch

R_EARTH = 6371.0088  # km
DX_REF = 3.0         # km, reference for log-spacing features


def patch_geometry(lat, lon):
    """lat: (B,h) degrees, lon: (B,w) degrees (both monotone along their axis).

    Returns dict:
      coords  (B,5,h,w): x_km/100, y_km/100 (centred on the patch), (lat-20)/10,
                         log(dx_local/DX_REF), log(dy/DX_REF)
      spacing (B,2): mean log dx, log dy  (km)       -> FiLM conditioning
      extent  (2,):  (Ly, Lx) physical extent in km (batch mean)
    Longitude spacing is converted to km with cos(latitude) evaluated per row, so the
    same angular spacing at different latitudes gives different physical distances.
    """
    B, h = lat.shape
    w = lon.shape[1]
    lat_r = torch.deg2rad(lat)
    dlat = (lat[:, 1:] - lat[:, :1]).abs()  # not used for spacing directly
    dy = R_EARTH * torch.deg2rad((lat[:, 1] - lat[:, 0]).abs())            # (B,)
    dlon = torch.deg2rad((lon[:, 1] - lon[:, 0]).abs())                    # (B,)
    cos_row = torch.cos(lat_r)                                              # (B,h)
    dx_row = R_EARTH * cos_row * dlon[:, None]                              # (B,h)
    y_km = R_EARTH * torch.deg2rad(lat - lat.mean(1, keepdim=True))        # (B,h)
    x_km = R_EARTH * cos_row[:, :, None] * torch.deg2rad(lon - lon.mean(1, keepdim=True))[:, None, :]
    ones = torch.ones(B, h, w, device=lat.device, dtype=lat.dtype)
    coords = torch.stack([
        x_km / 100.0,
        (y_km[:, :, None] * ones) / 100.0,
        ((lat[:, :, None] - 20.0) / 10.0) * ones,
        torch.log(dx_row / DX_REF)[:, :, None] * ones,
        torch.log(dy / DX_REF)[:, None, None] * ones,
    ], dim=1)
    spacing = torch.stack([torch.log(dx_row.mean(1) / DX_REF), torch.log(dy / DX_REF)], dim=1)
    extent = torch.stack([(dy * h).mean(), (dx_row.mean(1) * w).mean()])
    return dict(coords=coords, spacing=spacing, extent=extent)
