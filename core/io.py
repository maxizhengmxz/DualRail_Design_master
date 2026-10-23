"""Dataset persistence helpers."""

from pathlib import Path

import xarray as xr


def save_dataset(ds: xr.Dataset, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


__all__ = ["save_dataset"]
