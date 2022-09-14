# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
import os
import zarr as zr
import xarray as xr
import numpy as np
import h5netcdf
from time import time, sleep


DATASETS = 'TRMM_Precipitation'  # TRMM_Precipitation, EAR5_Precipitation
PERIOD = '1998_To_2019.nc4'
SEASON = 'JJA'
PERCENTILE_LIST = [90.]  # from 80% to 99%
TAUMAX = 7  # the maximum temporal delay
SPAZOOMOUT = 1  # zoom out in spatial


def event_core(p_ts, p_pct):
    """
    P1: time series,
    P2: percentile to tell extreme events,
    return core
    """

    # Step 1: the 'wet' days' daily precipitation
    wet_dys = p_ts[p_ts > 1]
    if wet_dys.shape[0] != 0:
        # Step 2: the threshold of p_pct percentile
        thd = np.percentile(wet_dys, p_pct)

        # Step 3: considering consecutive days
        evt_tim = np.where(p_ts > thd)[0]
        csd_idx = np.where(np.diff(evt_tim) == 1)[0] + 1
        evb_tim = np.delete(evt_tim, csd_idx)
        evb_num = evb_tim.shape[0]

        if evb_num >= 3 and thd > 2:
            return evb_tim, evb_num, thd
        return np.zeros(1, dtype=np.uint16), 0, 0
    return np.zeros(1, dtype=np.uint16), 0, 0


for PERCENTILE in PERCENTILE_LIST:

    if 'TRMM' in DATASETS:
        inp_nc4 = './Datasets/%s/%s.nc4' % (DATASETS, PERIOD.split('.')[0])
        oup_evt = './Results/%s/%s_%s_%s.zarr' % (DATASETS, PERIOD.split(
            '.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1])
        if SPAZOOMOUT != 1:
            oup_evt_X = './Results/%s/%s_%s_%s_X%s.zarr' % (DATASETS, PERIOD.split(
                '.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(SPAZOOMOUT))

        if not os.path.exists(oup_evt):
            with xr.open_dataset(inp_nc4, engine='h5netcdf') as dta:
                t1 = time()
                ttl = dta['time'].values
                if SEASON != '':
                    ses = dict(dta.groupby('time.season'))[SEASON]
                    tse = np.nonzero((ses['time'].values)[:, None] == ttl)[
                        1].astype(np.uint16)
                else:
                    tse = np.arange(ttl.shape[0], dtype=np.uint16)

                lat = dta['lat'].values.astype(np.float32)
                lon = dta['lon'].values.astype(np.float32)
                pcp = dta['precipitation']
                la_dim, lo_dim, t_dim = lat.shape[0], lon.shape[0], ttl.shape[0]

                evt_max = np.uint16(tse.shape[0] * (1 - PERCENTILE / 100.)) + 1
                evt_srs = np.zeros((la_dim, lo_dim, evt_max), dtype=np.uint16)
                evt_num = np.zeros((la_dim, lo_dim), dtype=np.uint16)
                evt_thd = np.zeros((la_dim, lo_dim), dtype=np.float32)

                tmp_pcp = np.zeros((t_dim, lo_dim), dtype=np.float32)
                for y in range(la_dim):
                    print("hello ", y)
                    tmp_pcp[tse, :] = pcp[tse, :, y].fillna(0)
                    for x in range(lo_dim):
                        rlt = event_core(tmp_pcp[:, x], PERCENTILE)
                        evt_thd[y, x] = rlt[2]
                        evt_num[y, x] = rlt[1]
                        evt_srs[y, x, :rlt[1]] = rlt[0]
                    print(time() - t1)

                # output to zarr file
                evt_dta = xr.Dataset(
                    {
                        'evt_thd': (('lat', 'lon'), evt_thd[:, :]),
                        'evt_srs': (('lat', 'lon', 'evt'), evt_srs[:, :, :]),
                        'evt_num': (('lat', 'lon'), evt_num[:, :])
                    },
                    coords={
                        'lat': lat,
                        'lon': lon,
                        'evt': np.arange(evt_max, dtype=np.uint16),
                        'tse': tse,
                        'ttl': np.arange(t_dim, dtype=np.uint16)
                    })
                cps = zr.Blosc(cname='zstd', clevel=3, shuffle=2)
                evt_dta.to_zarr(oup_evt,
                                consolidated=True,
                                encoding={'evt_thd': {'compressor': cps},
                                          'evt_srs': {'compressor': cps},
                                          'evt_num': {'compressor': cps}, }, mode='w')

    if 'ERA5' in DATASETS:
        inp_nc = './Datasets/%s/%s.nc' % (DATASETS, PERIOD.split('.')[0])
        oup_evt = './Results/%s/%s_%s_%s.zarr' % (DATASETS, PERIOD.split(
            '.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1])
        if SPAZOOMOUT != 1:
            oup_evt_X = './Results/%s/%s_%s_%s_X%s.zarr' % (DATASETS, PERIOD.split(
                '.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(SPAZOOMOUT))

        if not os.path.exists(oup_evt):
            with xr.open_dataset(inp_nc) as dta:
                t1 = time()
                ttl = dta['time'].values
                if SEASON != '':
                    ses = dict(dta.groupby('time.season'))[SEASON]
                    tse = np.nonzero((ses['time'].values)[:, None] == ttl)[
                        1].astype(np.uint16)
                else:
                    tse = np.arange(ttl.shape[0], dtype=np.uint16)

                lat = dta['latitude'].values.astype(np.float32)
                lon = dta['longitude'].values.astype(np.float32)
                pcp = dta['tp']
                la_dim, lo_dim, t_dim = lat.shape[0], lon.shape[0], ttl.shape[0]

                evt_max = np.uint16(tse.shape[0] * (1 - PERCENTILE / 100.)) + 1
                evt_srs = np.zeros((la_dim, lo_dim, evt_max), dtype=np.uint16)
                evt_num = np.zeros((la_dim, lo_dim), dtype=np.uint16)
                evt_thd = np.zeros((la_dim, lo_dim), dtype=np.float32)

                tmp_pcp = np.zeros((t_dim, lo_dim), dtype=np.float32)
                for y in range(la_dim):
                    print("hello ", y)
                    tmp_pcp[tse, :] = pcp[tse, y, :].fillna(0) * 1000
                    for x in range(lo_dim):
                        rlt = event_core(tmp_pcp[:, x], PERCENTILE)
                        evt_thd[y, x] = rlt[2]
                        evt_num[y, x] = rlt[1]
                        evt_srs[y, x, :rlt[1]] = rlt[0]
                    print(time() - t1)

                # output to zarr file
                evt_dta = xr.Dataset(
                    {
                        'evt_thd': (('lat', 'lon'), evt_thd[:, :]),
                        'evt_srs': (('lat', 'lon', 'evt'), evt_srs[:, :, :]),
                        'evt_num': (('lat', 'lon'), evt_num[:, :])
                    },
                    coords={
                        'lat': lat,
                        'lon': lon,
                        'evt': np.arange(evt_max, dtype=np.uint16),
                        'tse': tse,
                        'ttl': np.arange(t_dim, dtype=np.uint16)
                    })
                cps = zr.Blosc(cname='zstd', clevel=3, shuffle=2)
                evt_dta.to_zarr(oup_evt,
                                consolidated=True,
                                encoding={'evt_thd': {'compressor': cps},
                                          'evt_srs': {'compressor': cps},
                                          'evt_num': {'compressor': cps}, }, mode='w')

    # to zoom out in spatial
    sleep(2)
    if SPAZOOMOUT != 1:
        with xr.open_zarr(oup_evt) as dta:
            # output to zarr file
            evt_dta = xr.Dataset(
                {
                    'evt_thd': (('lat', 'lon'), dta['evt_thd'].values.astype(np.float32)[::SPAZOOMOUT, ::SPAZOOMOUT]),
                    'evt_srs': (('lat', 'lon', 'evt'), dta['evt_srs'].values.astype(np.uint16)[::SPAZOOMOUT, ::SPAZOOMOUT, :]),
                    'evt_num': (('lat', 'lon'), dta['evt_num'].values.astype(np.uint16)[::SPAZOOMOUT, ::SPAZOOMOUT])
                },
                coords={
                    'lat': dta['lat'].values.astype(np.float32)[::SPAZOOMOUT],
                    'lon': dta['lon'].values.astype(np.float32)[::SPAZOOMOUT],
                    'evt': dta['evt'].values.astype(np.uint16),
                    'tse': dta['tse'].values.astype(np.uint16),
                    'ttl': dta['ttl'].values.astype(np.uint16)
                })
            cps = zr.Blosc(cname='zstd', clevel=3, shuffle=2)
            evt_dta.to_zarr(oup_evt_X,
                            consolidated=True,
                            encoding={'evt_thd': {'compressor': cps},
                                      'evt_srs': {'compressor': cps},
                                      'evt_num': {'compressor': cps}, }, mode='w')
