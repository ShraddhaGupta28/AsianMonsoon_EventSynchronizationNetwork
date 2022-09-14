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
SEASON_LIST = ['JJA']
PERCENTILE_LIST = [90.]  # from 80% to 99%
TAUMAX = 7  # the maximum temporal delay
SPAZOOMOUT = 1  # zoom out in spatial

# BOX = [-50, 50, -180, 180]
# REMINDER = 'AST'  # as TRMM
BOX = [0, 50, 60, 160]
REMINDER = 'ASM'  # to Asia summer monsoon


for SEASON in SEASON_LIST:
    for PERCENTILE in PERCENTILE_LIST:

        inp_evt = './Results/%s/%s_%s_%s.zarr' % (DATASETS, PERIOD.split(
            '.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1])
        if SPAZOOMOUT != 1:
            inp_evt = './Results/%s/%s_%s_%s_X%s.zarr' % (DATASETS, PERIOD.split(
                '.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(SPAZOOMOUT))

        oup_evt_box = './Results/%s/%s%s_%s_%s.zarr' % (DATASETS, PERIOD.split(
            '.')[0], REMINDER, SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1])
        if SPAZOOMOUT != 1:
            oup_evt_box = './Results/%s/%s%s_%s_%s_X%s.zarr' % (DATASETS, PERIOD.split(
                '.')[0], REMINDER, SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(SPAZOOMOUT))

        # inp_evt = './Results/%s/%s_%s_%s_2.zarr' % (DATASETS, PERIOD.split(
        #     '.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1])
        # if SPAZOOMOUT != 1:
        #     inp_evt = './Results/%s/%s_%s_%s_X%s_2.zarr' % (DATASETS, PERIOD.split(
        #         '.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(SPAZOOMOUT))

        # oup_evt_box = './Results/%s/%s%s_%s_%s_2.zarr' % (DATASETS, PERIOD.split(
        #     '.')[0], REMINDER, SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1])
        # if SPAZOOMOUT != 1:
        #     oup_evt_box = './Results/%s/%s%s_%s_%s_X%s_2.zarr' % (DATASETS, PERIOD.split(
        #         '.')[0], REMINDER, SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(SPAZOOMOUT))

        # to zoom out in spatial
        with xr.open_zarr(inp_evt) as dta:
            lat = dta['lat'].values.astype(np.float32)
            lon = dta['lon'].values.astype(np.float32)
            la = np.intersect1d(np.where(lat >= BOX[0])[0],
                                np.where(lat <= BOX[1])[0])
            lo = np.intersect1d(np.where(lon >= BOX[2])[0],
                                np.where(lon <= BOX[3])[0])
            evt_thd = dta['evt_thd'].values.astype(np.float32)[la][:, lo]
            # output to zarr file
            evt_dta = xr.Dataset(
                {
                    'evt_thd': (('lat', 'lon'), dta['evt_thd'].values.astype(np.float32)[la][:, lo]),
                    'evt_srs': (('lat', 'lon', 'evt'), dta['evt_srs'].values.astype(np.uint16)[la][:, lo, :]),
                    'evt_num': (('lat', 'lon'), dta['evt_num'].values.astype(np.uint16)[la][:, lo])
                },
                coords={
                    'lat': dta['lat'].values.astype(np.float32)[la],
                    'lon': dta['lon'].values.astype(np.float32)[lo],
                    'evt': dta['evt'].values.astype(np.uint16),
                    'tse': dta['tse'].values.astype(np.uint16),
                    'ttl': dta['ttl'].values.astype(np.uint16)
                })
            cps = zr.Blosc(cname='zstd', clevel=3, shuffle=2)
            evt_dta.to_zarr(oup_evt_box,
                            consolidated=True,
                            encoding={'evt_thd': {'compressor': cps},
                                      'evt_srs': {'compressor': cps},
                                      'evt_num': {'compressor': cps}, }, mode='w')
