# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

DATASETS = 'TRMM_Precipitation'
PERIOD = '1998_To_2019ASM.nc4'
SEASON = 'JJA'
PERCENTILE = 90.
TAUMAX = 7
SIGNIFICANCE = 95.  # directed significance level
SPAZOOMOUT = 1
REGION_LIST = [['CMZ', 'NCN'],
               ['ARBSEA', 'SCN1']]


def regional_box(p_reg):
    if p_reg == 'CISM':
        lat0 = 21.
        lat1 = 28.
        lon0 = 71.
        lon1 = 88.
    if p_reg == 'NCN':
        lat0 = 36.
        lat1 = 42.
        lon0 = 108.
        lon1 = 118.
    if p_reg == 'NISM':
        lat0 = 25.
        lat1 = 32.
        lon0 = 71.
        lon1 = 88.
    if p_reg == 'CMZ':
        lat0 = 20.
        lat1 = 32.
        lon0 = 71.
        lon1 = 88.
    if p_reg == 'SISM':
        lat0 = 0.
        lat1 = 15.
        lon0 = 70.
        lon1 = 82.
    if p_reg == 'ARBSEA':
        lat0 = 5.
        lat1 = 20.
        lon0 = 60.
        lon1 = 75.
    if p_reg == 'EUR':
        lat0 = 42.
        lat1 = 50.
        lon0 = 3.
        lon1 = 15.
    if p_reg == 'SCN':
        lat0 = 25.5
        lat1 = 31.5
        lon0 = 113.
        lon1 = 130.
    if p_reg == 'SCN1':
        lat0 = 23.
        lat1 = 29.
        lon0 = 105.
        lon1 = 115.
    if p_reg == 'SCN2':
        lat0 = 27.
        lat1 = 33.
        lon0 = 112.
        lon1 = 122.
    if p_reg == 'JSEA':
        lat0 = 37.5
        lat1 = 41.5
        lon0 = 128.
        lon1 = 141.
    if p_reg == 'PHSEA':
        lat0 = 15.5
        lat1 = 25.5
        lon0 = 120.
        lon1 = 135.
    if p_reg == 'NCSISM':
        lat0 = 15.
        lat1 = 28.
        lon0 = 71.
        lon1 = 88.
    if p_reg == 'EJP':
        lat0 = 38.
        lat1 = 46.
        lon0 = 138.
        lon1 = 152.
    if p_reg == 'NEPF':
        lat0 = 10.
        lat1 = 16.
        lon0 = -170.
        lon1 = -155.
    if p_reg == 'SEEU':
        lat0 = 41.
        lat1 = 50.
        lon0 = 20.
        lon1 = 36.
    return np.array([lat0, lat1, lon0, lon1], dtype=np.float64)


inp_clim_wvfvc = '../Extreme_Evol_ES/Results/%s/Clim_ERA5_WVFVC_1X1_1998_To_2019.nc' % (
    DATASETS)
inp_clim_wvfuc = '../Extreme_Evol_ES/Results/%s/Clim_ERA5_WVFUC_1X1_1998_To_2019.nc' % (
    DATASETS)
with xr.open_dataset(inp_clim_wvfvc) as dta:
    lat = dta['latitude'].values.astype(np.float32)
    lon = dta['longitude'].values.astype(np.float32)
    ttl = dta['time'].values
    v = dta['p72.162']

with xr.open_dataset(inp_clim_wvfuc) as dta:
    u = dta['p71.162']

vt = dict(v.groupby('time.season'))[SEASON].values
ut = dict(u.groupby('time.season'))[SEASON].values
# wind speed composite
wvfcps = np.mean(np.sqrt(vt ** 2 + ut ** 2), axis=0)
# wind direction composite
vcps = np.mean(vt, axis=0)
ucps = np.mean(ut, axis=0)

for REGION in REGION_LIST:
    inp_reg_syncd = '../Extreme_Evol_ES/Results/%s/ES_%s_%s_%s_d%s_P%s_MS_SyncD[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(
        PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(REGION[0]), str(REGION[1]))
    if SPAZOOMOUT != 1:
        inp_reg_syncd = '../Extreme_Evol_ES/Results/%s/ES_%s_%s_%s_d%s_P%s_X%s_MS_SyncD[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(
            PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(SPAZOOMOUT), str(REGION[0]), str(REGION[1]))

    oup_ano = '../Extreme_Evol_ES/Results/%s/ES_%s_%s_%s_d%s_P%s_MSSD_ERA5WVFCAno[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(
        PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(REGION[0]), str(REGION[1]))
    if SPAZOOMOUT != 1:
        oup_ano = '../Extreme_Evol_ES/Results/%s/ES_%s_%s_%s_d%s_P%s_X%s_MSSD_ERA5WVFCAno[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(
            PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(SPAZOOMOUT), str(REGION[0]), str(REGION[1]))

    syncd12, d12, syncd21, d21 = np.load(inp_reg_syncd, allow_pickle=True)
    lags = np.arange(-30, 31, 1)
    sync_ano = np.zeros((9, lags.shape[0], lat.shape[0], lon.shape[0]))
    both = np.union1d(syncd12, syncd21)
    for i, l in enumerate(lags):
        print("Lag: ", l)
        vt = v.values[(syncd12 + l).astype(np.int32)]
        ut = u.values[(syncd12 + l).astype(np.int32)]
        ws_ano = np.mean(np.sqrt(vt ** 2 + ut ** 2), axis=0) - wvfcps
        v_ano = np.mean(vt, axis=0) - vcps
        u_ano = np.mean(ut, axis=0) - ucps
        sync_ano[0, i] = ws_ano
        sync_ano[1, i] = v_ano
        sync_ano[2, i] = u_ano
        vt = v.values[(syncd21 + l).astype(np.int32)]
        ut = u.values[(syncd21 + l).astype(np.int32)]
        ws_ano = np.mean(np.sqrt(vt ** 2 + ut ** 2), axis=0) - wvfcps
        v_ano = np.mean(vt, axis=0) - vcps
        u_ano = np.mean(ut, axis=0) - ucps
        sync_ano[3, i] = ws_ano
        sync_ano[4, i] = v_ano
        sync_ano[5, i] = u_ano
        vt = v.values[(both + l).astype(np.int32)]
        ut = u.values[(both + l).astype(np.int32)]
        ws_ano = np.mean(np.sqrt(vt ** 2 + ut ** 2), axis=0) - wvfcps
        v_ano = np.mean(vt, axis=0) - vcps
        u_ano = np.mean(ut, axis=0) - ucps
        sync_ano[6, i] = ws_ano
        sync_ano[7, i] = v_ano
        sync_ano[8, i] = u_ano

    np.save(oup_ano, sync_ano.astype(np.float32))
