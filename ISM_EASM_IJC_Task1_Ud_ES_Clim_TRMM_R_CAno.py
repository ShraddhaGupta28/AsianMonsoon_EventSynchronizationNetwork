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
REGION_LIST = [['EUR', 'NISM'],
               ['NISM', 'NCN'],
               ['NCSISM', 'PHSEA'],
               ['PHSEA', 'JSEA'],
               ['SEEU', 'SCN'],
               ['NEPF', 'EJP'],
               ['SISM', 'SCN'],
               ['ARBSEA', 'SCN']]
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


inp_clim_r = '../Extreme_Evol_ES/Results/%s/Clim_TRMM_R_1X1_1998_To_2019.nc4' % DATASETS
with xr.open_dataset(inp_clim_r) as dta:
    lat = dta['lat'].values.astype(np.float32)
    lon = dta['lon'].values.astype(np.float32)
    ttl = dta['time'].values
    pcp = dta['precipitation'].fillna(0)
    # pcps = dict(dta.groupby('time.season'))[SEASON].mean('time')['precipitation']
    # print(pcps[100, 100])

pcps = np.mean(dict(pcp.groupby('time.season'))[SEASON].values, axis=0)

for REGION in REGION_LIST:
    inp_reg_syncd = '../Extreme_Evol_ES/Results/%s/ES_%s_%s_%s_d%s_P%s_MS_SyncD[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(
        PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(REGION[0]), str(REGION[1]))
    if SPAZOOMOUT != 1:
        inp_reg_syncd = '../Extreme_Evol_ES/Results/%s/ES_%s_%s_%s_d%s_P%s_X%s_MS_SyncD[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(
            PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(SPAZOOMOUT), str(REGION[0]), str(REGION[1]))

    oup_ano = '../Extreme_Evol_ES/Results/%s/ES_%s_%s_%s_d%s_P%s_MSSD_TRMMRCAno[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(
        PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(REGION[0]), str(REGION[1]))
    if SPAZOOMOUT != 1:
        oup_ano = '../Extreme_Evol_ES/Results/%s/ES_%s_%s_%s_d%s_P%s_X%s_MSSD_TRMMRCAno[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(
            PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(SPAZOOMOUT), str(REGION[0]), str(REGION[1]))

    syncd12, d12, syncd21, d21 = np.load(inp_reg_syncd, allow_pickle=True)
    lags = np.arange(-10, 11, 1)
    sync_ano = np.zeros((3, lags.shape[0], lat.shape[0], lon.shape[0]))
    both = np.union1d(syncd12, syncd21)
    for i, l in enumerate(lags):
        print("Lag: ", l)
        ano = np.mean(
            pcp.values[(syncd12 + l).astype(np.int32)], axis=0) - pcps
        sync_ano[0, i] = ano.T
        ano = np.mean(
            pcp.values[(syncd21 + l).astype(np.int32)], axis=0) - pcps
        sync_ano[1, i] = ano.T
        ano = np.mean(
            pcp.values[(both + l).astype(np.int32)], axis=0) - pcps
        sync_ano[2, i] = ano.T

    np.save(oup_ano, sync_ano.astype(np.float32))

    # for i, t in enumerate(ttl):
    #     unix_epoch = np.datetime64(0, 's')
    #     one_second = np.timedelta64(1, 's')
    #     seconds_since_epoch = (t - unix_epoch) / one_second
    #     d = datetime.utcfromtimestamp(seconds_since_epoch)
    #     d = d.replace(hour=0, minute=0, second=0)
    #     ttl[i] = d
    # st = np.array([datetime(int(PERIOD.split('_To_')[0]), 1, 1)],
    #               dtype=np.datetime64)
    # i = np.where(ttl == st[0])[0][0]
    # ttl = ttl[i:]
    # v = v[i:, :, :]
    # v_ano = v - np.mean(v[:, :, :], axis=0)
