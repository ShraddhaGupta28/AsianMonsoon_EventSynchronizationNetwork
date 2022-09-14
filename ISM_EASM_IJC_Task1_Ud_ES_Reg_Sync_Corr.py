# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
import os
import xarray as xr
import numpy as np
import zarr as zr
from scipy.stats import ttest_rel, pearsonr, spearmanr
from scipy.signal import correlate, correlation_lags, filtfilt, cheby1, argrelmax, butter, savgol_filter
from scipy.ndimage.interpolation import shift
from itertools import product
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, rcParams
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
mpl.rc('text', usetex=True)

fnt = FontProperties(family='sans-serif',
                     size='medium',
                     style='normal',
                     weight='normal',
                     stretch='normal')


DATASETS = 'TRMM_Precipitation'
PERIOD = '1998_To_2019ASM.nc4'
SEASON = 'JJA'
PERCENTILE = 90.
TAUMAX = 7
SIGNIFICANCE = 95.  # directed significance level
SPAZOOMOUT = 1

MONTH = ''
REGION_LIST = [['CMZ', 'NCN'],
               ['ARBSEA', 'SCN1']]


def regional_box(p_reg):
    if p_reg == 'sNCN':
        lat0 = 36.
        lat1 = 42.
        lon0 = 108.
        lon1 = 118.
    if p_reg == 'sNISM':
        lat0 = 25.
        lat1 = 32.
        lon0 = 71.
        lon1 = 88.
    if p_reg == 'sEUR':
        lat0 = 44.
        lat1 = 50.
        lon0 = 3.
        lon1 = 17.
    if p_reg == 'sSCN':
        lat0 = 25.
        lat1 = 31.
        lon0 = 106.
        lon1 = 120.
    if p_reg == 'sSEEU':
        lat0 = 43.
        lat1 = 50.
        lon0 = 20.
        lon1 = 31.
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


for REGION in REGION_LIST:
    inp_evt = './Results/%s/%s_%s_%s.zarr' % (DATASETS, PERIOD.split(
        '.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1])

    inp_reg_sync = '../Extreme_Evol_ES/Results/%s/ES_%s_%s_%s_d%s_P%s_MS_SyncTs[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split(
        '.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], REGION[0], REGION[1])

    oup_reg_syncd = '../Extreme_Evol_ES/Results/%s/ES_%s_%s_%s_d%s_P%s_MS_SyncD[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(
        PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(REGION[0]), str(REGION[1]))

    with xr.open_zarr(inp_evt, consolidated=True) as dta:
        lat = dta['lat'].values.astype(np.float32)
        lon = dta['lon'].values.astype(np.float32)
        ttl = dta['ttl'].values.astype(np.uint16)
        tse = dta['tse'].values.astype(np.uint16)
        evt_srs = dta['evt_srs'].stack(nid=('lat', 'lon')).transpose(
            'nid', ...).values.astype(np.int32)
        la, lo = lat.shape[0], lon.shape[0]

    crd = np.array(list(product(lat, lon))).astype(np.float32)
    box1 = regional_box(REGION[0])
    box2 = regional_box(REGION[1])
    print(box1[0], box1[1], box1[2], box1[3])
    print(box2[0], box2[1], box2[2], box2[3])
    reg1 = np.where((crd[:, 0] >= box1[0]) &
                    (crd[:, 0] <= box1[1]) &
                    (crd[:, 1] >= box1[2]) &
                    (crd[:, 1] <= box1[3]))[0].astype(np.int32)
    reg2 = np.where((crd[:, 0] >= box2[0]) &
                    (crd[:, 0] <= box2[1]) &
                    (crd[:, 1] >= box2[2]) &
                    (crd[:, 1] <= box2[3]))[0].astype(np.int32)

    def cheby_lowpass_filter(x, order, rp, cutoff, fs):
        normal_cutoff = cutoff / (0.5 * fs)
        b, a = cheby1(order, rp, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, x)
        return y

    # https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7
    # from Github https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform

    def butter_lowpass_filter(data, cutoff_freq, fs, order):
        normal_cutoff = float(cutoff_freq / (0.5 * fs))
        b, a = butter(order, normal_cutoff, btype='low')
        y = filtfilt(b, a, data)
        return y

    st = datetime(int(PERIOD.split('_To_')[0]), 1, 1)
    d = np.array([st + timedelta(days=x)
                  for x in range(ttl.shape[0])], dtype=np.datetime64)
    dta_tim = xr.DataArray(ttl,
                           coords=[d],
                           dims=['time'])
    srs1 = evt_srs[reg1]
    srs2 = evt_srs[reg2]
    if MONTH != '':
        tim = dict(dta_tim.groupby('time.month'))[MONTH].values
    else:
        tim = dict(dta_tim.groupby('time.season'))[SEASON].values
    evt1 = np.zeros(ttl.shape[0], dtype=np.int32)
    evt2 = np.zeros(ttl.shape[0], dtype=np.int32)
    for t in tim:
        evt1[t] = np.where(srs1 == t)[0].shape[0]
        evt2[t] = np.where(srs2 == t)[0].shape[0]

    # output one ====================================================
    t121, t122, t211, t212 = np.load(inp_reg_sync).astype(np.int32)
    t12 = t121
    t21 = t211
    # 10 days cutoff;
    # 90 days for JJA, as the sample rate, 90 samples / s;
    # 22 years as 22 s in total, 90 * 22 as total samples;
    t12 = butter_lowpass_filter(t12, 10, 90, 8)
    t21 = butter_lowpass_filter(t21, 10, 90, 8)
    lm12 = argrelmax(t12)[0]
    lm21 = argrelmax(t21)[0]
    syncd12 = np.intersect1d(lm12,
                             np.where(t12 > np.percentile(t12[tse], 90))[0])
    syncd21 = np.intersect1d(lm21,
                             np.where(t21 > np.percentile(t21[tse], 90))[0])
    syncd12 = np.intersect1d(syncd12, tse)
    syncd21 = np.intersect1d(syncd21, tse)
    np.save(oup_reg_syncd, np.array(
        [syncd12, d[syncd12], syncd21, d[syncd21]]))
