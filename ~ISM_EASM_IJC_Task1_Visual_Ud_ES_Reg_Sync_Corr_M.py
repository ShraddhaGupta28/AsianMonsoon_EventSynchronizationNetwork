# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
import os
import xarray as xr
import numpy as np
import pandas as pd
import zarr as zr
from scipy.stats import ttest_rel, pearsonr, spearmanr, kendalltau
from scipy.signal import correlate, correlation_lags, filtfilt, cheby1, argrelmax, butter, savgol_filter
from datetime import datetime, timedelta
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, rcParams
import matplotlib as mpl
from scipy.stats import kstest
# mpl.rc('text', usetex=True)

fnt = FontProperties(family='sans-serif',
                     size='large',
                     style='normal',
                     weight='normal',
                     stretch='normal')

DATASETS = 'TRMM_Precipitation'
PERIOD = '1998_To_2019ASM.nc4'
SEASON = 'JJA'
PERCENTILE = 90.
TAUMAX = 7
SIGNIFICANCE = 95.  # directed significance level

REGION_LIST = [['ARBSEA', 'SCN1'],
               ['CMZ', 'NCN']]
CUTOFF = 7
SOUTHERNMON = 6
NORTHERNMON = 7

oup_reg_syncd_fig12 = './Submission/IJC/Fig_&_Table/MS_SyncMPDF_Corr_ISM_EASM_d%s_P%s_12_CF%s.png' % (
    str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(CUTOFF))
oup_reg_syncd_fig21 = './Submission/IJC/Fig_&_Table/MS_SyncMPDF_Corr_ISM_EASM_d%s_P%s_21_CF%s.png' % (
    str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(CUTOFF))
oup_reg_syncd_figboth = './Submission/IJC/Fig_&_Table/MS_SyncMPDF_Corr_ISM_EASM_d%s_P%s_Both_CF%s.png' % (
    str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(CUTOFF))


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


with xr.open_zarr('./Results/TRMM_Precipitation/1998_To_2019ASM_JJA_900.zarr',
                  consolidated=True) as dta:
    ttl = dta['ttl'].values.astype(np.uint16)
st = datetime(int(PERIOD.split('_To_')[0]), 1, 1)
d = np.array([st + timedelta(days=x)
              for x in range(ttl.shape[0])], dtype=np.datetime64)
dta_tim = xr.DataArray(ttl,
                       coords=[d],
                       dims=['time'])
June = dict(dta_tim.groupby('time.month'))[6].values
July = dict(dta_tim.groupby('time.month'))[7].values
August = dict(dta_tim.groupby('time.month'))[8].values

syn12_mlist = []
syn21_mlist = []
synboth_mlist = []
for REGION in REGION_LIST:
    inp_reg_syncd = './Results/TRMM_Precipitation/ES_1998_To_2019ASM_JJA_900_d%s_P%s_MS_SyncD[%s-%s].npy' % (
        str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], REGION[0], REGION[1])
    syncd12, d12, syncd21, d21 = np.load(inp_reg_syncd, allow_pickle=True)
    both = np.union1d(syncd12, syncd21)
    syn12_6 = syncd12[np.in1d(syncd12, June)].shape[0]
    syn12_7 = syncd12[np.in1d(syncd12, July)].shape[0]
    syn12_8 = syncd12[np.in1d(syncd12, August)].shape[0]
    syn12_mlist.append(np.array([syn12_6, syn12_7, syn12_8]))
    syn21_6 = syncd21[np.in1d(syncd21, June)].shape[0]
    syn21_7 = syncd21[np.in1d(syncd21, July)].shape[0]
    syn21_8 = syncd21[np.in1d(syncd21, August)].shape[0]
    syn21_mlist.append(np.array([syn21_6, syn21_7, syn21_8]))
    synboth_6 = both[np.in1d(both, June)].shape[0]
    synboth_7 = both[np.in1d(both, July)].shape[0]
    synboth_8 = both[np.in1d(both, August)].shape[0]
    synboth_mlist.append(np.array([synboth_6, synboth_7, synboth_8]))

for i, SEASON in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
    inp_evt = './Results/TRMM_Precipitation/1998_To_2019_%s_900.zarr' % (
        SEASON)
    with xr.open_zarr(inp_evt, consolidated=True) as dta:
        lat = dta['lat'].values.astype(np.float32)
        lon = dta['lon'].values.astype(np.float32)
        ttl = dta['ttl'].values.astype(np.uint16)
        evt_srs = dta['evt_srs'].stack(nid=('lat', 'lon')).transpose(
            'nid', ...).values.astype(np.int32)
        evt_num = dta['evt_num'].stack(nid=('lat', 'lon')).transpose(
            'nid', ...).values.astype(np.int32)
        if i == 0:
            srs = evt_srs.copy()
            num = evt_num.copy()
        else:
            srs = np.concatenate((srs, evt_srs), axis=1)
            num = num + evt_num
        del evt_srs, evt_num
srs = np.sort(srs, axis=1)

oup_list = [oup_reg_syncd_fig12, oup_reg_syncd_fig21, oup_reg_syncd_figboth]
y_mlist = [syn12_mlist, syn21_mlist, synboth_mlist]
for oup, m_list in zip(oup_list, y_mlist):
    fig = plt.figure(figsize=(10, 3))
    gs = fig.add_gridspec(2, 7)
    ax_m = fig.add_subplot(gs[0:, 0:3])
    ax_s = fig.add_subplot(gs[0, 3:])
    ax_n = fig.add_subplot(gs[1, 3:])
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.7, wspace=2.)
    for ax in [ax_m, ax_s, ax_n]:
        ax.set_facecolor('#EAEAF2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_position(('outward', 6))
        ax.spines['bottom'].set_position(('outward', 6))
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_tick_params(labelleft=True)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(labelbottom=True)
    plt.text(0, 1.025, 'a',
             family=fnt.get_family(),
             fontsize='x-large',
             fontweight='bold',
             fontstretch=fnt.get_stretch(),
             transform=ax_m.transAxes)
    plt.text(0, 1.025, 'b',
             family=fnt.get_family(),
             fontsize='x-large',
             fontweight='bold',
             fontstretch=fnt.get_stretch(),
             transform=ax_s.transAxes)
    plt.text(0, 1.025, 'c',
             family=fnt.get_family(),
             fontsize='x-large',
             fontweight='bold',
             fontstretch=fnt.get_stretch(),
             transform=ax_n.transAxes)
    # for month-wise distribution ========================================
    labels = ['June', 'July', 'August']
    tik = np.arange(len(labels))
    width = 0.2
    x_list = [tik - width / 2, tik + width / 2]
    bar12_list = []
    for x, y, REGION in zip(x_list, m_list, REGION_LIST):
        region = REGION.copy()
        if region[1] == 'SCN1':
            region[1] = 'SCN'
        if region[0] == 'ARBSEA':
            region[0] = 'ARB'
        a = y / sum(y)
        if region[0] == 'ARB':
            bar = ax_m.bar(x, y / sum(y),
                           width,
                           label=r'Southern mode',
                           zorder=2)
            bar12_list.append(bar)
        if region[0] == 'CMZ':
            bar = ax_m.bar(x, y / sum(y),
                           width,
                           label=r'Northern mode',
                           zorder=2)
            bar12_list.append(bar)
        # if np.array_equal(np.array(m_list), np.array(syn12_mlist)):
        #     bar = ax_m.bar(x, y / sum(y),
        #                    width,
        #                    label=region[0] + r'$\rightarrow$' + region[1],
        #                    zorder=2)
        #     bar12_list.append(bar)
        # if np.array_equal(np.array(m_list), np.array(syn21_mlist)):
        #     bar = ax_m.bar(x, y / sum(y),
        #                    width,
        #                    label=region[1] + r'$\rightarrow$' + region[0],
        #                    zorder=2)
        #     bar12_list.append(bar)
        # if np.array_equal(np.array(m_list), np.array(synboth_mlist)):
        #     bar = ax_m.bar(x, y / sum(y),
        #                    width,
        #                    label=region[0] + r'$-$' + region[1],
        #                    zorder=2)
        #     bar12_list.append(bar)
    ax_m.set_ylim(0, 0.75)
    ax_m.set_yticks(np.arange(0, 0.751, 0.15))
    ax_m.set_xticks(tik)
    ax_m.set_xticklabels(labels)
    ax_m.set_ylabel(r'Relative frequency',
                    family=fnt.get_family(),
                    fontsize=fnt.get_size(),
                    fontweight=fnt.get_weight(),
                    fontstretch=fnt.get_stretch())
    ax_m.set_xlabel(r'Month',
                    family=fnt.get_family(),
                    fontsize=fnt.get_size(),
                    fontweight=fnt.get_weight(),
                    fontstretch=fnt.get_stretch())
    ax_m.grid(True, linestyle='-', linewidth=1, color='white',
              which='major', axis='both', zorder=1)
    for label in ax_m.get_yticklabels():
        label.set_fontproperties(fnt)
    for label in ax_m.get_xticklabels():
        label.set_fontproperties(fnt)
    mpl.rc('text', usetex=True)
    leg = []
    leg_labels = []
    for l in bar12_list:
        leg.append(l)
        leg_labels.append(l.get_label())
    lg = ax_m.legend(leg, leg_labels,
                     ncol=2, mode='expand',
                     fontsize=fnt.get_size(),
                     edgecolor='none',
                     bbox_to_anchor=(0.05, 0.98, 0.95, .102), loc='lower right',
                     borderaxespad=0.)
    lg.get_frame().set_alpha(None)
    lg.get_frame().set_facecolor((0, 0, 0, 0))
    mpl.rc('text', usetex=False)

    # for month-wise distribution ========================================
    ax_list = [ax_s, ax_n]
    box_list = [[regional_box('ARBSEA'), regional_box('SCN1')],
                [regional_box('CMZ'), regional_box('NCN')]]
    mon_list = [['JJA', [SOUTHERNMON]], ['JJA', [NORTHERNMON]]]
    for ax, box, mon in zip(ax_list, box_list, mon_list):
        crd = np.array(list(product(lat, lon))).astype(np.float32)
        reg1 = np.where((crd[:, 0] >= box[0][0]) &
                        (crd[:, 0] <= box[0][1]) &
                        (crd[:, 1] >= box[0][2]) &
                        (crd[:, 1] <= box[0][3]))[0].astype(np.int32)
        reg2 = np.where((crd[:, 0] >= box[1][0]) &
                        (crd[:, 0] <= box[1][1]) &
                        (crd[:, 1] >= box[1][2]) &
                        (crd[:, 1] <= box[1][3]))[0].astype(np.int32)

        def cheby_lowpass_filter(x, order, rp, cutoff, fs):
            normal_cutoff = cutoff / (0.5 * fs)
            b, a = cheby1(order, rp, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, x)
            return y

        def butter_lowpass_filter(data, cutoff_freq, fs, order):
            normal_cutoff = float(cutoff_freq / (0.5 * fs))
            b, a = butter(order, normal_cutoff, btype='low')
            y = filtfilt(b, a, data)
            return y

        srs1 = srs[reg1]
        srs2 = srs[reg2]
        num1 = num[reg1]
        num2 = num[reg2]
        srs1_t = np.zeros((srs1.shape[0], ttl.shape[0]), dtype=np.int32)
        srs2_t = np.zeros((srs2.shape[0], ttl.shape[0]), dtype=np.int32)
        for i in np.arange(srs1.shape[0]):
            idx = srs1[i][-num1[i]:]
            srs1_t[i, idx] = 1
        for i in np.arange(srs2.shape[0]):
            idx = srs2[i][-num2[i]:]
            srs2_t[i, idx] = 1
        evt1 = np.sum(srs1_t, axis=0)
        evt2 = np.sum(srs2_t, axis=0)
        print(kstest(evt1, 'norm'))
        print(kstest(evt2, 'norm'))
        # output two ====================================================
        # 10 days cutoff;
        # 90 days for JJA, as the sample rate, 90 samples / s;
        # 22 years as 22 s in total, 90 * 22 as total samples;
        evt1 = butter_lowpass_filter(evt1, CUTOFF, 90, 8)
        evt2 = butter_lowpass_filter(evt2, CUTOFF, 90, 8)
        print(kstest(evt1, 'norm'))
        print(kstest(evt2, 'norm'))
        st = datetime(int(PERIOD.split('_To_')[0]), 1, 1)
        d = np.array([st + timedelta(days=x)
                      for x in range(ttl.shape[0])], dtype=np.datetime64)
        dta_tim = xr.DataArray(ttl,
                               coords=[d],
                               dims=['time'])
        tim1 = np.array([], dtype=np.int32)
        tim2 = np.array([], dtype=np.int32)
        for idx in mon:
            if idx == 'JJA':
                t = dict(dta_tim.groupby('time.season'))[idx].values
                tim1 = np.concatenate((tim1, t))
            else:
                for m in idx:
                    t = dict(dta_tim.groupby('time.month'))[m].values
                    tim2 = np.concatenate((tim2, t))
        tim1 = np.sort(tim1)
        tim2 = np.sort(tim2)
        lag = np.arange(-40, 41)
        cor_p = np.zeros((lag.shape[0], 5))
        cor_p[:, 0] = lag
        # Spearman correlation ============================================
        for i, v in enumerate(lag):
            cor_p[i, 1], cor_p[i, 2] = spearmanr(evt1[tim1], evt2[tim1+v])
            cor_p[i, 3], cor_p[i, 4] = spearmanr(evt1[tim2], evt2[tim2+v])
        np.save('./Results/TRMM_Precipitation/Visual/ISM_EASM/Cor_PV_cf%s.npy' %
                str(CUTOFF), cor_p)
        idx_mc1 = argrelmax(cor_p[:, 1])[0]
        idx_mc1 = idx_mc1[cor_p[:, 2][idx_mc1] < 0.05]
        idx_mc1 = idx_mc1[cor_p[:, 1][idx_mc1] > 0.]
        idx1 = idx_mc1[np.argmin(np.abs(idx_mc1 - np.max(lag)))]
        mlag1 = lag[idx1]
        print(np.max(cor_p[:, 1]), np.min(cor_p[:, 1]))
        print(np.max(cor_p[:, 2]), np.min(cor_p[:, 2]))
        print(mlag1,
              np.round(cor_p[:, 1][idx1]),
              np.round(cor_p[:, 2][idx1]))
        print(idx_mc1)
        print('')
        idx_mc2 = argrelmax(cor_p[:, 3])[0]
        idx_mc2 = idx_mc2[cor_p[:, 4][idx_mc2] < 0.05]
        idx_mc2 = idx_mc2[cor_p[:, 3][idx_mc2] > 0.]
        idx2 = idx_mc2[np.argmin(np.abs(idx_mc2 - np.max(lag)))]
        mlag2 = lag[idx2]
        print(np.max(cor_p[:, 3]), np.min(cor_p[:, 3]))
        print(np.max(cor_p[:, 4]), np.min(cor_p[:, 4]))
        print(mlag2,
              np.round(cor_p[:, 3][idx2]),
              np.round(cor_p[:, 4][idx2]))
        print(idx_mc2)
        print('')

        l_list = []
        l = ax.plot(cor_p[:, 0][10: 71],
                    cor_p[:, 1][10: 71],
                    linestyle='-', linewidth=2.5,
                    label=r'JJA',
                    color='black')[0]
        l_list.append(l)
        if ax == ax_s:
            l = ax.plot(cor_p[:, 0][10: 71],
                        cor_p[:, 3][10: 71],
                        linestyle='-', linewidth=2.5,
                        label=r'June',
                        color='royalblue')[0]
            l_list.append(l)
            ax.set_ylim(-0.1, 0.2)
            ax.set_yticks(np.arange(-0.1, 0.21, 0.1))
        if ax == ax_n:
            l = ax.plot(cor_p[:, 0][10: 71],
                        cor_p[:, 3][10: 71],
                        linestyle='-', linewidth=2.5,
                        label=r'July',
                        color='royalblue')[0]
            l_list.append(l)
            ax.set_ylim(-0.2, 0.4)
            ax.set_yticks(np.arange(-0.2, 0.41, 0.2))
        ax.set_xlim(-30, 30)
        ax.set_xticks(np.arange(-30, 31, 10))
        ax.set_ylabel(r'Correlation',
                      family=fnt.get_family(),
                      fontsize=fnt.get_size(),
                      fontweight=fnt.get_weight(),
                      fontstretch=fnt.get_stretch())
        ax.tick_params(axis='y',
                       labelcolor='black',
                       color='black'
                       )
        if ax == ax_n:
            ax.set_xlabel(r'Lag (days)',
                          family=fnt.get_family(),
                          fontsize=fnt.get_size(),
                          fontweight=fnt.get_weight(),
                          fontstretch=fnt.get_stretch())
        ax.grid(True, linestyle='-', linewidth=1, color='white',
                which='major', axis='both')
        for label in ax.get_yticklabels():
            label.set_fontproperties(fnt)
        for label in ax.get_xticklabels():
            label.set_fontproperties(fnt)
        ax.axvline(x=mlag1,
                   linestyle='-', linewidth=2.5,
                   color='r')
        ax.axvline(x=mlag2,
                   linestyle='-', linewidth=2.5,
                   color='r')
        mpl.rc('text', usetex=True)
        leg = []
        leg_labels = []
        for l in l_list:
            leg.append(l)
            leg_labels.append(l.get_label())
        lg = ax.legend(leg, leg_labels,
                       ncol=2, mode='expand',
                       fontsize=fnt.get_size(),
                       edgecolor='none',
                       bbox_to_anchor=(0.6, 0.9, 0.4, .102), loc='lower right',
                       borderaxespad=0.)
        lg.get_frame().set_alpha(None)
        lg.get_frame().set_facecolor((0, 0, 0, 0))
        mpl.rc('text', usetex=False)

    if 'png' in oup:
        plt.savefig(oup, dpi=300, bbox_inches='tight')
    if 'tif' in oup:
        plt.savefig(oup, dpi=300, bbox_inches='tight')
    if 'pdf' in oup:
        plt.savefig(oup, dpi=300, bbox_inches='tight')
    plt.close()
