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
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, rcParams
import matplotlib as mpl
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
CUTOFF = 10
SOUTHERNMON = 6
NORTHERNMON = 7

oup_reg_syncd_fig12 = './Submission/IJC/Fig_&_Table/ES_%s_%s_%s_d%s_P%s_Syncd_BSISO_MJO_12A.png' % (PERIOD.split('.')[0], SEASON, str(
    PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])

oup_reg_syncd_fig21 = './Submission/IJC/Fig_&_Table/ES_%s_%s_%s_d%s_P%s_Syncd_BSISO_MJO_21A.png' % (PERIOD.split('.')[0], SEASON, str(
    PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])

oup_reg_syncd_figboth = './Submission/IJC/Fig_&_Table/ES_%s_%s_%s_d%s_P%s_Syncd_BSISO_MJO_BothA.png' % (PERIOD.split('.')[0], SEASON, str(
    PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])


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

abs1 = np.load('./Datasets/BSISO_1998_2019.npy')[:, -2]
abs2 = np.load('./Datasets/BSISO_1998_2019.npy')[:, -1]
amjo = np.load('./Datasets/MJO_Amplitude_1998_2019.npy')
bs1 = np.load('./Datasets/Bsiso1_1998_2019.npy')
bs2 = np.load('./Datasets/Bsiso2_1998_2019.npy')
mjo = np.load('./Datasets/MJO_1998_2019.npy')
abs1 = np.where(abs1 > 1)[0]
abs2 = np.where(abs2 > 1)[0]
amjo = np.where(amjo > 1)[0]

st = datetime(int(PERIOD.split('_To_')[0]), 1, 1)
d = np.array([st + timedelta(days=x)
              for x in range(ttl.shape[0])], dtype=np.datetime64)
dta_tim = xr.DataArray(ttl,
                       coords=[d],
                       dims=['time'])
jja = dict(dta_tim.groupby('time.season'))[SEASON].values
abs1 = np.intersect1d(abs1, jja)
abs2 = np.intersect1d(abs2, jja)
amjo = np.intersect1d(amjo, jja)

pctg12_list = []
pctg21_list = []
pctgboth_list = []
for REGION in REGION_LIST:
    inp_reg_syncd = './Results/%s/ES_%s_%s_%s_d%s_P%s_MS_SyncD[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(
        PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(REGION[0]), str(REGION[1]))

    syncd12, d12, syncd21, d21 = np.load(inp_reg_syncd, allow_pickle=True)
    syncd12 = syncd12.astype(np.int32)
    syncd21 = syncd21.astype(np.int32)
    both = np.union1d(syncd12, syncd21)
    syncd12_abs1 = np.intersect1d(syncd12, abs1)
    syncd12_abs2 = np.intersect1d(syncd12, abs2)
    syncd12_mjo = np.intersect1d(syncd12, amjo)
    cmb12 = reduce(np.union1d, (syncd12_abs1,
                                syncd12_abs2,
                                syncd12_mjo))
    pctg12_list.append([100 * syncd12_mjo.shape[0] / syncd12.shape[0],
                        100 * syncd12_abs1.shape[0] / syncd12.shape[0],
                        100 * syncd12_abs2.shape[0] / syncd12.shape[0],
                        ])
    syncd21_abs1 = np.intersect1d(syncd21, abs1)
    syncd21_abs2 = np.intersect1d(syncd21, abs2)
    syncd21_mjo = np.intersect1d(syncd21, amjo)
    cmb21 = reduce(np.union1d, (syncd21_abs1,
                                syncd21_abs2,
                                syncd21_mjo))
    pctg21_list.append([100 * syncd21_mjo.shape[0] / syncd21.shape[0],
                        100 * syncd21_abs1.shape[0] / syncd21.shape[0],
                        100 * syncd21_abs2.shape[0] / syncd21.shape[0],
                        ])
    both_abs1 = np.intersect1d(both, abs1)
    both_abs2 = np.intersect1d(both, abs2)
    both_mjo = np.intersect1d(both, amjo)
    cmbboth = reduce(np.union1d, (both_abs1,
                                  both_abs2,
                                  both_mjo))
    pctgboth_list.append([100 * both_mjo.shape[0] / both.shape[0],
                          100 * both_abs1.shape[0] / both.shape[0],
                          100 * both_abs2.shape[0] / both.shape[0],
                          ])

syn12_list = []
syn21_list = []
synboth_list = []
for osl in [mjo[amjo], bs1[abs1], bs2[abs2]]:
    unq, cnt = np.unique(osl, return_counts=True)
    idx = np.argsort(unq)
    unq = unq[idx].astype(np.int32)
    cnt = cnt[idx]
    frq = np.zeros(8)
    frq[unq - 1] = cnt
    syn12_list.append(frq)
    syn21_list.append(frq)
    synboth_list.append(frq)
for REGION in REGION_LIST:
    inp_reg_syncd = './Results/%s/ES_%s_%s_%s_d%s_P%s_MS_SyncD[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(
        PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(REGION[0]), str(REGION[1]))

    syncd12, d12, syncd21, d21 = np.load(inp_reg_syncd, allow_pickle=True)
    syncd12 = syncd12.astype(np.int32)
    syncd21 = syncd21.astype(np.int32)
    both = np.union1d(syncd12, syncd21)
    for osl in [mjo[np.intersect1d(syncd12, amjo)],
                bs1[np.intersect1d(syncd12, abs1)],
                bs2[np.intersect1d(syncd12, abs2)]]:
        unq, cnt = np.unique(osl, return_counts=True)
        idx = np.argsort(unq)
        unq = unq[idx].astype(np.int32)
        cnt = cnt[idx]
        frq = np.zeros(8)
        frq[unq - 1] = cnt
        syn12_list.append(frq)
    for osl in [mjo[np.intersect1d(syncd21, amjo)],
                bs1[np.intersect1d(syncd21, abs1)],
                bs2[np.intersect1d(syncd21, abs2)]]:
        unq, cnt = np.unique(osl, return_counts=True)
        idx = np.argsort(unq)
        unq = unq[idx].astype(np.int32)
        cnt = cnt[idx]
        frq = np.zeros(8)
        frq[unq - 1] = cnt
        syn21_list.append(frq)
    for osl in [mjo[np.intersect1d(both, amjo)],
                bs1[np.intersect1d(both, abs1)],
                bs2[np.intersect1d(both, abs2)]]:
        unq, cnt = np.unique(osl, return_counts=True)
        idx = np.argsort(unq)
        unq = unq[idx].astype(np.int32)
        cnt = cnt[idx]
        frq = np.zeros(8)
        frq[unq - 1] = cnt
        synboth_list.append(frq)


oup_list = [oup_reg_syncd_fig12, oup_reg_syncd_fig21, oup_reg_syncd_figboth]
y_mlist = [pctg12_list, pctg21_list, pctgboth_list]
for oup, m_list in zip(oup_list, y_mlist):
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(3, 8)
    ax_all = fig.add_subplot(gs[0:, 0:3])
    ax_mjo = fig.add_subplot(gs[0, 3:])
    ax_bs1 = fig.add_subplot(gs[1, 3:])
    ax_bs2 = fig.add_subplot(gs[2, 3:])
    # fig.tight_layout()
    plt.subplots_adjust(hspace=0.7, wspace=2.5)
    for ax in [ax_all, ax_mjo, ax_bs1, ax_bs2]:
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
             transform=ax_all.transAxes)
    plt.text(0, 1.025, 'b',
             family=fnt.get_family(),
             fontsize='x-large',
             fontweight='bold',
             fontstretch=fnt.get_stretch(),
             transform=ax_mjo.transAxes)
    plt.text(0, 1.025, 'c',
             family=fnt.get_family(),
             fontsize='x-large',
             fontweight='bold',
             fontstretch=fnt.get_stretch(),
             transform=ax_bs1.transAxes)
    plt.text(0, 1.025, 'd',
             family=fnt.get_family(),
             fontsize='x-large',
             fontweight='bold',
             fontstretch=fnt.get_stretch(),
             transform=ax_bs2.transAxes)
    # for month-wise distribution ========================================
    labels = ['MJO', 'BSISO1', 'BSISO2']
    tik = np.arange(len(labels))
    width = 0.15
    x_list = [tik - width / 2, tik + width / 2]
    bar12_list = []
    clr_jja = 'mediumorchid'
    clr_s = (0.12156862745098039, 0.4666666666666667,
             0.7058823529411765, 1.0)
    clr_n = (1.0, 0.4980392156862745, 0.054901960784313725, 1.0)
    for x, y, REGION in zip(x_list, m_list, REGION_LIST):
        region = REGION.copy()
        if region[1] == 'SCN1':
            region[1] = 'SCN'
        if region[0] == 'ARBSEA':
            region[0] = 'ARB'
        if region[0] == 'ARB':
            bar = ax_all.bar(x, y,
                             width,
                             label=r'Southern mode',
                             color=clr_s,
                             zorder=2)
            bar12_list.append(bar)
        if region[0] == 'CMZ':
            bar = ax_all.bar(x, y,
                             width,
                             label=r'Northern mode',
                             color=clr_n,
                             zorder=2)
            bar12_list.append(bar)
        ax_all.set_ylim(0, 100)
        ax_all.set_yticks(np.arange(0, 100.1, 25))
        ax_all.set_xticks(tik)
        ax_all.set_xticklabels(labels)
        ax_all.set_ylabel(r'Percentage (%) of Active days',
                          family=fnt.get_family(),
                          fontsize=fnt.get_size(),
                          fontweight=fnt.get_weight(),
                          fontstretch=fnt.get_stretch())
        ax_all.grid(True, linestyle='-', linewidth=1, color='white',
                    which='major', axis='both', zorder=1)
        for label in ax_all.get_yticklabels():
            label.set_fontproperties(fnt)
        for label in ax_all.get_xticklabels():
            label.set_fontproperties(fnt)
        mpl.rc('text', usetex=True)
        leg = []
        leg_labels = []
        for l in bar12_list:
            leg.append(l)
            leg_labels.append(l.get_label())
        lg = ax_all.legend(leg, leg_labels,
                           ncol=2, mode='expand',
                           fontsize='medium',
                           edgecolor='none',
                           bbox_to_anchor=(0.05, 1., 0.95, .102), loc='lower right',
                           borderaxespad=0.)
        lg.get_frame().set_alpha(None)
        lg.get_frame().set_facecolor((0, 0, 0, 0))
        mpl.rc('text', usetex=False)

    # for mode-wise distribution ========================================
    labels = [str(i) for i in range(1, 9)]
    tik = np.arange(len(labels))
    width = 0.2
    ax_list = [ax_mjo, ax_bs1, ax_bs2]
    x_list = [[tik - width, tik, tik + width],
              [tik - width, tik, tik + width],
              [tik - width, tik, tik + width]]
    y_list = [[syn12_list[0], syn12_list[3], syn12_list[6]],
              [syn12_list[1], syn12_list[4], syn12_list[7]],
              [syn12_list[2], syn12_list[5], syn12_list[8]]]
    region_list = [[['MJO', ''], REGION_LIST[0], REGION_LIST[1]],
                   [['BSISO1', ''], REGION_LIST[0], REGION_LIST[1]],
                   [['BSISO2', ''], REGION_LIST[0], REGION_LIST[1]]]
    for ax, X, Y, Region in zip(ax_list, x_list, y_list, region_list):
        bar12_list = []
        for x, y, region in zip(X, Y, Region):
            if region[1] == 'SCN1':
                region[1] = 'SCN'
            if region[0] == 'ARBSEA':
                region[0] = 'ARB'
            if region[0] == 'ARB':
                bar = ax.bar(x, y / sum(y),
                             width,
                             label=r'Southern mode',
                             color=clr_s,
                             zorder=2)
                bar12_list.append(bar)
            if region[0] == 'CMZ':
                bar = ax.bar(x, y / sum(y),
                             width,
                             label=r'Northern mode',
                             color=clr_n,
                             zorder=2)
                bar12_list.append(bar)
            if region[0] in ['MJO', 'BSISO1', 'BSISO2']:
                bar = ax.bar(x, y / sum(y),
                             width,
                             label=r'JJA',
                             color=clr_jja,
                             zorder=2)
                bar12_list.append(bar)
        ax.set_ylim(0, 0.45)
        ax.set_yticks(np.arange(0, 0.451, 0.15))
        ax.set_xticks(tik)
        ax.set_xticklabels(labels)
        if ax == ax_bs1:
            ax.set_ylabel(r'Relative frequency',
                          family=fnt.get_family(),
                          fontsize=fnt.get_size(),
                          fontweight=fnt.get_weight(),
                          fontstretch=fnt.get_stretch())
        if ax == ax_bs2:
            ax.set_xlabel(r'Phase',
                          family=fnt.get_family(),
                          fontsize=fnt.get_size(),
                          fontweight=fnt.get_weight(),
                          fontstretch=fnt.get_stretch())
        ax.grid(True, linestyle='-', linewidth=1, color='white',
                which='major', axis='both', zorder=1)
        for label in ax.get_yticklabels():
            label.set_fontproperties(fnt)
        for label in ax.get_xticklabels():
            label.set_fontproperties(fnt)
        if ax == ax_mjo:
            leg = []
            leg_labels = []
            for bar12 in bar12_list:
                leg.append(bar12)
                leg_labels.append(bar12.get_label())
            lg = ax.legend(leg, leg_labels,
                           ncol=3, mode='expand',
                           fontsize='medium',
                           edgecolor='none',
                           bbox_to_anchor=(0.05, 0.98, 0.90, .102), loc='lower right',
                           borderaxespad=0.)
            lg.get_frame().set_alpha(None)
            lg.get_frame().set_facecolor((0, 0, 0, 0))
        plt.text(.99, .95, Region[0][0] + ' (Active days)', ha='right', va='top',
                 family='Arial',
                 fontsize='medium',
                 fontweight=fnt.get_weight(),
                 fontstretch=fnt.get_stretch(),
                 transform=ax.transAxes,
                 bbox=dict(facecolor='white',
                           edgecolor='none',
                           pad=1.,
                           alpha=0.9))

    if 'png' in oup:
        plt.savefig(oup, dpi=300, bbox_inches='tight')
    if 'tif' in oup:
        plt.savefig(oup, dpi=300, bbox_inches='tight')
    if 'pdf' in oup:
        plt.savefig(oup, dpi=300, bbox_inches='tight')
    plt.close()
