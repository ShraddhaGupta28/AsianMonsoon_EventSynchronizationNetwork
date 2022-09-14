# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.optimize as so
import cmocean as oc
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties, rcParams
from matplotlib.animation import FuncAnimation
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from itertools import product
import matplotlib as mpl
# mpl.rc('text', usetex=True)

fnt = FontProperties(family='sans-serif',
                     size='large',
                     style='normal',
                     weight='normal',
                     stretch='normal')


DATASETS = 'TRMM_Precipitation'
PERIOD = '1998_To_2019ASM.nc4'
SEASON = ''
PERCENTILE = 90.
TAUMAX = 7
SIGNIFICANCE = 95.  # directed significance level
SPAZOOMOUT = 1
DAY = 0
ITV = 6


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


sz250 = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5250ZCAno[ARBSEA-SCN1]_Mon.npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
sz500 = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5500ZCAno[ARBSEA-SCN1]_Mon.npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
sz850 = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5850ZCAno[ARBSEA-SCN1]_Mon.npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
swv250 = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5250WVCAno[ARBSEA-SCN1]_Mon.npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
swv500 = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5500WVCAno[ARBSEA-SCN1]_Mon.npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
swv850 = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5850WVCAno[ARBSEA-SCN1]_Mon.npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
nz250 = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5250ZCAno[CMZ-NCN]_Mon.npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
nz500 = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5500ZCAno[CMZ-NCN]_Mon.npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
nz850 = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5850ZCAno[CMZ-NCN]_Mon.npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
nwv250 = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5250WVCAno[CMZ-NCN]_Mon.npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
nwv500 = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5500WVCAno[CMZ-NCN]_Mon.npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
nwv850 = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5850WVCAno[CMZ-NCN]_Mon.npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
with xr.open_dataset('C:/~Dataset/Clim_ERA5_GPH_250hpa_1X1_1998_To_2019.nc') as dta:
    lat = dta['latitude'].values.astype(np.float32)
    lon = dta['longitude'].values.astype(np.float32)

lat0, lat1, lon0, lon1 = -15, 51, 0, 160
latlabels = np.arange(int(lat0 / 10) * 10, int(lat1 / 10) * 10 + 1, 20)
lonlabels = np.arange(int(lon0 / 10) * 10, int(lon1 / 10) * 10 + 1, 40)
bm_lac, bm_loc = (lat0 + lat1) / 2, (lon0 + lon1) / 2
shift = bm_loc

lags = np.arange(-10, 11, 1)
oup_ano12 = './Submission/IJC/Fig_&_Table/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5GPHSN12_Day%s_MonRF.png' % (
    PERIOD.split(
        '.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(DAY))
oup_ano21 = './Submission/IJC/Fig_&_Table/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5GPHSN21_Day%s_MonRF.png' % (
    PERIOD.split(
        '.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(DAY))
oup_anoboth = './Submission/IJC/Fig_&_Table/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5GPHSNboth_Day%s_MonRF.png' % (
    PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(DAY))

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8.5, 7.5))
fig.tight_layout()
plt.subplots_adjust(hspace=.2, wspace=.2)
ax1, ax2 = axs[0][0], axs[0][1]
ax3, ax4 = axs[1][0], axs[1][1]
ax5, ax6 = axs[2][0], axs[2][1]
plt.text(0., 1.025, 'a',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax1.transAxes)
plt.text(0., 1.025, 'b',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax2.transAxes)
plt.text(0.3, 1.025, 'Southern mode',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight=fnt.get_weight(),
         fontstretch=fnt.get_stretch(),
         transform=ax1.transAxes)
plt.text(0.3, 1.025, 'Northern mode',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight=fnt.get_weight(),
         fontstretch=fnt.get_stretch(),
         transform=ax2.transAxes)
plt.text(0., 1.025, 'c',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax3.transAxes)
plt.text(0., 1.025, 'd',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax4.transAxes)
plt.text(0., 1.025, 'e',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax5.transAxes)
plt.text(0., 1.025, 'f',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax6.transAxes)
for i, l in enumerate(lags):
    ax_list = []
    if l == DAY:
        ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    if ax_list != []:
        z_list = [sz250[0, i], nz250[0, i],
                  sz500[0, i], nz500[1, i],
                  sz850[0, i], nz850[0, i]]
        v_list = [swv250[1, i], nwv250[1, i],
                  swv500[1, i], nwv500[1, i],
                  swv850[1, i], nwv850[1, i]]
        u_list = [swv250[2, i], nwv250[2, i],
                  swv500[2, i], nwv500[2, i],
                  swv850[2, i], nwv850[2, i]]
        box_list = [[regional_box('ARBSEA'), regional_box('SCN1')],
                    [regional_box('CMZ'), regional_box('NCN')],
                    [regional_box('ARBSEA'), regional_box('SCN1')],
                    [regional_box('CMZ'), regional_box('NCN')],
                    [regional_box('ARBSEA'), regional_box('SCN1')],
                    [regional_box('CMZ'), regional_box('NCN')]]
        pre_list = ['250', '250',
                    '500', '500',
                    '850', '850']
        for ax, z, v, u, box, pre in zip(ax_list, z_list, v_list, u_list, box_list, pre_list):
            bm = Basemap(ax=ax,
                         projection='mill',
                         llcrnrlon=lon0, llcrnrlat=lat0,
                         urcrnrlon=lon1, urcrnrlat=lat1,
                         lat_0=bm_lac, lon_0=shift)
            bm.drawparallels(latlabels,
                             labels=[1, 0, 0, 0],
                             family=fnt.get_family(),
                             fontsize=fnt.get_size(),
                             fontweight=fnt.get_weight(),
                             fontstretch=fnt.get_stretch(),
                             rotation='0',
                             dashes=[1, 1],
                             linewidth=.5,
                             color='darkgrey')
            bm.drawmeridians(lonlabels,
                             labels=[0, 0, 0, 1],
                             family=fnt.get_family(),
                             fontsize=fnt.get_size(),
                             fontweight=fnt.get_weight(),
                             fontstretch=fnt.get_stretch(),
                             rotation='0',
                             dashes=[1, 1],
                             linewidth=.3,
                             color='darkgrey')
            bm.drawcoastlines(linewidth=1., color='darkslategray')
            bm.drawcountries(linewidth=0.5, color='darkslategray')
            bm.drawmapboundary(linewidth=0., fill_color='aliceblue')

            lo, z = bm.shiftdata(lon, z, lon_0=shift)
            lo, la = np.meshgrid(lo, lat)
            lo, la = bm(lo, la)
            if pre == '250':
                cf_bm = bm.contourf(lo, la, z,
                                    levels=np.arange(-9, 10, 1.5),
                                    cmap=plt.cm.coolwarm,
                                    alpha=1.,
                                    extend='both',
                                    zorder=1)
                cb_bm = bm.colorbar(cf_bm, location='bottom',
                                    ticks=np.arange(-9, 10, 3),
                                    pad=.2, size=.1)
            if pre == '500':
                cf_bm = bm.contourf(lo, la, z,
                                    levels=np.arange(-9, 10, 1.5),
                                    cmap=plt.cm.coolwarm,
                                    alpha=1.,
                                    extend='both',
                                    zorder=1)
                cb_bm = bm.colorbar(cf_bm, location='bottom',
                                    ticks=np.arange(-9, 10, 3),
                                    pad=.2, size=.1)
            if pre == '850':
                cf_bm = bm.contourf(lo, la, z,
                                    levels=np.arange(-9, 10, 1.5),
                                    cmap=plt.cm.coolwarm,
                                    alpha=1.,
                                    extend='both',
                                    zorder=1)
                cb_bm = bm.colorbar(cf_bm, location='bottom',
                                    ticks=np.arange(-9, 10, 3),
                                    pad=.2, size=.1)
            cb_bm.ax.set_xlabel(r'GPH (%shPa) (m)' % pre,
                                family=fnt.get_family(),
                                fontsize=fnt.get_size(),
                                fontweight=fnt.get_weight(),
                                fontstretch=fnt.get_stretch())
            cb_bm.ax.tick_params(direction='in', color='black')
            cb_bm.outline.set_visible(False)
            for label in cb_bm.ax.get_xticklabels():
                label.set_fontproperties(fnt)

            lo, v = bm.shiftdata(lon, v, lon_0=shift)
            lo, u = bm.shiftdata(lon, u, lon_0=shift)
            lo, la = np.meshgrid(lo, lat)
            lo, la = bm(lo, la)
            if pre == '250':
                bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                          u[::ITV, ::ITV], v[::ITV, ::ITV],
                          scale=65, color='black',
                          # pivot='mid', units='width',
                          headwidth=7., headlength=7.,
                          zorder=2)
            if pre == '500':
                bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                          u[::ITV, ::ITV], v[::ITV, ::ITV],
                          scale=40, color='black',
                          # pivot='mid', units='width',
                          headwidth=7., headlength=7.,
                          zorder=2)
            if pre == '850':
                bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                          u[::ITV, ::ITV], v[::ITV, ::ITV],
                          scale=40, color='black',
                          # pivot='mid', units='width',
                          headwidth=7., headlength=7.,
                          zorder=2)
            patches = []
            reg1 = np.array([[box[0][2], box[0][0]],
                            [box[0][2], box[0][1]],
                            [box[0][3], box[0][1]],
                            [box[0][3], box[0][0]]])
            reg2 = np.array([[box[1][2], box[1][0]],
                            [box[1][2], box[1][1]],
                            [box[1][3], box[1][1]],
                            [box[1][3], box[1][0]]])
            x, y = bm(reg1[:, 0], reg1[:, 1])
            xy = np.vstack([x, y]).T
            poly = Polygon(xy,
                           facecolor='None',
                           edgecolor='gold',
                           linewidth=3,
                           zorder=2)
            plt.gca().add_patch(poly)
            x, y = bm(reg2[:, 0], reg2[:, 1])
            xy = np.vstack([x, y]).T
            poly = Polygon(xy,
                           facecolor='None',
                           edgecolor='gold',
                           linewidth=3,
                           zorder=2)
            plt.gca().add_patch(poly)
            plt.text(.12, .95, SEASON, ha='right', va='top',
                     family='Arial',
                     fontsize='x-large',
                     fontweight=fnt.get_weight(),
                     fontstretch=fnt.get_stretch(),
                     transform=ax.transAxes,
                     bbox=dict(facecolor='white',
                               edgecolor='none',
                               pad=1.,
                               alpha=0.9))
if 'png' in oup_ano12:
    plt.savefig(oup_ano12, dpi=300, bbox_inches='tight')
if 'tif' in oup_ano12:
    plt.savefig(oup_ano12, dpi=300, bbox_inches='tight')
if 'pdf' in oup_ano12:
    plt.savefig(oup_ano12, dpi=300, bbox_inches='tight')
plt.close()

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8.5, 7.5))
fig.tight_layout()
plt.subplots_adjust(hspace=.2, wspace=.2)
ax1, ax2 = axs[0][0], axs[0][1]
ax3, ax4 = axs[1][0], axs[1][1]
ax5, ax6 = axs[2][0], axs[2][1]
plt.text(0., 1.025, 'a',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax1.transAxes)
plt.text(0., 1.025, 'b',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax2.transAxes)
plt.text(0.3, 1.025, 'Southern mode',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight=fnt.get_weight(),
         fontstretch=fnt.get_stretch(),
         transform=ax1.transAxes)
plt.text(0.3, 1.025, 'Northern mode',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight=fnt.get_weight(),
         fontstretch=fnt.get_stretch(),
         transform=ax2.transAxes)
plt.text(0., 1.025, 'c',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax3.transAxes)
plt.text(0., 1.025, 'd',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax4.transAxes)
plt.text(0., 1.025, 'e',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax5.transAxes)
plt.text(0., 1.025, 'f',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax6.transAxes)
for i, l in enumerate(lags):
    ax_list = []
    if l == DAY:
        ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    if ax_list != []:
        z_list = [sz250[1, i], nz250[1, i],
                  sz500[1, i], nz500[1, i],
                  sz850[1, i], nz850[1, i]]
        v_list = [swv250[4, i], nwv250[4, i],
                  swv500[4, i], nwv500[4, i],
                  swv850[4, i], nwv850[4, i]]
        u_list = [swv250[5, i], nwv250[5, i],
                  swv500[5, i], nwv500[5, i],
                  swv850[5, i], nwv850[5, i]]
        box_list = [[regional_box('ARBSEA'), regional_box('SCN1')],
                    [regional_box('CMZ'), regional_box('NCN')],
                    [regional_box('ARBSEA'), regional_box('SCN1')],
                    [regional_box('CMZ'), regional_box('NCN')],
                    [regional_box('ARBSEA'), regional_box('SCN1')],
                    [regional_box('CMZ'), regional_box('NCN')]]
        pre_list = ['250', '250',
                    '500', '500',
                    '850', '850']
        for ax, z, v, u, box, pre in zip(ax_list, z_list, v_list, u_list, box_list, pre_list):
            bm = Basemap(ax=ax,
                         projection='mill',
                         llcrnrlon=lon0, llcrnrlat=lat0,
                         urcrnrlon=lon1, urcrnrlat=lat1,
                         lat_0=bm_lac, lon_0=shift)
            bm.drawparallels(latlabels,
                             labels=[1, 0, 0, 0],
                             family=fnt.get_family(),
                             fontsize=fnt.get_size(),
                             fontweight=fnt.get_weight(),
                             fontstretch=fnt.get_stretch(),
                             rotation='0',
                             dashes=[1, 1],
                             linewidth=.5,
                             color='darkgrey')
            bm.drawmeridians(lonlabels,
                             labels=[0, 0, 0, 1],
                             family=fnt.get_family(),
                             fontsize=fnt.get_size(),
                             fontweight=fnt.get_weight(),
                             fontstretch=fnt.get_stretch(),
                             rotation='0',
                             dashes=[1, 1],
                             linewidth=.3,
                             color='darkgrey')
            bm.drawcoastlines(linewidth=1., color='darkslategray')
            bm.drawcountries(linewidth=0.5, color='darkslategray')
            bm.drawmapboundary(linewidth=0., fill_color='aliceblue')

            lo, z = bm.shiftdata(lon, z, lon_0=shift)
            lo, la = np.meshgrid(lo, lat)
            lo, la = bm(lo, la)
            if pre == '250':
                cf_bm = bm.contourf(lo, la, z,
                                    levels=np.arange(-9, 10, 1.5),
                                    cmap=plt.cm.coolwarm,
                                    alpha=1.,
                                    extend='both',
                                    zorder=1)
                cb_bm = bm.colorbar(cf_bm, location='bottom',
                                    ticks=np.arange(-9, 10, 3),
                                    pad=.2, size=.1)
            if pre == '500':
                cf_bm = bm.contourf(lo, la, z,
                                    levels=np.arange(-9, 10, 1.5),
                                    cmap=plt.cm.coolwarm,
                                    alpha=1.,
                                    extend='both',
                                    zorder=1)
                cb_bm = bm.colorbar(cf_bm, location='bottom',
                                    ticks=np.arange(-9, 10, 3),
                                    pad=.2, size=.1)
            if pre == '850':
                cf_bm = bm.contourf(lo, la, z,
                                    levels=np.arange(-9, 10, 1.5),
                                    cmap=plt.cm.coolwarm,
                                    alpha=1.,
                                    extend='both',
                                    zorder=1)
                cb_bm = bm.colorbar(cf_bm, location='bottom',
                                    ticks=np.arange(-9, 10, 3),
                                    pad=.2, size=.1)
            cb_bm.ax.set_xlabel(r'GPH (%shPa) (m)' % pre,
                                family=fnt.get_family(),
                                fontsize=fnt.get_size(),
                                fontweight=fnt.get_weight(),
                                fontstretch=fnt.get_stretch())
            cb_bm.ax.tick_params(direction='in', color='black')
            cb_bm.outline.set_visible(False)
            for label in cb_bm.ax.get_xticklabels():
                label.set_fontproperties(fnt)

            lo, v = bm.shiftdata(lon, v, lon_0=shift)
            lo, u = bm.shiftdata(lon, u, lon_0=shift)
            lo, la = np.meshgrid(lo, lat)
            lo, la = bm(lo, la)
            if pre == '250':
                bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                          u[::ITV, ::ITV], v[::ITV, ::ITV],
                          scale=65, color='black',
                          # pivot='mid', units='width',
                          headwidth=7., headlength=7.,
                          zorder=2)
            if pre == '500':
                bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                          u[::ITV, ::ITV], v[::ITV, ::ITV],
                          scale=40, color='black',
                          # pivot='mid', units='width',
                          headwidth=7., headlength=7.,
                          zorder=2)
            if pre == '850':
                bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                          u[::ITV, ::ITV], v[::ITV, ::ITV],
                          scale=40, color='black',
                          # pivot='mid', units='width',
                          headwidth=7., headlength=7.,
                          zorder=2)
            patches = []
            reg1 = np.array([[box[0][2], box[0][0]],
                            [box[0][2], box[0][1]],
                            [box[0][3], box[0][1]],
                            [box[0][3], box[0][0]]])
            reg2 = np.array([[box[1][2], box[1][0]],
                            [box[1][2], box[1][1]],
                            [box[1][3], box[1][1]],
                            [box[1][3], box[1][0]]])
            x, y = bm(reg1[:, 0], reg1[:, 1])
            xy = np.vstack([x, y]).T
            poly = Polygon(xy,
                           facecolor='None',
                           edgecolor='gold',
                           linewidth=3,
                           zorder=2)
            plt.gca().add_patch(poly)
            x, y = bm(reg2[:, 0], reg2[:, 1])
            xy = np.vstack([x, y]).T
            poly = Polygon(xy,
                           facecolor='None',
                           edgecolor='gold',
                           linewidth=3,
                           zorder=2)
            plt.gca().add_patch(poly)
            plt.text(.12, .95, SEASON, ha='right', va='top',
                     family='Arial',
                     fontsize='x-large',
                     fontweight=fnt.get_weight(),
                     fontstretch=fnt.get_stretch(),
                     transform=ax.transAxes,
                     bbox=dict(facecolor='white',
                               edgecolor='none',
                               pad=1.,
                               alpha=0.9))
if 'png' in oup_ano21:
    plt.savefig(oup_ano21, dpi=300, bbox_inches='tight')
if 'tif' in oup_ano21:
    plt.savefig(oup_ano21, dpi=300, bbox_inches='tight')
if 'pdf' in oup_ano21:
    plt.savefig(oup_ano21, dpi=300, bbox_inches='tight')
plt.close()

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8.5, 7.5))
fig.tight_layout()
plt.subplots_adjust(hspace=.2, wspace=.2)
ax1, ax2 = axs[0][0], axs[0][1]
ax3, ax4 = axs[1][0], axs[1][1]
ax5, ax6 = axs[2][0], axs[2][1]
plt.text(0., 1.025, 'a',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax1.transAxes)
plt.text(0., 1.025, 'b',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax2.transAxes)
plt.text(0.3, 1.025, 'Southern mode',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight=fnt.get_weight(),
         fontstretch=fnt.get_stretch(),
         transform=ax1.transAxes)
plt.text(0.3, 1.025, 'Northern mode',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight=fnt.get_weight(),
         fontstretch=fnt.get_stretch(),
         transform=ax2.transAxes)
plt.text(0., 1.025, 'c',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax3.transAxes)
plt.text(0., 1.025, 'd',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax4.transAxes)
plt.text(0., 1.025, 'e',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax5.transAxes)
plt.text(0., 1.025, 'f',
         family=fnt.get_family(),
         fontsize='x-large',
         fontweight='bold',
         fontstretch=fnt.get_stretch(),
         transform=ax6.transAxes)
for i, l in enumerate(lags):
    ax_list = []
    if l == DAY:
        ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    if ax_list != []:
        z_list = [sz250[2, i], nz250[2, i],
                  sz500[2, i], nz500[2, i],
                  sz850[2, i], nz850[2, i]]
        v_list = [swv250[7, i], nwv250[7, i],
                  swv500[7, i], nwv500[7, i],
                  swv850[7, i], nwv850[7, i]]
        u_list = [swv250[8, i], nwv250[8, i],
                  swv500[8, i], nwv500[8, i],
                  swv850[8, i], nwv850[8, i]]
        box_list = [[regional_box('ARBSEA'), regional_box('SCN1')],
                    [regional_box('CMZ'), regional_box('NCN')],
                    [regional_box('ARBSEA'), regional_box('SCN1')],
                    [regional_box('CMZ'), regional_box('NCN')],
                    [regional_box('ARBSEA'), regional_box('SCN1')],
                    [regional_box('CMZ'), regional_box('NCN')]]
        pre_list = ['250', '250',
                    '500', '500',
                    '850', '850']
        for ax, z, v, u, box, pre in zip(ax_list, z_list, v_list, u_list, box_list, pre_list):
            bm = Basemap(ax=ax,
                         projection='mill',
                         llcrnrlon=lon0, llcrnrlat=lat0,
                         urcrnrlon=lon1, urcrnrlat=lat1,
                         lat_0=bm_lac, lon_0=shift)
            bm.drawparallels(latlabels,
                             labels=[1, 0, 0, 0],
                             family=fnt.get_family(),
                             fontsize=fnt.get_size(),
                             fontweight=fnt.get_weight(),
                             fontstretch=fnt.get_stretch(),
                             rotation='0',
                             dashes=[1, 1],
                             linewidth=.5,
                             color='darkgrey')
            bm.drawmeridians(lonlabels,
                             labels=[0, 0, 0, 1],
                             family=fnt.get_family(),
                             fontsize=fnt.get_size(),
                             fontweight=fnt.get_weight(),
                             fontstretch=fnt.get_stretch(),
                             rotation='0',
                             dashes=[1, 1],
                             linewidth=.3,
                             color='darkgrey')
            bm.drawcoastlines(linewidth=1., color='darkslategray')
            bm.drawcountries(linewidth=0.5, color='darkslategray')
            bm.drawmapboundary(linewidth=0., fill_color='aliceblue')

            lo, z = bm.shiftdata(lon, z, lon_0=shift)
            lo, la = np.meshgrid(lo, lat)
            lo, la = bm(lo, la)
            if pre == '250':
                cf_bm = bm.contourf(lo, la, z,
                                    levels=np.arange(-9, 10, 1.5),
                                    cmap=plt.cm.coolwarm,
                                    alpha=1.,
                                    extend='both',
                                    zorder=1)
                cb_bm = bm.colorbar(cf_bm, location='bottom',
                                    ticks=np.arange(-9, 10, 3),
                                    pad=.2, size=.1)
            if pre == '500':
                cf_bm = bm.contourf(lo, la, z,
                                    levels=np.arange(-9, 10, 1.5),
                                    cmap=plt.cm.coolwarm,
                                    alpha=1.,
                                    extend='both',
                                    zorder=1)
                cb_bm = bm.colorbar(cf_bm, location='bottom',
                                    ticks=np.arange(-9, 10, 3),
                                    pad=.2, size=.1)
            if pre == '850':
                cf_bm = bm.contourf(lo, la, z,
                                    levels=np.arange(-9, 10, 1.5),
                                    cmap=plt.cm.coolwarm,
                                    alpha=1.,
                                    extend='both',
                                    zorder=1)
                cb_bm = bm.colorbar(cf_bm, location='bottom',
                                    ticks=np.arange(-9, 10, 3),
                                    pad=.2, size=.1)
            cb_bm.ax.set_xlabel(r'GPH (%shPa) (m)' % pre,
                                family=fnt.get_family(),
                                fontsize=fnt.get_size(),
                                fontweight=fnt.get_weight(),
                                fontstretch=fnt.get_stretch())
            cb_bm.ax.tick_params(direction='in', color='black')
            cb_bm.outline.set_visible(False)
            for label in cb_bm.ax.get_xticklabels():
                label.set_fontproperties(fnt)

            lo, v = bm.shiftdata(lon, v, lon_0=shift)
            lo, u = bm.shiftdata(lon, u, lon_0=shift)
            lo, la = np.meshgrid(lo, lat)
            lo, la = bm(lo, la)
            if pre == '250':
                bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                          u[::ITV, ::ITV], v[::ITV, ::ITV],
                          scale=65, color='black',
                          # pivot='mid', units='width',
                          headwidth=7., headlength=7.,
                          zorder=2)
            if pre == '500':
                bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                          u[::ITV, ::ITV], v[::ITV, ::ITV],
                          scale=40, color='black',
                          # pivot='mid', units='width',
                          headwidth=7., headlength=7.,
                          zorder=2)
            if pre == '850':
                bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                          u[::ITV, ::ITV], v[::ITV, ::ITV],
                          scale=40, color='black',
                          # pivot='mid', units='width',
                          headwidth=7., headlength=7.,
                          zorder=2)
            patches = []
            reg1 = np.array([[box[0][2], box[0][0]],
                            [box[0][2], box[0][1]],
                            [box[0][3], box[0][1]],
                            [box[0][3], box[0][0]]])
            reg2 = np.array([[box[1][2], box[1][0]],
                            [box[1][2], box[1][1]],
                            [box[1][3], box[1][1]],
                            [box[1][3], box[1][0]]])
            x, y = bm(reg1[:, 0], reg1[:, 1])
            xy = np.vstack([x, y]).T
            poly = Polygon(xy,
                           facecolor='None',
                           edgecolor='gold',
                           linewidth=3,
                           zorder=2)
            plt.gca().add_patch(poly)
            x, y = bm(reg2[:, 0], reg2[:, 1])
            xy = np.vstack([x, y]).T
            poly = Polygon(xy,
                           facecolor='None',
                           edgecolor='gold',
                           linewidth=3,
                           zorder=2)
            plt.gca().add_patch(poly)
            plt.text(.12, .95, SEASON, ha='right', va='top',
                     family='Arial',
                     fontsize='x-large',
                     fontweight=fnt.get_weight(),
                     fontstretch=fnt.get_stretch(),
                     transform=ax.transAxes,
                     bbox=dict(facecolor='white',
                               edgecolor='none',
                               pad=1.,
                               alpha=0.9))
if 'png' in oup_anoboth:
    plt.savefig(oup_anoboth, dpi=300, bbox_inches='tight')
if 'tif' in oup_anoboth:
    plt.savefig(oup_anoboth, dpi=300, bbox_inches='tight')
if 'pdf' in oup_anoboth:
    plt.savefig(oup_anoboth, dpi=300, bbox_inches='tight')
plt.close()
