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
STARTDAY = 0
ENDDAYS1 = 4
ENDDAYS2 = 14
ENDDAYN1 = 2
ENDDAYN2 = 14
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


swvano = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5WVFCAno[ARBSEA-SCN1].npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
# swvcps = np.load(
#     './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5WVFCps[ARBSEA-SCN1].npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
nwvano = np.load(
    './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5WVFCAno[CMZ-NCN].npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
# nwvcps = np.load(
#     './Results/TRMM_Precipitation/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5WVFCps[CMZ-NCN].npy' % (PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1]))
with xr.open_dataset('C:/~Dataset/Clim_ERA5_WVFVC_1X1_1998_To_2019.nc') as dta:
    lat = dta['latitude'].values.astype(np.float32)
    lon = dta['longitude'].values.astype(np.float32)

lat0, lat1, lon0, lon1 = -15, 51, 0, 160
latlabels = np.arange(int(lat0 / 10) * 10, int(lat1 / 10) * 10 + 1, 20)
lonlabels = np.arange(int(lon0 / 10) * 10, int(lon1 / 10) * 10 + 1, 40)
bm_lac, bm_loc = (lat0 + lat1) / 2, (lon0 + lon1) / 2
shift = bm_loc

lags = np.arange(-30, 31, 1)
oup_ano12 = './Submission/IJC/Fig_&_Table/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5WVFSN12.png' % (
    PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])
oup_ano21 = './Submission/IJC/Fig_&_Table/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5WVFSN21.png' % (
    PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])
oup_anoboth = './Submission/IJC/Fig_&_Table/ES_%s_JJA_900_d%s_P%s_MSSD_ERA5WVFSNboth.png' % (
    PERIOD.split('.')[0], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8.5, 5))
fig.tight_layout()
plt.subplots_adjust(hspace=.35, wspace=.2)
ax1, ax2 = axs[0][0], axs[0][1]
ax3, ax4 = axs[1][0], axs[1][1]
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
ax1_wvf = np.zeros_like(swvano[0, 0])
ax1_v = np.zeros_like(swvano[1, 0])
ax1_u = np.zeros_like(swvano[2, 0])
ax2_wvf = np.zeros_like(swvano[0, 0])
ax2_v = np.zeros_like(swvano[1, 0])
ax2_u = np.zeros_like(swvano[2, 0])
ax3_wvf = np.zeros_like(nwvano[0, 0])
ax3_v = np.zeros_like(nwvano[1, 0])
ax3_u = np.zeros_like(nwvano[2, 0])
ax4_wvf = np.zeros_like(nwvano[0, 0])
ax4_v = np.zeros_like(nwvano[1, 0])
ax4_u = np.zeros_like(nwvano[2, 0])
for i, l in enumerate(lags):
    if STARTDAY <= l <= ENDDAYS1:
        ax1_wvf += swvano[0, i]
        apt = np.sqrt(swvano[1, i] ** 2 + swvano[2, i] ** 2)
        ax1_v += swvano[1, i] / apt
        ax1_u += swvano[2, i] / apt
    if STARTDAY <= l <= ENDDAYS2:
        ax2_wvf += swvano[0, i]
        apt = np.sqrt(swvano[1, i] ** 2 + swvano[2, i] ** 2)
        ax2_v += swvano[1, i] / apt
        ax2_u += swvano[2, i] / apt
    if STARTDAY <= l <= ENDDAYN1:
        ax3_wvf += nwvano[0, i]
        apt = np.sqrt(nwvano[1, i] ** 2 + nwvano[2, i] ** 2)
        ax3_v += nwvano[1, i] / apt
        ax3_u += nwvano[2, i] / apt
    if STARTDAY <= l <= ENDDAYN2:
        ax4_wvf += nwvano[0, i]
        apt = np.sqrt(nwvano[1, i] ** 2 + nwvano[2, i] ** 2)
        ax4_v += nwvano[1, i] / apt
        ax4_u += nwvano[2, i] / apt
ax_list = [ax1, ax2, ax3, ax4]
wvf_list = [ax1_wvf, ax3_wvf, ax2_wvf, ax4_wvf]
v_list = [ax1_v, ax3_v, ax2_v, ax4_v]
u_list = [ax1_u, ax3_u, ax2_u, ax4_u]
box_list = [[regional_box('ARBSEA'), regional_box('SCN1')],
            [regional_box('CMZ'), regional_box('NCN')],
            [regional_box('ARBSEA'), regional_box('SCN1')],
            [regional_box('CMZ'), regional_box('NCN')]]
for ax, wvf, v, u, box in zip(ax_list, wvf_list, v_list, u_list, box_list):
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
    lo, wvf = bm.shiftdata(lon, wvf, lon_0=shift)
    lo, la = np.meshgrid(lo, lat)
    lo, la = bm(lo, la)
    if ax in [ax1, ax2]:
        cf_bm = bm.contourf(lo, la, wvf,
                            levels=np.arange(-300, 300.1, 10),
                            cmap=plt.cm.coolwarm,
                            alpha=1.,
                            extend='both',
                            zorder=1)
        cb_bm = bm.colorbar(cf_bm, location='bottom',
                            ticks=np.arange(-300, 300.1, 100),
                            pad=.2, size=.1)
    if ax in [ax3, ax4]:
        cf_bm = bm.contourf(lo, la, wvf,
                            levels=np.arange(-600, 600.1, 20),
                            cmap=plt.cm.coolwarm,
                            alpha=1.,
                            extend='both',
                            zorder=1)
        cb_bm = bm.colorbar(cf_bm, location='bottom',
                            ticks=np.arange(-600, 600.1, 200),
                            pad=.2, size=.1)
    cb_bm.ax.set_xlabel(r'IVT (kg m$^{-1}$s$^{-1}$)',
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
    if ax == ax1:
        bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                  u[::ITV, ::ITV], v[::ITV, ::ITV],
                  scale=150, color='black',
                  # pivot='mid', units='width',
                  headwidth=7., headlength=7.,
                  zorder=2)
    if ax == ax2:
        bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                  u[::ITV, ::ITV], v[::ITV, ::ITV],
                  scale=100, color='black',
                  # pivot='mid', units='width',
                  headwidth=7., headlength=7.,
                  zorder=2)
    if ax == ax3:
        bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                  u[::ITV, ::ITV], v[::ITV, ::ITV],
                  scale=350, color='black',
                  # pivot='mid', units='width',
                  headwidth=7., headlength=7.,
                  zorder=2)
    if ax == ax4:
        bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                  u[::ITV, ::ITV], v[::ITV, ::ITV],
                  scale=450, color='black',
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

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8.5, 5))
fig.tight_layout()
plt.subplots_adjust(hspace=.35, wspace=.2)
ax1, ax2 = axs[0][0], axs[0][1]
ax3, ax4 = axs[1][0], axs[1][1]
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
ax1_wvf = np.zeros_like(swvano[0, 0])
ax1_v = np.zeros_like(swvano[1, 0])
ax1_u = np.zeros_like(swvano[2, 0])
ax2_wvf = np.zeros_like(swvano[0, 0])
ax2_v = np.zeros_like(swvano[1, 0])
ax2_u = np.zeros_like(swvano[2, 0])
ax3_wvf = np.zeros_like(nwvano[0, 0])
ax3_v = np.zeros_like(nwvano[1, 0])
ax3_u = np.zeros_like(nwvano[2, 0])
ax4_wvf = np.zeros_like(nwvano[0, 0])
ax4_v = np.zeros_like(nwvano[1, 0])
ax4_u = np.zeros_like(nwvano[2, 0])
for i, l in enumerate(lags):
    if STARTDAY <= l <= ENDDAYS1:
        ax1_wvf += swvano[3, i]
        apt = np.sqrt(swvano[4, i] ** 2 + swvano[5, i] ** 2)
        ax1_v += swvano[4, i] / apt
        ax1_u += swvano[5, i] / apt
    if STARTDAY <= l <= ENDDAYS2:
        ax2_wvf += swvano[3, i]
        apt = np.sqrt(swvano[4, i] ** 2 + swvano[5, i] ** 2)
        ax2_v += swvano[4, i] / apt
        ax2_u += swvano[5, i] / apt
    if STARTDAY <= l <= ENDDAYN1:
        ax3_wvf += nwvano[3, i]
        apt = np.sqrt(nwvano[4, i] ** 2 + nwvano[5, i] ** 2)
        ax3_v += nwvano[4, i] / apt
        ax3_u += nwvano[5, i] / apt
    if STARTDAY <= l <= ENDDAYN2:
        ax4_wvf += nwvano[3, i]
        apt = np.sqrt(nwvano[4, i] ** 2 + nwvano[5, i] ** 2)
        ax4_v += nwvano[4, i] / apt
        ax4_u += nwvano[5, i] / apt
ax_list = [ax1, ax2, ax3, ax4]
wvf_list = [ax1_wvf, ax3_wvf, ax2_wvf, ax4_wvf]
v_list = [ax1_v, ax3_v, ax2_v, ax4_v]
u_list = [ax1_u, ax3_u, ax2_u, ax4_u]
box_list = [[regional_box('ARBSEA'), regional_box('SCN1')],
            [regional_box('CMZ'), regional_box('NCN')],
            [regional_box('ARBSEA'), regional_box('SCN1')],
            [regional_box('CMZ'), regional_box('NCN')]]
for ax, wvf, v, u, box in zip(ax_list, wvf_list, v_list, u_list, box_list):
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
    lo, wvf = bm.shiftdata(lon, wvf, lon_0=shift)
    lo, la = np.meshgrid(lo, lat)
    lo, la = bm(lo, la)
    if ax in [ax1, ax2]:
        cf_bm = bm.contourf(lo, la, wvf,
                            levels=np.arange(-300, 300.1, 10),
                            cmap=plt.cm.coolwarm,
                            alpha=1.,
                            extend='both',
                            zorder=1)
        cb_bm = bm.colorbar(cf_bm, location='bottom',
                            ticks=np.arange(-300, 300.1, 100),
                            pad=.2, size=.1)
    if ax in [ax3, ax4]:
        cf_bm = bm.contourf(lo, la, wvf,
                            levels=np.arange(-600, 600.1, 20),
                            cmap=plt.cm.coolwarm,
                            alpha=1.,
                            extend='both',
                            zorder=1)
        cb_bm = bm.colorbar(cf_bm, location='bottom',
                            ticks=np.arange(-600, 600.1, 200),
                            pad=.2, size=.1)
    cb_bm.ax.set_xlabel(r'IVT (kg m$^{-1}$s$^{-1}$)',
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
    if ax == ax1:
        bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                  u[::ITV, ::ITV], v[::ITV, ::ITV],
                  scale=150, color='black',
                  # pivot='mid', units='width',
                  headwidth=7., headlength=7.,
                  zorder=2)
    if ax == ax2:
        bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                  u[::ITV, ::ITV], v[::ITV, ::ITV],
                  scale=100, color='black',
                  # pivot='mid', units='width',
                  headwidth=7., headlength=7.,
                  zorder=2)
    if ax == ax3:
        bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                  u[::ITV, ::ITV], v[::ITV, ::ITV],
                  scale=350, color='black',
                  # pivot='mid', units='width',
                  headwidth=7., headlength=7.,
                  zorder=2)
    if ax == ax4:
        bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                  u[::ITV, ::ITV], v[::ITV, ::ITV],
                  scale=450, color='black',
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

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8.5, 5))
fig.tight_layout()
plt.subplots_adjust(hspace=.35, wspace=.2)
ax1, ax2 = axs[0][0], axs[0][1]
ax3, ax4 = axs[1][0], axs[1][1]
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
ax1_wvf = np.zeros_like(swvano[0, 0])
ax1_v = np.zeros_like(swvano[1, 0])
ax1_u = np.zeros_like(swvano[2, 0])
ax2_wvf = np.zeros_like(swvano[0, 0])
ax2_v = np.zeros_like(swvano[1, 0])
ax2_u = np.zeros_like(swvano[2, 0])
ax3_wvf = np.zeros_like(nwvano[0, 0])
ax3_v = np.zeros_like(nwvano[1, 0])
ax3_u = np.zeros_like(nwvano[2, 0])
ax4_wvf = np.zeros_like(nwvano[0, 0])
ax4_v = np.zeros_like(nwvano[1, 0])
ax4_u = np.zeros_like(nwvano[2, 0])
for i, l in enumerate(lags):
    if STARTDAY <= l <= ENDDAYS1:
        ax1_wvf += swvano[6, i]
        apt = np.sqrt(swvano[7, i] ** 2 + swvano[8, i] ** 2)
        ax1_v += swvano[7, i] / apt
        ax1_u += swvano[8, i] / apt
    if STARTDAY <= l <= ENDDAYS2:
        ax2_wvf += swvano[6, i]
        apt = np.sqrt(swvano[7, i] ** 2 + swvano[8, i] ** 2)
        ax2_v += swvano[7, i] / apt
        ax2_u += swvano[8, i] / apt
    if STARTDAY <= l <= ENDDAYN1:
        ax3_wvf += nwvano[6, i]
        apt = np.sqrt(nwvano[7, i] ** 2 + nwvano[8, i] ** 2)
        ax3_v += nwvano[7, i] / apt
        ax3_u += nwvano[8, i] / apt
    if STARTDAY <= l <= ENDDAYN2:
        ax4_wvf += nwvano[6, i]
        apt = np.sqrt(nwvano[7, i] ** 2 + nwvano[8, i] ** 2)
        ax4_v += nwvano[7, i] / apt
        ax4_u += nwvano[8, i] / apt
ax_list = [ax1, ax2, ax3, ax4]
wvf_list = [ax1_wvf, ax3_wvf, ax2_wvf, ax4_wvf]
v_list = [ax1_v, ax3_v, ax2_v, ax4_v]
u_list = [ax1_u, ax3_u, ax2_u, ax4_u]
box_list = [[regional_box('ARBSEA'), regional_box('SCN1')],
            [regional_box('CMZ'), regional_box('NCN')],
            [regional_box('ARBSEA'), regional_box('SCN1')],
            [regional_box('CMZ'), regional_box('NCN')]]
for ax, wvf, v, u, box in zip(ax_list, wvf_list, v_list, u_list, box_list):
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
    lo, wvf = bm.shiftdata(lon, wvf, lon_0=shift)
    lo, la = np.meshgrid(lo, lat)
    lo, la = bm(lo, la)
    if ax in [ax1, ax2]:
        cf_bm = bm.contourf(lo, la, wvf,
                            levels=np.arange(-300, 300.1, 10),
                            cmap=plt.cm.coolwarm,
                            alpha=1.,
                            extend='both',
                            zorder=1)
        cb_bm = bm.colorbar(cf_bm, location='bottom',
                            ticks=np.arange(-300, 300.1, 100),
                            pad=.2, size=.1)
    if ax in [ax3, ax4]:
        cf_bm = bm.contourf(lo, la, wvf,
                            levels=np.arange(-600, 600.1, 20),
                            cmap=plt.cm.coolwarm,
                            alpha=1.,
                            extend='both',
                            zorder=1)
        cb_bm = bm.colorbar(cf_bm, location='bottom',
                            ticks=np.arange(-600, 600.1, 200),
                            pad=.2, size=.1)
    cb_bm.ax.set_xlabel(r'IVT (kg m$^{-1}$s$^{-1}$)',
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
    if ax == ax1:
        bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                  u[::ITV, ::ITV], v[::ITV, ::ITV],
                  scale=150, color='black',
                  # pivot='mid', units='width',
                  headwidth=7., headlength=7.,
                  zorder=2)
    if ax == ax2:
        bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                  u[::ITV, ::ITV], v[::ITV, ::ITV],
                  scale=100, color='black',
                  # pivot='mid', units='width',
                  headwidth=7., headlength=7.,
                  zorder=2)
    if ax == ax3:
        bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                  u[::ITV, ::ITV], v[::ITV, ::ITV],
                  scale=350, color='black',
                  # pivot='mid', units='width',
                  headwidth=7., headlength=7.,
                  zorder=2)
    if ax == ax4:
        bm.quiver(lo[::ITV, ::ITV], la[::ITV, ::ITV],
                  u[::ITV, ::ITV], v[::ITV, ::ITV],
                  scale=450, color='black',
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
