import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.optimize as so
import cmocean as oc
import types
from matplotlib import patches
from matplotlib.font_manager import FontProperties, rcParams
from matplotlib.animation import FuncAnimation
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from itertools import product
from matplotlib import ticker
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
SIGNIFICANCE = 95.
SPAZOOMOUT = 1
COVERAGE = 'ASM'
PROJECTION = 'mill'


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
    if p_reg == 'ARBSEA1':
        lat0 = 5.
        lat1 = 15.
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
    if p_reg == 'KJ':
        lat0 = 30.
        lat1 = 38.
        lon0 = 125.
        lon1 = 132.
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


oup_fig = './Submission/IJC/Fig_&_Table/Gupta_Figure_1.png'
deg = np.load(
    './Results/TRMM_Precipitation/ES_1998_To_2019ASM_JJA_900_d7_P950_Deg.npy')
pdeg1 = np.load(
    './Results/TRMM_Precipitation/ES_1998_To_2019ASM_JJA_900_d7_P950_RegTele[ASM-ARBSEA]_Deg.npy')
pdeg2 = np.load(
    './Results/TRMM_Precipitation/ES_1998_To_2019ASM_JJA_900_d7_P950_RegTele[ASM-CMZ]_Deg.npy')
pdeg3 = np.load(
    './Results/TRMM_Precipitation/ES_1998_To_2019ASM_6_900_d7_P950_Deg.npy')
pdeg4 = np.load(
    './Results/TRMM_Precipitation/ES_1998_To_2019ASM_7_900_d7_P950_Deg.npy')
pdeg5 = np.load(
    './Results/TRMM_Precipitation/ES_1998_To_2019ASM_8_900_d7_P950_Deg.npy')
with xr.open_zarr('./Results/TRMM_Precipitation/1998_To_2019ASM_JJA_900.zarr', consolidated=True) as dta:
    lat = dta['lat'].values
    lon = dta['lon'].values
    evt_num = dta['evt_num'].values
    crd = np.array(list(product(lat, lon)))
hch = np.where(evt_num < 3, -1, 1)

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
fig.tight_layout()
plt.subplots_adjust(hspace=.1, wspace=.2)
bm_lac = (lat.max() + lat.min()) / 2
bm_loc = (lon.max() + lon.min()) / 2
shift = bm_loc
ax1, ax2, ax3 = axs[0][0], axs[0][1], axs[0][2]
ax4, ax5, ax6 = axs[1][0], axs[1][1], axs[1][2]
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
latlabels = np.arange(int(lat.min() / 10 + 1) * 10,
                      int(lat.max() / 10 + 1) * 10, 15)
lonlabels = np.arange(int(lon.min() / 10 + 1) * 10,
                      int(lon.max() / 10 + 1) * 10, 30)

ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]
deg_list = [pdeg3, pdeg4, pdeg5, deg, pdeg1, pdeg2]
# box_list = ['',
#             '',
#             '',
#             '',
#             regional_box('ARBSEA'),
#             regional_box('CMZ')]
lvls_list = [np.arange(2000, 7001, 1000),
             np.arange(2000, 7001, 1000),
             np.arange(2000, 7001, 1000),
             np.arange(1000, 11001, 2000),
             np.arange(0, 550, 100),
             np.arange(0, 550, 100)]
tks_list = [np.arange(2500, 6501, 1000),
            np.arange(2500, 6501, 1000),
            np.arange(2500, 6501, 1000),
            np.arange(2000, 10001, 2000),
            np.arange(50, 451, 100),
            np.arange(50, 451, 100)]
blb_list = [r'Degree',
            r'Degree',
            r'Degree',
            r'Degree',
            r'Partial degree',
            r'Partial degree']
caption_list = ['June',
                'July',
                'August',
                SEASON,
                SEASON,
                SEASON]

for ax, deg, lvls, tks, blb, caption in zip(ax_list,
                                            deg_list,
                                            lvls_list,
                                            tks_list,
                                            blb_list,
                                            caption_list):
    bm = Basemap(ax=ax,
                 projection=PROJECTION,
                 llcrnrlon=int(lon.min()), llcrnrlat=int(lat.min()),
                 urcrnrlon=int(lon.max()), urcrnrlat=int(lat.max()),
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
                     linewidth=.8,
                     color='darkgrey')
    bm.drawcoastlines(linewidth=1.)
    bm.drawcountries(linewidth=.5, color='black')
    bm.drawmapboundary(linewidth=0., fill_color='aliceblue')
    lo, deg = bm.shiftdata(lon,
                           deg.reshape((lat.shape[0], lon.shape[0])),
                           lon_0=shift)
    lo, la = np.meshgrid(lo, lat)
    lo, la = bm(lo, la)
    cf = bm.contourf(lo, la, deg,
                     levels=lvls,
                     cmap=plt.cm.summer,
                     extend='both',
                     alpha=.8,
                     zorder=1)
    rcParams['hatch.linewidth'] = .2
    if 'png' in oup_fig:
        rcParams['hatch.color'] = 'black'
    if 'pdf' in oup_fig:
        rcParams['hatch.color'] = 'lightgray'
    lo, hch = bm.shiftdata(lon, hch, lon_0=shift)
    lo, la = np.meshgrid(lo, lat)
    lo, la = bm(lo, la)
    bm.contourf(lo, la, hch,
                levels=[-2, 0],
                colors='#EAEAF2',
                hatches=['//////'],
                alpha=.6,
                zorder=1)
    cb = bm.colorbar(cf, location='bottom',
                     ticks=tks,
                     pad=.2, size=.15)
    if ax in [ax1, ax2, ax3, ax4]:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((2, 2))
        cb.ax.xaxis.set_major_formatter(formatter)
    cb.ax.set_xlabel(blb,
                     family=fnt.get_family(),
                     fontsize=fnt.get_size(),
                     fontweight=fnt.get_weight(),
                     fontstretch=fnt.get_stretch())
    cb.ax.tick_params(direction='in', color='black')
    cb.outline.set_visible(False)
    for label in cb.ax.get_xticklabels():
        label.set_fontproperties(fnt)
    if ax in [ax1, ax2, ax3]:
        for box, clr in zip([regional_box('ARBSEA'),
                             regional_box('CMZ')],
                            ['red', 'blue']):
            ctr = np.array([[box[2], box[0]],
                            [box[2], box[1]],
                            [box[3], box[1]],
                            [box[3], box[0]]])
            x, y = bm(ctr[:, 0], ctr[:, 1])
            xy = np.vstack([x, y]).T
            poly = Polygon(xy,
                           facecolor='None',
                           edgecolor=clr,
                           linewidth=3,
                           zorder=2)
            plt.gca().add_patch(poly)
    if ax == ax5:
        for box, clr, c in zip([regional_box('ARBSEA'), regional_box('SCN1')],
                               ['red', 'red'],
                               [1, 2]):
            ctr = np.array([[box[2], box[0]],
                            [box[2], box[1]],
                            [box[3], box[1]],
                            [box[3], box[0]]])
            x, y = bm(ctr[:, 0], ctr[:, 1])
            xy = np.vstack([x, y]).T
            if c == 1:
                poly = Polygon(xy,
                               facecolor='None',
                               edgecolor=clr,
                               linewidth=3,
                               zorder=2)
            if c == 2:
                poly = Polygon(xy,
                               facecolor='None',
                               edgecolor=clr,
                               linewidth=3,
                               linestyle=(0, (1, 1)),
                               zorder=2)
            plt.gca().add_patch(poly)
            plt.text(.01, .30, 'ARB', ha='left', va='top',
                     family='Arial',
                     fontsize='x-large',
                     fontweight='bold',
                     fontstretch=fnt.get_stretch(),
                     color='black',
                     transform=ax.transAxes,
                     bbox=dict(facecolor='none',
                               edgecolor='none',
                               pad=1.,
                               alpha=0.9))
            plt.text(.44, .65, 'SCN', ha='left', va='top',
                     family='Arial',
                     fontsize='x-large',
                     fontweight='bold',
                     fontstretch=fnt.get_stretch(),
                     color='black',
                     transform=ax.transAxes,
                     bbox=dict(facecolor='none',
                               edgecolor='none',
                               pad=1.,
                               alpha=0.9))
    if ax == ax6:
        for box, clr, c in zip([regional_box('CMZ'), regional_box('NCN')],
                               ['blue', 'blue'],
                               [1, 2]):
            ctr = np.array([[box[2], box[0]],
                            [box[2], box[1]],
                            [box[3], box[1]],
                            [box[3], box[0]]])
            x, y = bm(ctr[:, 0], ctr[:, 1])
            xy = np.vstack([x, y]).T
            if c == 1:
                poly = Polygon(xy,
                               facecolor='None',
                               edgecolor=clr,
                               linewidth=3,
                               zorder=2)
            if c == 2:
                poly = Polygon(xy,
                               facecolor='None',
                               edgecolor=clr,
                               linewidth=3,
                               linestyle=(0, (1, 1)),
                               zorder=2)
            plt.gca().add_patch(poly)
            plt.text(.13, .55, 'CMZ', ha='left', va='top',
                     family='Arial',
                     fontsize='x-large',
                     fontweight='bold',
                     fontstretch=fnt.get_stretch(),
                     transform=ax.transAxes,
                     color='black',
                     bbox=dict(facecolor='none',
                               edgecolor='none',
                               pad=1.,
                               alpha=0.9))
            plt.text(.34, .80, 'NCN', ha='left', va='top',
                     family='Arial',
                     fontsize='x-large',
                     fontweight='bold',
                     fontstretch=fnt.get_stretch(),
                     transform=ax.transAxes,
                     color='black',
                     bbox=dict(facecolor='none',
                               edgecolor='none',
                               pad=1.,
                               alpha=0.9))
    if ax in [ax1, ax2, ax3]:
        plt.text(.05, .95, caption, ha='left', va='top',
                 family='Arial',
                 fontsize='x-large',
                 fontweight=fnt.get_weight(),
                 fontstretch=fnt.get_stretch(),
                 transform=ax.transAxes,
                 bbox=dict(facecolor='white',
                           edgecolor='none',
                           pad=1.,
                           alpha=0.9))
    else:
        plt.text(.05, .95, caption, ha='left', va='top',
                 family='Arial',
                 fontsize='x-large',
                 fontweight=fnt.get_weight(),
                 fontstretch=fnt.get_stretch(),
                 transform=ax.transAxes,
                 bbox=dict(facecolor='white',
                           edgecolor='none',
                           pad=1.,
                           alpha=0.9))

if 'png' in oup_fig:
    plt.savefig(oup_fig, dpi=300, bbox_inches='tight')
if 'tif' in oup_fig:
    plt.savefig(oup_fig, dpi=300, bbox_inches='tight')
if 'pdf' in oup_fig:
    plt.savefig(oup_fig, dpi=300, bbox_inches='tight')
