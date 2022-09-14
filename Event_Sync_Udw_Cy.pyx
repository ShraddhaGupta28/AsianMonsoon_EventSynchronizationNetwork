# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
# distutils: language = c
# cython: c_string_type=unicode, c_string_encoding=utf8
import os
import zarr as zr
import xarray as xr
import numpy as np
cimport numpy as cnp
cimport cython as cy
from cython.parallel import prange


cdef str DATASETS = 'TRMM_Precipitation'
cdef str PERIOD = '1998_To_2019ASM.nc4'
cdef str SEASON = 'JJA'
cdef cnp.float64_t PERCENTILE = 90.  # from 80% to 99%
cdef cnp.int8_t TAUMAX = 7  # the maximum temporal delay
cdef cnp.float64_t SIGNIFICANCE = 95.  # the significance level, 99.5%
cdef cnp.int8_t SPAZOOMOUT = 1  # zoom out in spatial

cdef cnp.int8_t CPU = 15
cdef cnp.int8_t THREAD = 16
cdef cnp.int32_t ADJGRPSIZE = 100000000

cdef str inp_evt
inp_evt = './Results/%s/%s_%s_%s.zarr' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1])
if SPAZOOMOUT != 1:
    inp_evt = './Results/%s/%s_%s_%s_X%s.zarr' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(SPAZOOMOUT))

cdef str inp_sig = './Results/%s/ESNM_%s_%s_%s_d%s_P%s.npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])

cdef str oup_net
oup_net = './Results/%s/ES_%s_%s_%s_d%s_P%s_Udw' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])
if SPAZOOMOUT != 1:
    oup_net = './Results/%s/ES_%s_%s_%s_d%s_P%s_X%s_Udw' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(SPAZOOMOUT))


@cy.boundscheck(False)
@cy.wraparound(False)
cpdef event_sync():
    
    cdef:
        cnp.int32_t[:, :] evt_srs
        cnp.uint16_t[:] evt_num
        cnp.float32_t[:, :] sig, rlt_v
        cnp.ndarray[cnp.float32_t, ndim=2] adj, rlt
        Py_ssize_t cnt_adj=0, cnt_grp=0, new=0, nds=0, i=0, j=0
    
    with xr.open_zarr(inp_evt, consolidated=True) as dta:
        evt_srs = dta['evt_srs'].stack(nid=('lat', 'lon')).transpose('nid', ...).values.astype(np.int32)
        evt_num = dta['evt_num'].stack(nid=('lat', 'lon')).values.astype(np.uint16)
        nds = evt_num.shape[0]

        # load significance
        sig = np.load(inp_sig)
        adj = np.zeros((ADJGRPSIZE, 3), dtype=np.float32)

        cnt_adj=0
        cnt_grp=0
        new=0
        for i in range(nds):
            print("Working on id: ", i + 1)
            # grid cells which have events, in parallelization
            if evt_num[i] != 0:
                rlt = np.zeros((nds, 3), dtype=np.float32)
                rlt_v = rlt
                for j in prange(i + 1, nds, 1, num_threads=THREAD, nogil=True):
                    es_core(i, j,
                            evt_srs[i], evt_srs[j],
                            evt_num[i], evt_num[j],
                            TAUMAX, sig[evt_num[i], evt_num[j]],
                            rlt_v[j, :])
                rlt = rlt[np.where(rlt[:, 0] != 0)[0], :]
            else:
                rlt = np.empty((0, 0), dtype=np.float32)
                
            if rlt.shape[0] != 0:
                new = cnt_adj + rlt.shape[0]
                # insert
                if new < ADJGRPSIZE:
                    adj[cnt_adj:new, :] = rlt
                    cnt_adj += rlt.shape[0]
                if new >= ADJGRPSIZE:
                    adj[cnt_adj:, :] = rlt[:ADJGRPSIZE - cnt_adj, :]
            # append and update
            if (new >= ADJGRPSIZE) or (i == nds - 1):
                cnt_grp += 1
                np.save(oup_net + '_' + str(cnt_grp) + '.npy', adj)
            if new >= ADJGRPSIZE:
                adj[:new - ADJGRPSIZE, :] = rlt[ADJGRPSIZE - cnt_adj:, :]
                cnt_adj = new - ADJGRPSIZE
                new = cnt_adj


@cy.boundscheck(False)
@cy.wraparound(False)
cpdef inline void es_core(Py_ssize_t p_li, Py_ssize_t p_lj,
cnp.int32_t[:] p_eli, cnp.int32_t[:] p_elj,
cnp.uint16_t p_nli, cnp.uint16_t p_nlj,
cnp.int8_t p_tmx, cnp.float32_t p_sig,
cnp.float32_t[:] p_rlt) nogil:
    """
    P1 P2: adj location i j,
    P3 P4: event series at i j,
    P5 P6: event number at i j,
    P7: maximum temporal delay,
    P8: significance level,
    P9: result array,
    output core
    """

    cdef:
        cnp.uint16_t es = 0
        Py_ssize_t i_dim, j_dim, i, j
        cnp.int32_t dly
        cnp.float64_t tij

    i_dim, j_dim = p_eli.shape[0], p_elj.shape[0]
    for i in range(1, p_nli - 1):
        for j in range(1, p_nlj - 1):
            dly = p_eli[i] - p_elj[j]
            if dly > p_tmx:
                continue
            if dly < -p_tmx:
                break
            tij = min(min(p_eli[i] - p_eli[i - 1],
                          p_eli[i + 1] - p_eli[i]),
                      min(p_elj[j] - p_elj[j - 1],
                          p_elj[j + 1] - p_elj[j])) / 2
            if dly < 0:
                dly = -dly
            if dly < tij and dly <= p_tmx:
                es += 1
    if es > p_sig:
        p_rlt[0], p_rlt[1], p_rlt[2] = <float>(p_li + 1), <float>(p_lj + 1), <float>(es)