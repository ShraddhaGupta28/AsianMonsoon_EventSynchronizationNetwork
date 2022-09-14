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
from itertools import product


cdef str DATASETS = 'TRMM_Precipitation'
cdef str PERIOD = '1998_To_2019ASM.nc4'
cdef str SEASON = 'JJA'
cdef cnp.float64_t PERCENTILE = 90.  # from 80% to 99%
cdef cnp.int8_t TAUMAX = 7  # the maximum temporal delay
cdef cnp.float64_t SIGNIFICANCE = 95.  # the significance level, 99.5%
cdef cnp.int8_t SPAZOOMOUT = 1  # zoom out in spatial

cdef list REGION_LIST = [['ARBSEA', 'SCN1'],
                         ['CMZ', 'NCN']]

cdef cnp.int8_t CPU = 15
cdef cnp.int8_t THREAD = 16


@cy.boundscheck(False)
@cy.wraparound(False)
cpdef object regional_box(str p_reg):
    cdef cnp.float64_t lat0=0.
    cdef cnp.float64_t lat1=0.
    cdef cnp.float64_t lon0=0.
    cdef cnp.float64_t lon1=0.
    if p_reg == 'sBALKANS':
        lat0 = 39.
        lat1 = 47.
        lon0 = 15.
        lon1 = 29.
    if p_reg == 'sSCN1':
        lat0=24.
        lat1=30.
        lon0=105.
        lon1=118.
    if p_reg == 'sWCEU':
        lat0=44.
        lat1=50.
        lon0=0.
        lon1=15.
    if p_reg == 'sNISM':
        lat0=25.
        lat1=32.
        lon0=71.
        lon1=88.
    if p_reg == 'sARB':
        lat0 = 5.
        lat1 = 15.
        lon0 = 60.
        lon1 = 75.
    if p_reg == 'sGC':
        lat0 = 15.
        lat1 = 25.
        lon0 = -95.
        lon1 = -80.
    if p_reg == 'sNSA':
        lat0 = -3.
        lat1 = 8.
        lon0 = -77.
        lon1 = -67.
    if p_reg == 'sCNA':
        lat0 = 30.
        lat1 = 40.
        lon0 = -102.
        lon1 = -93.
    if p_reg == 'sAF1':
        lat0 = 2.
        lat1 = 7.
        lon0 = -15.
        lon1 = 10.
    if p_reg == 'sNCN':
        lat0=36.
        lat1=42.
        lon0=108.
        lon1=118.
    if p_reg == 'sEUR':
        lat0=44.
        lat1=50.
        lon0=3.
        lon1=17.
    if p_reg == 'sSCN':
        lat0=25.
        lat1=31.
        lon0=106.
        lon1=120.
    if p_reg == 'sSEEU':
        lat0=43.
        lat1=50.
        lon0=20.
        lon1=31.
    if p_reg == 'CISM':
        lat0=21.
        lat1=28.
        lon0=71.
        lon1=88.
    if p_reg == 'NCN':
        lat0=36.
        lat1=42.
        lon0=108.
        lon1=118.
    if p_reg == 'NISM':
        lat0=25.
        lat1=32.
        lon0=71.
        lon1=88.
    if p_reg == 'CMZ':
        lat0=20.
        lat1=32.
        lon0=71.
        lon1=88.
    if p_reg == 'SISM':
        lat0=0.
        lat1=15.
        lon0=70.
        lon1=82.
    if p_reg == 'ARBSEA':
        lat0=5.
        lat1=20.
        lon0=60.
        lon1=75.
    if p_reg == 'EUR':
        lat0=42.
        lat1=50.
        lon0=3.
        lon1=15.
    if p_reg == 'SCN':
        lat0=25.5
        lat1=31.5
        lon0=113.
        lon1=130.
    if p_reg == 'SCN1':
        lat0=23.
        lat1=29.
        lon0=105.
        lon1=115.
    if p_reg == 'SCN2':
        lat0=27.
        lat1=33.
        lon0=112.
        lon1=122.
    if p_reg == 'JSEA':
        lat0=37.5
        lat1=41.5
        lon0=128.
        lon1=141.
    if p_reg == 'PHSEA':
        lat0=15.5
        lat1=25.5
        lon0=120.
        lon1=135.
    if p_reg == 'NCSISM':
        lat0=15.
        lat1=28.
        lon0=71.
        lon1=88.
    if p_reg == 'EJP':
        lat0=38.
        lat1=46.
        lon0=138.
        lon1=152.
    if p_reg == 'NEPF':
        lat0=10.
        lat1=16.
        lon0=-170.
        lon1=-155.
    if p_reg == 'SEEU':
        lat0=41.
        lat1=50.
        lon0=20.
        lon1=36.
    return np.array([lat0, lat1, lon0, lon1], dtype=np.float64)


@cy.boundscheck(False)
@cy.wraparound(False)
cpdef regional_sync():

    cdef:
        cnp.ndarray[cnp.float32_t, ndim=1] lat, lon
        cnp.ndarray[cnp.uint16_t, ndim=1] ttl, tse, evt_num, num1, num2
        cnp.ndarray[cnp.int32_t, ndim=2] evt_srs, srs1, srs2
        cnp.ndarray[cnp.float32_t, ndim=2] crd
        cnp.ndarray[cnp.float64_t, ndim=1] box1, box2
        cnp.ndarray[cnp.int32_t, ndim=1] reg1, reg2, rlt121, rlt122, rlt211, rlt212
        cnp.ndarray[cnp.int64_t, ndim=1] t121, t122, t211, t212
        cnp.uint16_t[:] tse_v
        cnp.int32_t[:, :] srs1_v, srs2_v
        cnp.uint16_t[:] num1_v, num2_v
        cnp.int32_t[:] rlt121_v, rlt122_v, rlt211_v, rlt212_v
        cnp.float32_t[:, :] sig
        Py_ssize_t nds1=0, nds2=0, lt=0, lse=0, t=0, c121=0, c122=0, c211=0, c212=0, i=0, j=0
        str inp_evt=''
        str inp_sig=''
        str oup_reg_sync=''
        list REGION=[]

    for REGION in REGION_LIST:
        inp_evt = './Results/%s/%s_%s_%s.zarr' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1])
        if SPAZOOMOUT != 1:
            inp_evt = './Results/%s/%s_%s_%s_X%s.zarr' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(SPAZOOMOUT))

        inp_sig = './Results/%s/ES2NM_%s_%s_%s_d%s_P%s.npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])
        
        oup_reg_sync = './Results/%s/ES_%s_%s_%s_d%s_P%s_MS_SyncTs[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], REGION[0], REGION[1])
        if SPAZOOMOUT != 1:
            oup_reg_sync = './Results/%s/ES_%s_%s_%s_d%s_P%s_X%s_MS_SyncTs[%s-%s].npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(SPAZOOMOUT), REGION[0], REGION[1])
            
        with xr.open_zarr(inp_evt, consolidated=True) as dta:
            lat = dta['lat'].values.astype(np.float32)
            lon = dta['lon'].values.astype(np.float32)
            ttl = dta['ttl'].values.astype(np.uint16)
            tse = dta['tse'].values.astype(np.uint16)
            evt_srs = dta['evt_srs'].stack(nid=('lat', 'lon')).transpose('nid', ...).values.astype(np.int32)
            evt_num = dta['evt_num'].stack(nid=('lat', 'lon')).values.astype(np.uint16)

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
        srs1 = evt_srs[reg1]
        srs2 = evt_srs[reg2]
        num1 = evt_num[reg1]
        num2 = evt_num[reg2]
        nds1 = reg1.shape[0]
        nds2 = reg2.shape[0]
        lt = ttl.shape[0]
        lse = tse.shape[0]
        t121 = np.zeros(lt, dtype=np.int64)
        t122 = np.zeros(lt, dtype=np.int64)
        t211 = np.zeros(lt, dtype=np.int64)
        t212 = np.zeros(lt, dtype=np.int64)
        tse_v = tse
        srs1_v = srs1
        srs2_v = srs2
        num1_v = num1
        num2_v = num2
        sig = np.load(inp_sig)
        print(nds1, nds2)

        for t in range(1, lse - 1):
            print("Working on: ", t)
            c121 = 0
            c122 = 0
            for i in range(nds1):
                # print("Working on: ", i)
                rlt121 = np.zeros(nds2, dtype=np.int32)
                rlt121_v = rlt121
                for j in prange(nds2, num_threads=THREAD, nogil=True):
                # for j in range(nds2):
                    regional_sync_core(j, srs1_v[i], srs2_v[j],
                                       num1_v[i], num2_v[j],
                                       sig[num1_v[i], num2_v[j]], tse_v[t], TAUMAX,
                                       rlt121_v, 1)
                c121 += np.where(rlt121 != 0)[0].shape[0]
            t121[tse_v[t]] = c121
            for i in range(nds2):
                # print("Working on: ", i)
                rlt122 = np.zeros(nds1, dtype=np.int32)
                rlt122_v = rlt122
                for j in prange(nds1, num_threads=THREAD, nogil=True):
                # for j in range(nds1):
                    regional_sync_core(j, srs1_v[j], srs2_v[i],
                                       num1_v[j], num2_v[i],
                                       sig[num1_v[j], num2_v[i]], tse_v[t], TAUMAX,
                                       rlt122_v, 2)
                c122 += np.where(rlt122 != 0)[0].shape[0]
            t122[tse_v[t]] = c122

            c211 = 0
            c212 = 0
            for i in range(nds2):
                rlt211 = np.zeros(nds1, dtype=np.int32)
                rlt211_v = rlt211
                for j in prange(nds1, num_threads=THREAD, nogil=True):
                    regional_sync_core(j, srs2_v[i], srs1_v[j],
                                       num2_v[i], num1_v[j],
                                       sig[num2_v[i], num1_v[j]], tse_v[t], TAUMAX,
                                       rlt211_v, 1)
                c211 += np.where(rlt211 != 0)[0].shape[0]
            t211[tse_v[t]] = c211
            for i in range(nds1):
                rlt212 = np.zeros(nds2, dtype=np.int32)
                rlt212_v = rlt212
                for j in prange(nds2, num_threads=THREAD, nogil=True):
                    regional_sync_core(j, srs2_v[j], srs1_v[i],
                                       num2_v[j], num1_v[i],
                                       sig[num2_v[j], num1_v[i]], tse_v[t], TAUMAX,
                                       rlt212_v, 2)
                c212 += np.where(rlt212 != 0)[0].shape[0]
            t212[tse_v[t]] = c212
        print('Regional sync done!...............................')
        np.save(oup_reg_sync, np.array([t121, t122, t211, t212], dtype=np.int32))


@cy.boundscheck(False)
@cy.wraparound(False)
cpdef inline void regional_sync_core(Py_ssize_t p_lj,
cnp.int32_t[:] p_eli, cnp.int32_t[:] p_elj,
cnp.uint16_t p_nli, cnp.uint16_t p_nlj,
cnp.float32_t p_sig, cnp.uint16_t p_t, cnp.int8_t p_tmx,
cnp.int32_t[:] p_rlt, Py_ssize_t p_flag) nogil:

    cdef:
        cnp.uint16_t es=0
        Py_ssize_t i=0, j=0
        cnp.int32_t dly=0
        cnp.float64_t tij=0

    es=0
    for i in range(1, p_nli - 1):
        for j in range(1, p_nlj - 1):
            dly = p_elj[j] - p_eli[i]
            if dly < 0:
                continue
            if dly > p_tmx:
                break
            tij = min(min(p_eli[i] - p_eli[i - 1],
                            p_eli[i + 1] - p_eli[i]),
                        min(p_elj[j] - p_elj[j - 1],
                            p_elj[j + 1] - p_elj[j])) / 2
            if dly < tij and dly <= p_tmx:
                es += 1
    if es > p_sig:
        if p_flag == 1:
            for i in range(1, p_nli - 1):
                if p_eli[i] == p_t:
                    for j in range(1, p_nlj - 1):
                        dly = p_elj[j] - p_eli[i]
                        if dly < 0:
                            continue
                        if dly > p_tmx:
                            break
                        tij = min(min(p_eli[i] - p_eli[i - 1],
                                    p_eli[i + 1] - p_eli[i]),
                                min(p_elj[j] - p_elj[j - 1],
                                    p_elj[j + 1] - p_elj[j])) / 2
                        if dly < tij and dly <= p_tmx:
                            p_rlt[p_lj] = 1
                            break
                else:
                    continue
        if p_flag == 2:
            for j in range(1, p_nlj - 1):
                if p_elj[j] == p_t:
                    for i in range(1, p_nli - 1):
                        dly = p_elj[j] - p_eli[i]
                        if dly < 0:
                            break
                        if dly > p_tmx:
                            continue
                        tij = min(min(p_eli[i] - p_eli[i - 1],
                                    p_eli[i + 1] - p_eli[i]),
                                min(p_elj[j] - p_elj[j - 1],
                                    p_elj[j + 1] - p_elj[j])) / 2
                        if dly < tij and dly <= p_tmx:
                            p_rlt[p_lj] = 1
                            break
                else:
                    continue