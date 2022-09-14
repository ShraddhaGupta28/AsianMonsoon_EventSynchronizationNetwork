# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
# distutils: language = c
# cython: c_string_type=unicode, c_string_encoding=utf8
import os
import xarray as xr
import numpy as np
cimport numpy as cnp
cimport cython as cy
from cython.parallel import prange
from time import time, sleep


cdef str DATASETS = 'TRMM_Precipitation'
cdef str PERIOD = '1998_To_2019ASM.nc4'
cdef str SEASON = 'JJA'
cdef cnp.float64_t PERCENTILE = 90.  # from 80% to 99%
cdef cnp.int8_t TAUMAX = 7  # the maximum temporal delay
cdef cnp.float64_t SIGNIFICANCE = 95.  # the significance level, 99.5%
cdef cnp.int8_t SPAZOOMOUT = 1  # zoom out in spatial

cdef cnp.int8_t CPU = 16
cdef cnp.int8_t THREAD = CPU

cdef str inp_evt = './Results/%s/%s_%s_%s.zarr' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1])
cdef str oup_sig = './Results/%s/ESNM_%s_%s_%s_d%s' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX))


@cy.boundscheck(False)
@cy.wraparound(False)
cpdef event_sync_null_model():
    cdef:
        str oup_sig_P950, oup_sig_P995, oup_sig_P999
        cnp.ndarray[cnp.int32_t, ndim=1] tim
        cnp.ndarray[cnp.int32_t, ndim=2] eli, elj
        cnp.int32_t[:, :] eli_v, elj_v
        cnp.ndarray[cnp.float32_t, ndim=2] P95, P995, P999, P996, P997, P998
        cnp.ndarray[cnp.float32_t, ndim=2] P990, P991, P992, P993, P994, P985, P980
        cnp.ndarray[cnp.float32_t, ndim=2] P986, P987, P988, P989
        cnp.ndarray[cnp.uint16_t, ndim=1] rlt
        cnp.uint16_t[:] rlt_v
        Py_ssize_t idl_max=0, i=0, j=0, r=0


    oup_sig_P950 = oup_sig + '_P950.npy'
    oup_sig_P995 = oup_sig + '_P995.npy'
    oup_sig_P999 = oup_sig + '_P999.npy'
    # for complementary
    oup_sig_P996 = oup_sig + '_P996.npy'
    oup_sig_P997 = oup_sig + '_P997.npy'
    oup_sig_P998 = oup_sig + '_P998.npy'
    oup_sig_P990 = oup_sig + '_P990.npy'
    oup_sig_P991 = oup_sig + '_P991.npy'
    oup_sig_P992 = oup_sig + '_P992.npy'
    oup_sig_P993 = oup_sig + '_P993.npy'
    oup_sig_P994 = oup_sig + '_P994.npy'
    oup_sig_P985 = oup_sig + '_P985.npy'
    oup_sig_P980 = oup_sig + '_P980.npy'
    oup_sig_P986 = oup_sig + '_P986.npy'
    oup_sig_P987 = oup_sig + '_P987.npy'
    oup_sig_P988 = oup_sig + '_P988.npy'
    oup_sig_P989 = oup_sig + '_P989.npy'

    print(inp_evt)
    with xr.open_zarr(inp_evt, consolidated=True) as dta:
        tim = dta['tse'].values.astype(np.int32)

        # ideally maximum events
        idl_max = dta['evt_srs'].shape[2]
        P95 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P995 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P999 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        # for complementary
        P996 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P997 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P998 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P990 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P991 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P992 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P993 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P994 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P985 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P980 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P986 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P987 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P988 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)
        P989 = np.zeros((idl_max + 1, idl_max + 1), dtype=np.float32)

        rlt = np.zeros(2000, dtype=np.uint16)
        rlt_v = rlt
        # at least 3 events
        for i in range(3, idl_max + 1):
            print("Starting: ", i)
            for j in range(3, i + 1):
                eli = np.array([np.sort(np.random.choice(tim, size=i, replace=False), kind='mergesort') for r in range(2000)])
                eli_v = eli
                elj = np.array([np.sort(np.random.choice(tim, size=j, replace=False), kind='mergesort') for r in range(2000)])
                elj_v = elj
                for r in prange(2000, num_threads=THREAD, nogil=True):
                    es_null_model_core(eli_v[r], elj_v[r], TAUMAX, rlt_v, r)
                P95[i, j], P995[i, j], P999[i, j] = np.float32(np.percentile(rlt, 95)), np.float32(np.percentile(rlt, 99.5)), np.float32(np.percentile(rlt, 99.9))
                P95[j, i], P995[j, i], P999[j, i] = P95[i, j], P995[i, j], P999[i, j]
                # for complementary
                P996[i, j], P997[i, j], P998[i, j] = np.float32(np.percentile(rlt, 99.6)), np.float32(np.percentile(rlt, 99.7)), np.float32(np.percentile(rlt, 99.8))
                P996[j, i], P997[j, i], P998[j, i] = P996[i, j], P997[i, j], P998[i, j]
                P990[i, j] = np.float32(np.percentile(rlt, 99.0))
                P991[i, j] = np.float32(np.percentile(rlt, 99.1))
                P992[i, j] = np.float32(np.percentile(rlt, 99.2))
                P993[i, j] = np.float32(np.percentile(rlt, 99.3))
                P994[i, j] = np.float32(np.percentile(rlt, 99.4))
                P985[i, j] = np.float32(np.percentile(rlt, 98.5))
                P980[i, j] = np.float32(np.percentile(rlt, 98.0))
                P990[j, i] = P990[i, j]
                P991[j, i] = P991[i, j]
                P992[j, i] = P992[i, j]
                P993[j, i] = P993[i, j]
                P994[j, i] = P994[i, j]
                P985[j, i] = P985[i, j]
                P980[j, i] = P980[i, j]
                P986[i, j] = np.float32(np.percentile(rlt, 98.6))
                P987[i, j] = np.float32(np.percentile(rlt, 98.7))
                P988[i, j] = np.float32(np.percentile(rlt, 98.8))
                P989[i, j] = np.float32(np.percentile(rlt, 98.9))
                P986[j, i] = P986[i, j]
                P987[j, i] = P987[i, j]
                P988[j, i] = P988[i, j]
                P989[j, i] = P989[i, j]

        # output files
        np.save(oup_sig_P950, P95)
        np.save(oup_sig_P995, P995)
        np.save(oup_sig_P999, P999)
        # for complementary
        np.save(oup_sig_P996, P996)
        np.save(oup_sig_P997, P997)
        np.save(oup_sig_P998, P998)
        np.save(oup_sig_P990, P990)
        np.save(oup_sig_P991, P991)
        np.save(oup_sig_P992, P992)
        np.save(oup_sig_P993, P993)
        np.save(oup_sig_P994, P994)
        np.save(oup_sig_P985, P985)
        np.save(oup_sig_P980, P980)
        np.save(oup_sig_P986, P986)
        np.save(oup_sig_P987, P987)
        np.save(oup_sig_P988, P988)
        np.save(oup_sig_P989, P989)   


@cy.boundscheck(False)
@cy.wraparound(False)
cpdef inline void es_null_model_core(cnp.int32_t[:] p_eli, cnp.int32_t[:] p_elj, cnp.int8_t p_tmx, cnp.uint16_t[:] rlt, Py_ssize_t r) nogil:
    """
    P1: event series at locations i,
    P2: at j,
    P3: maximum temporal delay,
    output core
    """

    cdef:
        cnp.uint16_t es = 0
        Py_ssize_t i_dim, j_dim, i, j
        cnp.int32_t dly
        cnp.float64_t tij

    i_dim, j_dim = p_eli.shape[0], p_elj.shape[0]
    for i in range(1, i_dim - 1):
        for j in range(1, j_dim - 1):
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
    # check again
    if i_dim < 3 or j_dim < 3:
        es = 0
    rlt[r] = es
