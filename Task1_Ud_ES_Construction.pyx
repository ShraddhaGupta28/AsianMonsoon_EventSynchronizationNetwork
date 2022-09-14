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
from libc.stdio cimport printf
from libc.math cimport acos
from time import time, sleep
import gc


cdef str DATASETS = 'TRMM_Precipitation'
cdef str PERIOD = '1998_To_2019ASM.nc4'
cdef str SEASON = 'JJA'
cdef cnp.float64_t PERCENTILE = 90.  # from 80% to 99%
cdef cnp.int8_t TAUMAX = 7  # the maximum temporal delay
cdef cnp.float64_t SIGNIFICANCE = 95.  # the significance level, 99.5%
cdef cnp.int8_t SPAZOOMOUT = 1  # zoom out in spatial

cdef cnp.int8_t CPU = 5
cdef cnp.int8_t THREAD = 5

cdef str inp_evt
inp_evt = './Results/%s/%s_%s_%s.zarr' % (DATASETS, PERIOD.split(
    '.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1])
if SPAZOOMOUT != 1:
    inp_evt = './Results/%s/%s_%s_%s_X%s.zarr' % (DATASETS, PERIOD.split('.')[0], SEASON, str(
        PERCENTILE).split('.')[0] + str(PERCENTILE).split('.')[1], str(SPAZOOMOUT))

cdef str inp_net
inp_net = './Results/%s/ES_%s_%s_%s_d%s_P%s_Udw' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split(
    '.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])
if SPAZOOMOUT != 1:
    inp_net = './Results/%s/ES_%s_%s_%s_d%s_P%s_X%s_Udw' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(
        PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(SPAZOOMOUT))

cdef str oup_deg
oup_deg = './Results/%s/ES_%s_%s_%s_d%s_P%s_Deg.npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split(
    '.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])
if SPAZOOMOUT != 1:
    oup_deg = './Results/%s/ES_%s_%s_%s_d%s_P%s_X%s_Deg.npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(
        PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(SPAZOOMOUT))

cdef str oup_cudeg
oup_cudeg = './Results/%s/ES_%s_%s_%s_d%s_P%s_CuDeg.npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split(
    '.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])
if SPAZOOMOUT != 1:
    oup_cudeg = './Results/%s/ES_%s_%s_%s_d%s_P%s_X%s_CuDeg.npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(
        PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(SPAZOOMOUT))

cdef str oup_adj
oup_adj = './Results/%s/ES_%s_%s_%s_d%s_P%s_Adj.npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split(
    '.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])
if SPAZOOMOUT != 1:
    oup_adj = './Results/%s/ES_%s_%s_%s_d%s_P%s_X%s_Adj.npy' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(
        PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(SPAZOOMOUT))

cdef str oup_graph
oup_graph = './Results/%s/ES_%s_%s_%s_d%s_P%s_Gph_Ud.graph' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split(
    '.')[0] + str(PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1])
if SPAZOOMOUT != 1:
    oup_graph = './Results/%s/ES_%s_%s_%s_d%s_P%s_X%s_Gph_Ud.graph' % (DATASETS, PERIOD.split('.')[0], SEASON, str(PERCENTILE).split('.')[0] + str(
        PERCENTILE).split('.')[1], str(TAUMAX), str(SIGNIFICANCE).split('.')[0] + str(SIGNIFICANCE).split('.')[1], str(SPAZOOMOUT))


@cy.boundscheck(False)
@cy.wraparound(False)
cpdef network():
    cdef: 
        cnp.ndarray[cnp.float32_t, ndim=1] lat, lon
        cnp.ndarray[cnp.int32_t, ndim=2] edg
        cnp.ndarray[cnp.int32_t, ndim=1] tail, deg, unq, cudeg, shift
        cnp.ndarray[cnp.uint32_t, ndim=1] adj, adj_i
        cnp.ndarray[cnp.int64_t, ndim=1] cnt
        cnp.int32_t c=0, nd1=0, nd2=0
        Py_ssize_t la=0, lo=0, nodes=0, i=0, edg_num=0, e=0
        list fil_adj
        str f

    with xr.open_zarr(inp_evt, consolidated=True) as dta:
        lat = dta['lat'].values.astype(np.float32)
        lon = dta['lon'].values.astype(np.float32)
        la, lo = lat.shape[0], lon.shape[0]
        nodes = la * lo

    # load edges
    deg = np.zeros(nodes, dtype=np.int32)
    fil_adj = [f for f in os.listdir('./Results/%s/' % (DATASETS)) if inp_net.split('/')[len(inp_net.split('/')) - 1] in f]
    fil_adj = sorted(fil_adj)
    for i, f in enumerate(fil_adj):
        edg = np.load('./Results/%s/%s' % (DATASETS, f))[:, :2].astype(np.int32)
        if i == (len(fil_adj) - 1):
            tail = np.where(edg[:, 0] == np.max(edg[:, 0]))[0].astype(np.int32)
            edg = edg[:(np.max(tail) + 1), :]
            del tail
        edg -= 1
        edg_num += edg.shape[0]
        unq, cnt = np.unique(edg.flatten(), return_counts=True)
        deg[unq] += cnt.astype(np.int32)
        del edg, unq, cnt

    # degree
    np.save(oup_deg, deg)
    print("degree done")

    # cumulative degree: cudeg[0] = 0, [1] = deg1, [2] = deg12
    cudeg = np.zeros(nodes + 1, dtype=np.int32)
    c = 0
    for i in range(nodes + 1):
        cudeg[i] = c
        c += deg[i]
    np.save(oup_cudeg, cudeg)
    print("cumulative degree done")

    # adjacency list
    shift = cudeg.copy()
    adj = np.zeros(np.sum(deg), dtype=np.uint32)
    for i, f in enumerate(fil_adj):
        edg = np.load('./Results/%s/%s' % (DATASETS, f))[:, :2].astype(np.int32)
        if i == (len(fil_adj) - 1):
            tail = np.where(edg[:, 0] == np.max(edg[:, 0]))[0].astype(np.int32)
            edg = edg[:(np.max(tail) + 1), :]
            del tail
        edg -= 1
        for e in range(edg.shape[0]):
            nd1, nd2 = edg[e][0], edg[e][1]
            adj[shift[nd1]] = nd2
            adj[shift[nd2]] = nd1
            shift[nd1] += 1
            shift[nd2] += 1
        del edg
    np.save(oup_adj, adj)
    del shift
    gc.collect()
    print("adjacency list done")

    # to .graph file, isolated nodes to be self-loop
    # edg_num = edg_num + np.sum(deg == 0)
    # with open(oup_graph, 'w', newline='') as gph:
    #     gph.write(str(nodes) + ' ' + str(edg_num) + '\n')
    # with open(oup_graph, 'a', newline='') as gph:
    #     for i in range(nodes):
    #         if deg[i] != 0:
    #             adj_i = adj[cudeg[i]:cudeg[i + 1]] + 1
    #         else:
    #             adj_i = np.array([i], dtype=np.uint32) + 1
    #         np.savetxt(gph, adj_i, fmt='%d', newline=' ')
    #         gph.write('\n')
    # print(".graph file output done")