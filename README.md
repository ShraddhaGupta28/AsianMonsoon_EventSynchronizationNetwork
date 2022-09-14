# AsianMonsoon_EventSynchronizationNetwork

The code is based on the paper:  
Gupta, S., Su, Z., Boers, N., Kurths, J., Marwan, N. and Pappenberger, F. (2022), Interconnection between the Indian and the East Asian Summer Monsoon: spatial synchronization patterns of extreme rainfall events. Int J Climatol. Accepted Author Manuscript. https://doi.org/10.1002/joc.7861

All coding files are implemented using Python 3.9:  
(Packages including: cython 0.29.32 + xarray 2022.6.0 + zarr 2.12.0 + netcdf4 1.6.0 + basemap 1.2.2 + cmocean 2.0):
1. Data pre-processing  
    Extreme.py  
    Extreme_Under_Box.py  

2. Network reconstruction  
    Event_Sync_Null_Model_Cy.pyx  
    Event_Sync2_Null_Model_Cy.pyx  
    Event_Sync_Udw_Cy.pyx  
    Task1_Ud_ES_Construction.pyx  

3. Specific times of high extreme rainfall synchronicity  
    Task1_Ud_ES_Regional_Sync_Corr.pyx  
    ISM_EASM_IJC_Task1_Ud_ES_Reg_Sync_Corr.py  

4. Composite anomalies  
    ISM_EASM_IJC_Task1_Ud_ES_Clim_ERA5_GPH_CAno_Mon.py  
    ISM_EASM_IJC_Task1_Ud_ES_Clim_ERA5_OLR_CAno_Mon.py  
    ISM_EASM_IJC_Task1_Ud_ES_Clim_ERA5_W_CAno_Mon.py  
    ISM_EASM_IJC_Task1_Ud_ES_Clim_ERA5_WVF_CAno_Mon.py  
    ISM_EASM_IJC_Task1_Ud_ES_Clim_TRMM_R_CAno.py  

5. Visualization in the main text  
    Fig.1: ~ISM_EASM_IJC_Task1_Visual_ES_Reg_TPDeg_ASM_And_Mon.py  
    Fig.2: ~ISM_EASM_IJC_Task1_Visual_Ud_ES_Reg_Sync_Corr_M.py  
    Fig.3: ~ISM_EASM_IJC_Task1_Visual_Ud_ES_Clim_GPH_CAnoSN_MonRF.py  
    Fig.4: ~ISM_EASM_IJC_Task1_Visual_Ud_ES_Clim_WVC_CAnoSN_Mon.py  
    Fig.5: ~ISM_EASM_IJC_Task1_Visual_Ud_ES_Clim_WVF_CAnoSN.py  
    Fig.6: ~ISM_EASM_IJC_Task1_Visual_Ud_ES_Clim_OLR_CAnoSN_Single.py  
    Fig.7: ~ISM_EASM_IJC_Task1_Visual_Ud_ES_Reg_Sync_MJO_BSISO.py  

For the original version of code on climate network reconstruction, please refer to [1]:  
https://github.com/niklasboers/rainfall-teleconnections.git

[1] Boers, N., Goswami, B., Rheinwalt, A., Bookhagen, B., Hoskins, B., & Kurths, J. (2019). Complex networks reveal global pattern of extreme-rainfall teleconnections. Nature, 566(7744), 373-377.