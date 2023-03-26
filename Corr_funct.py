#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:40:33 2023

@author: ezequiel
"""

# ------------------------------------------------------------------------------
# Program:
# -------
#
# This program computes the cross-correlation function between UHECRs
# and a galaxy sample.
#
# ------------------------------------------------------------------------------


host = 'local'

if host == 'local':
    #root_home = '/home/ezequiel/'                                  # Local
    root_home = '/Users/ezequielboero/'                            # PC Local path
    fileProj  = root_home+'Projects/AGN_Auger/'                    # Folder of work for the Article
    data      = fileProj+'Auger_data_I/'                           # Events and Fluxes of Cosmic Rays
    graficos  = fileProj+'graphics/'                               # All the graphics

elif host == 'IATE':
    root_home = '/home/zboero/'                                    # Clemente y IATE
    fileProj  = root_home+'Projects/CMB/'                          # Folder of work for the article
    data      = fileProj+'data/'                                   # Folder with the data
    graficos  = fileProj+'graphics/'                               # Folder with the plots
    #


data_Events_4_8  = data+'events_4-8.dat'                           # Auger Event with energies: 4Eev < En < 8Eev
data_Events_a8   = data+'events_a8.dat'                            # Auger Event with energies: En > 8Eev
data_Flux_a8     = data+'flux_a8.dat'                              # Auger Flux for energies with En > 8Eev

data_2MRS        = fileProj+'2MRS_data/2mrs_1175_done.dat'         # 2MRS catalog
data_2MRS_lowM   = fileProj+'2MRS_data/2MRS_debil23_5.txt'         # 2MRS catalog of low luminosities Mabs > - 23.5

data_LVS         = fileProj+'data/VLS/VLS.txt'
data_LVS_SF      = fileProj+'data/VLS/VLS_SF.txt'
data_LVS_Passive = fileProj+'data/VLS/VLS_Passive.txt'
data_LVS_Faint   = fileProj+'data/VLS/VLS_Faint.txt'
data_LVS_Bright  = fileProj+'data/VLS/VLS_Bright.txt'

####################################################################################
####################################################################################

import numpy  as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt

import random


# Load the data files
cols_Auger = ['year', 'day', 'dec', 'RA', 'azimuth', 'weight']
df_Auger = pd.read_table(data_Events_a8, skiprows=33, names= cols_Auger, sep="\s+", index_col=False)

#df_gxs   = pd.read_table(data_2MRS, skiprows=10,\
#                         names=['RAdeg', 'DECdeg', 'l', 'b', 'k_c', 'h_c', 'j_c', 'k_tc', 'h_tc', 'j_tc',\
#                               'e_k', 'e_h', 'e_j', 'e_kt', 'e_ht', 'e_jt', 'e_bv', 'r_iso', 'r_ext',\
#                               'b/a', 'flgs', 'type', 'ts', 'v', 'e_v', 'c'], sep="\s+",\
#                         index_col=False)

#cols_gxs = ['ID', 'RAdeg', 'DECdeg', 'cz', 'Ktmag_abs', 'l_deg', 'b_deg']
#df_gxs = pd.read_table(data_2MRS_lowM, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)
cols_gxs = [ 'RAdeg', 'DECdeg', 'Kcmag', 'Hcmag', 'Jcmag', 'Ktmag', 'K_abs', 'type', 'cz',
             'JNAME', 'W1mag', 'W2mag', 'W3mag', 'class(1AGN,2SF,3Passive)']
df_LVS         = pd.read_table(data_LVS, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)
df_LVS_SF      = pd.read_table(data_LVS_SF, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)
df_LVS_Passive = pd.read_table(data_LVS_Passive, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)
df_LVS_Faint   = pd.read_table(data_LVS_Faint, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)
df_LVS_Bright  = pd.read_table(data_LVS_Bright, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)
    
#df_gxs = df_LVS_Faint
df_gxs = pd.concat([df_LVS_Faint, df_LVS_Bright]).reset_index(drop=True)

##########################################################################################################
# 0) We first compute the number of pairs (rdm_gxs/UHECRs) at a given radius
##########################################################################################################

# Define the number of galaxies and the size of the map
n_evnt = 50000
#nside  = 64

# Generate a model distribution for the incoming UHECRs
# 1) The random component
def rdm_sample(n_evnt):
    ''' Produces a uniform random sample of points in the unit sphere
        
    Parameters
    ----------
    n_events : int
        The total number of events of the map
        
    Returns
    -------
    A pandas dataframe with the events
    '''

    lon_array   = np.random.uniform( low= 0.0 , high= 2*np.pi, size= n_evnt )
    costheta    = np.random.uniform( low= -1.0, high= 1.0    , size= n_evnt )
    colat_array = np.arccos( costheta )
    lat_array   = 0.5*np.pi - colat_array       # colat_array - 0.5*np.pi

    cols_df_rdm = { 'l (rad)': lon_array,
                    'colat (rad)': colat_array,
                    'b (rad)': lat_array
                  }
    df_rdm = pd.DataFrame( data= cols_df_rdm )
    # Consider the cut in declinations > 45.0 deg
    #df_rdm = df_rdm[ df_rdm["DECdeg"] < 45.01 ]
    return df_rdm
    
df_rdm_gxs = rdm_sample( n_evnt )
df_rdm_gxs = df_rdm_gxs[ df_rdm_gxs['colat (rad)'] > np.deg2rad(45.01) ]

# Adjust the sample to have the number equal to the number of galaxies in our sample
n_gxs   = len(df_gxs)                                            # Number of galaxies
n_rdm   = n_gxs                                                  # Number of rdm event in our model
id_rdm  = df_rdm_gxs.index                                       # This variable contains all the index of the df_rdm
list_id = id_rdm.tolist()                                        # Here we transform it to a list for do iterations below...
id_pick = random.sample( list_id, n_rdm )                        # We choose the n_rdm indices of uhe_events from the df_rdm sample...
df_rdm_gxs  = df_rdm_gxs.loc[ id_pick ]                          # We choose those events from dr_rdm

# We now compute the correlation function between this map and the Auger map
# Define the radial profile function
def radial_profile( th, phi, th_gxs, phi_gxs, bins ):
    ''' Radial profile around the position of galaxies
    Parameters
    ----------
    th : numpy array
        co-latitude (0,pi) in radians for each pixel of a healpix map.
    phi : numpy array
        longitude (0,2*pi) in radians for each pixel of a healpix map.
    th_gxs : numpy array
        co-latitude (0,pi) in radians for each galaxy in the sample.
    phi_gxs : numpy array
        longitude (0,2*pi) in radians for each galaxy in the sample.
    bins : numpy array
        the bins of the radial profile

    Returns
    -------
    A numpy array with the number of events for each bin of the radial profile
    
    '''
    import numpy as np

    # cos_gamma note that the formula below corresponds to -1 <= cos(th) <= 1 convention
    # which implies that 0 <= th <= 180 in deg or 0 <= th <= pi in rad...This is th must be a co-latitude
    cos_sep = np.cos(th_gxs) * np.cos(th) + np.sin(th_gxs) * np.sin(th) * np.cos(phi - phi_gxs)
    omega = np.arccos( cos_sep )
    prof  = np.histogram( omega, bins )[0]
        
    return prof

#bins = np.deg2rad( np.arange(5,91,5) )
#th_uhe  = np.deg2rad( 90 - df_Ev8['dec'].to_numpy() )
#phi_uhe = np.deg2rad( df_Ev8['RA'].to_numpy() )
#th_rdm  = df_rdm['colat (rad)'].to_numpy()
#phi_rdm = df_rdm['l (rad)'].to_numpy()
#
#prof_rdm = np.zeros( len(bins) - 1 )
#for i in range( len(df_rdm) ):
#    prof_rdm_i = radial_profile( th_rdm[i], phi_rdm[i], th_uhe, phi_uhe, bins )
#    prof_rdm  += prof_rdm_i

cols_df_CRs = { 'colat (rad)' : th_CRs,           # colatitude in radians
                'l (rad)'     : phi_CRs           # longitude  in radians
              }
df_CRs = pd.DataFrame( data= cols_df_CRs )
df_CRs = df_CRs[ df_CRs['colat (rad)'] > np.deg2rad(45.01) ]

th_uhe_model  = df_CRs['colat (rad)'].to_numpy()
phi_uhe_model = df_CRs['l (rad)'].to_numpy()
th_rdm        = df_rdm_gxs['colat (rad)'].to_numpy()
phi_rdm       = df_rdm_gxs['l (rad)'].to_numpy()

bins = np.deg2rad( np.arange(5,91,5) )
prof_rdm = np.zeros( len(bins) - 1 )
for i in range( len(df_rdm_gxs) ):
    prof_rdm_i = radial_profile( th_rdm[i], phi_rdm[i], th_uhe_model, phi_uhe_model, bins )
    prof_rdm  += prof_rdm_i

# plot of the scatter plot
th_str, phi_str = 'Dec (deg)', 'RA (deg)'
title = 'Model of UHECRs incoming events'
output_file = graficos+'skymap_model.png'
sky_plot( th_uhe_model, phi_uhe_model, th_str, phi_str, coord_sys, title, output_file )

##########################################################################################################
# 1) We now compute the number of pairs (gxs/UHECRs) at a given radius
##########################################################################################################

#bins = np.deg2rad( np.arange(5,91,5) )
th_gxs  = np.deg2rad( 90.0 - df_gxs['DECdeg'].to_numpy() )
phi_gxs = np.deg2rad( df_gxs['RAdeg'].to_numpy() )

prof_gxs = np.zeros( len(bins) - 1 )
for i in range( len(df_gxs) ):
    prof_gxs_i = radial_profile( th_gxs[i], phi_gxs[i], th_uhe_model, phi_uhe_model, bins )
    prof_gxs += prof_gxs_i


##########################################################################################################
# 2) We now compute the cross-correlation function between (gxs/UHERCs)
##########################################################################################################
Corr_func = prof_gxs / prof_rdm - 1.0


# We plot the correlation function
output_file = graficos+'xCross_B+F_vs_model.png'
plt.figure()
#plt.fill_between( x_fx[0] * rad2deg, x_fx[1] - sigma, x_fx[1] + sigma, color='darkcyan', alpha=0.4, linestyle='solid' )
plt.plot( np.rad2deg( bins[1:] ), Corr_func, color='darkcyan', label='Cross-Corr B+F', linestyle='solid' )
#
#plt.title ( 'Correlation function', loc='center', fontsize='x-large')
plt.xlabel( 'Angle[deg]', fontsize='x-large')
plt.ylabel( 'Amplitude', fontsize='x-large')
plt.axes
#plt.ylim(-0.01,0.25)
plt.legend(loc='upper right', fontsize='large', markerscale=3.0)
plt.savefig(output_file)
plt.tight_layout()
plt.close()



