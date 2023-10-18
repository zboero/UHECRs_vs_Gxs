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

data_2MRSxWISE_VLS              = fileProj+'data/VLS/2MRSxWISE_VLS.txt'               # Ultimas muestras 18/10/2023
data_2MRSxWISE_VLS_passivecrop  = fileProj+'data/VLS/2MRSxWISE_VLS_passivecrop.txt'   # Ultimas muestras 18/10/2023
data_2MRSxWISE_VLS_d1d5         = fileProj+'data/VLS/2MRSxWISE_VLS_d1d5.txt'
data_2MRSVLS_passive_cropd5     = fileProj+'data/VLS/2MRSVLS_passive_cropd5.txt'

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

df_2MRS   = pd.read_table(data_2MRS, skiprows=10,\
                         names=['RAdeg', 'DECdeg', 'l', 'b', 'k_c', 'h_c', 'j_c', 'k_tc', 'h_tc', 'j_tc',\
                               'e_k', 'e_h', 'e_j', 'e_kt', 'e_ht', 'e_jt', 'e_bv', 'r_iso', 'r_ext',\
                               'b/a', 'flgs', 'type', 'ts', 'v', 'e_v', 'c'], sep="\s+",\
                         index_col=False)

cols_gxs = [ 'RAdeg', 'DECdeg', 'Kcmag', 'Hcmag', 'Jcmag', 'Ktmag', 'K_abs', 'type', 'cz',
             'JNAME', 'W1mag', 'W2mag', 'W3mag', 'class(1AGN,2SF,3Passive)']
df_LVS         = pd.read_table(data_LVS, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)
df_LVS_SF      = pd.read_table(data_LVS_SF, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)
df_LVS_Passive = pd.read_table(data_LVS_Passive, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)
df_LVS_Faint   = pd.read_table(data_LVS_Faint, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)
df_LVS_Bright  = pd.read_table(data_LVS_Bright, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)

df_2MRSxWISE_VLS              = pd.read_table(data_2MRSxWISE_VLS, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)
df_2MRSxWISE_VLS_passivecrop  = pd.read_table(data_2MRSxWISE_VLS_passivecrop, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)
df_2MRSxWISE_VLS_d1d5         = pd.read_table(data_2MRSxWISE_VLS_d1d5, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)
df_2MRSVLS_passive_cropd5     = pd.read_table(data_2MRSVLS_passive_cropd5, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)

#df_gxs = df_2MRS
#df_gxs = df_LVS_SF
#df_gxs = df_LVS_Passive
#df_gxs = df_LVS_Faint
#df_gxs = df_LVS_Bright
df_gxs = df_2MRSxWISE_VLS
#df_gxs = df_2MRSxWISE_VLS_passivecrop
#df_gxs = df_2MRSxWISE_VLS_d1d5
#df_gxs = df_2MRSVLS_passive_cropd5

#df_gxs = pd.concat([df_LVS_Faint, df_LVS_Bright]).reset_index(drop=True)

# Filters...
#df_gxs = df_gxs[ df_gxs['class(1AGN,2SF,3Passive)'] == 3]
df_gxs = df_gxs[ df_gxs['class(1AGN,2SF,3Passive)'] == 3]

##########################################################################################################
# 0) We first compute the number of pairs (rdm_gxs/UHECRs) at a given radius
##########################################################################################################

# Define the number of galaxies and the size of the map
n_evnt = 80000

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
df_rdm_gxs = df_rdm_gxs[ df_rdm_gxs['colat (rad)'] > np.deg2rad(45.0001) ]     # Aplicamos el corte en colatitud de Auger

# Adjust the sample to have the number equal to the number of galaxies in our sample
n_gxs   = len(df_gxs)                                            # Number of galaxies
n_rdm   = n_gxs                                                  # Number of rdm event in our model
id_rdm  = df_rdm_gxs.index                                       # This variable contains all the index of the df_rdm
list_id = id_rdm.tolist()                                        # Here we transform it to a list for do iterations below...
id_pick = random.sample( list_id, n_rdm )                        # We choose the n_rdm indices of uhe_events from the df_rdm sample...
df_rdm_gxs  = df_rdm_gxs.loc[ id_pick ]                          # We choose those events from dr_rdm

# plot of the rdm sample
def sky_plot( th, phi, th_str, phi_str, coord_sys, title, output_file ):
    ''' Produces a scatter map in Mollweide projection with the position
    of events in the sky.
        
    Parameters
    ----------
    th, phi : numpy arrays
        The angular coordinates on the sphere of the events: co-latitude and longitude
        Units must be in radians.
    th_str, phi_str : strings
        The labels of the coordinates (for example, 'RA', 'Dec' or 'lon', 'lat')
    coord_sys : str
        The coordinate system under use: 'Galactic', 'Equatorial', etc
    title : str
        The title of the plot
    output_file : str
        The output file (.png in general)
        
    Returns
    -------
    The figure with the scatter plot in Mollweide projection.
    '''

    org =0                                                 # Origin of the map
    projection ='mollweide'                                # 2D projection
    x = np.remainder( phi + 2.0*np.pi - org, 2.0*np.pi)    # shift phi (RA) values
    ind = x > np.pi
    x[ind] -= 2.0*np.pi                                    # scale conversion to [-180, 180]
    x = -x                                                 # reverse the scale: East to the left
    y = 0.5*np.pi - th                                     # From colatitude to latitude
    
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder( tick_labels + 360 + org, 360 )
    
    fig = plt.figure( figsize=(10, 5) )
    ax = fig.add_subplot( 111, projection=projection )
    ax.scatter( x, y, s=1.5 )
    ax.set_xticklabels( tick_labels )                      # we add the scale on the x axis
    ax.set_title( title )
    ax.title.set_fontsize(15)
    ax.set_xlabel( phi_str )
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel( th_str )
    ax.yaxis.label.set_fontsize(12)
    ax.grid('True', linestyle='--')
    ax.text( 5.5, -1.3 , coord_sys , fontsize=12)
    plt.savefig( output_file )
    #plt.savefig(graficos+'vMF_dist.png')

# plot: we plot the random sample that we built...
th_rdm        = df_rdm_gxs['colat (rad)'].to_numpy()
phi_rdm       = df_rdm_gxs['l (rad)'].to_numpy()
th_str, phi_str = 'Dec (deg)', 'RA (deg)'
coord_sys = 'Equatorial'
title = 'Rdm'
output_file = graficos+'skymap_rdm.png'
sky_plot( th_rdm, phi_rdm, th_str, phi_str, coord_sys, title, output_file )


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


def xCorr( th_rdm, phi_rdm, th_gxs, phi_gxs, th_uhe_model, phi_uhe_model, bins ):
    ''' Radial profile around the position of galaxies
    Parameters
    ----------
    th_rdm : numpy array
        co-latitude (0,pi) in radians for each point of the rdm sample.
    phi_rdm : numpy array
        longitude (0,2*pi) in radians for each point of the rdm sample.
    th_gxs : numpy array
        co-latitude (0,pi) in radians for each galaxy in the sample.
    phi_gxs : numpy array
        longitude (0,2*pi) in radians for each galaxy in the sample.
    th_uhe_model : numpy array
        co-latitude (0,pi) in radians for each galaxy in the 2nd sample.
    phi_uhe_model : numpy array
        longitude (0,2*pi) in radians for each galaxy in the 2nd sample.
    bins : numpy array
        the bins of the radial profile

    Returns
    -------
    A numpy array with the xCorrelation function
    
    '''
    prof_rdm = np.zeros( len(bins) - 1 )
    for i in range( len(th_rdm) ):
        prof_rdm_i = radial_profile( th_rdm[i], phi_rdm[i], th_uhe_model, phi_uhe_model, bins )
        prof_rdm  += prof_rdm_i

    prof_gxs = np.zeros( len(bins) - 1 )
    for i in range( len(th_gxs) ):
        prof_gxs_i = radial_profile( th_gxs[i], phi_gxs[i], th_uhe_model, phi_uhe_model, bins )
        prof_gxs += prof_gxs_i

    Corr_func = prof_gxs / prof_rdm - 1.0
    
    return Corr_func

##########################################################################################################
# 1) We now compute the number of pairs (gxs/UHECRs) at a given radius
##########################################################################################################

# We compute the cross-correlation function
th_rdm        = df_rdm_gxs['colat (rad)'].to_numpy()
phi_rdm       = df_rdm_gxs['l (rad)'].to_numpy()
th_gxs        = np.deg2rad( 90.0 - df_gxs['DECdeg'].to_numpy() )
phi_gxs       = np.deg2rad( df_gxs['RAdeg'].to_numpy() )
th_uhe_model  = th_uhe #th_CRs                # Si use Augerdata --> th_uhe si use UHECRs_model --> th_CRs
phi_uhe_model = phi_uhe #phi_CRs              # Si use Augerdata --> phi_uhe si use UHECRs_model --> phi_CRs
#th_uhe_model  = np.deg2rad( 90.0 - df_Auger['DECdeg'].to_numpy() )
#phi_uhe_model = np.deg2rad( df_Auger['RAdeg'].to_numpy() )

bins = np.deg2rad( np.arange(5,91,5) )
xCorr_measured = xCorr(th_rdm, phi_rdm, th_gxs, phi_gxs, th_uhe_model, phi_uhe_model, bins)

# We compute the bootstrap error..
n_boots   = 20
idx_boots = np.random.choice( len(th_gxs), size=( n_boots, len(th_gxs) ), replace=True )
xCorr_list_boots = []#np.array([])
for i in range( 0, n_boots ):
    th_gxs_i  = th_gxs[ idx_boots[i] ]
    phi_gxs_i = phi_gxs[ idx_boots[i] ]
    xCorr_i = xCorr(th_rdm, phi_rdm, th_gxs_i, phi_gxs_i, th_uhe_model, phi_uhe_model, bins)
    xCorr_list_boots.append( xCorr_i )
    
percentiles = np.percentile( xCorr_list_boots, np.array( [15.87,84.13] ), axis=0 )
err_low_boot  = percentiles[0]
err_high_boot = percentiles[1]
err = 0.5 * ( err_high_boot - err_low_boot )


# We plot the correlation function
output_file = graficos+'xCross_SF+Passive_vs_Auger.png'#model.png'
xCorr_SF  = xCorr_measured
xCorr_Passive  = xCorr_measured
plt.figure()
#plt.fill_between( np.rad2deg( bins[1:] ), xCorr - err, xCorr + err, color='darkcyan', alpha=0.4, linestyle='solid' )
plt.plot( np.rad2deg( bins[1:] ), xCorr_SF, color='darkcyan', label='Cross-Corr SF', linestyle='solid' )
plt.plot( np.rad2deg( bins[1:] ), xCorr_Passive, color='magenta', label='Cross-Corr Passive', linestyle='solid' )
#
#plt.title ( 'Correlation function', loc='center', fontsize='x-large')
plt.xlabel( 'Angle[deg]', fontsize='x-large')
plt.ylabel( 'Amplitude', fontsize='x-large')
plt.axes
plt.ylim(-0.01,0.25)
plt.legend(loc='upper right', fontsize='large', markerscale=3.0)
plt.savefig(output_file)
plt.tight_layout()
plt.close()



