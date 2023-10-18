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
# This program produces a Healpix maps to study the correlation of UHECRs
# with the distribution of nearby galaxies (d <= 50Mpc) in the local Universe.
# The incoming direction of UHECRs events in the sky are modeled here
# by means of a distribution computed from the galaxy positions.
#
# This maps are correlated with the position of galaxies itself and with the
# observed distribution of UHECRs by the Pierre Auger Observatory.
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


raw_data         = data+'dataSummary.csv'                          # All the events released by Auger
data_Events_4_8  = data+'events_4-8.dat'                           # Auger Event with energies: 4Eev < En < 8Eev
data_Events_a8   = data+'events_a8.dat'                            # Auger Event with energies: En > 8Eev
data_Flux_a8     = data+'flux_a8.dat'                              # Auger Flux for energies with En > 8Eev

data_2MRS        = fileProj+'2MRS_data/2mrs_1175_done.dat'         # 2MRS catalog
data_2MRS_lowM   = fileProj+'2MRS_data/2MRS_debil23_5.txt'         # 2MRS catalog of low luminosities Mabs > - 23.5

####################################################################################
####################################################################################

import numpy  as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
import csv

# First we get the headers of the Auger file
with open(raw_data, 'r') as f:
    reader = csv.DictReader(f)                   # Create a DictReader object
    headers = reader.fieldnames                  # Get the headers
    
# Load the df with the Auger data:
# -------------------------------
# Note that this data set contains about 24k events and its quite different from the
# data set with events used in Piere Auger Observatory, Science 357, 1266–1270 (2017).
#
df_Auger = pd.read_csv(raw_data, skiprows=1, names= headers, sep=",", index_col=False)

cols_E8 = ['year', 'day', 'dec', 'RA', 'azimuth', 'weight']
df_Ev8 = pd.read_table(data_Events_a8, skiprows=33, names= cols_E8, sep="\s+", index_col=False)


#E_thresh = 8.0 # EeV
#E_cut = ( df_Auger['sd_energy'] > E_thresh )
#df_data = df_Auger[ E_cut ]
#totalExposure = df_data['sd_exposure'].iloc[-1]

# The map with events
def map_events( nside, n_events, ipix):
    ''' Built a map of event counts
        Requires healpy
        
    Parameters
    ----------
    nside : int
        The nside parameter of a Healpix map
    n_events : int
        The total number of events of the map
    ipix : numpy array
        Array with the pixel that corresponds to each event.
        
    Returns
    -------
    A numpy array with the number counts in each pixel of the map.
    '''

    uhemap     = np.zeros( hp.nside2npix(nside) )#, dtype=np.float )
    counts     = np.ones(  n_events )
    np.add.at( uhemap, ipix, counts )
    #uhemap     = uhemap / np.sum(uhemap)                            # We normalize the map
    return uhemap


def hp_plot(map, title, Tmin, Tmax, output_file):
    ''' Produces a healpy map in Mollweide projection
        Requires healpy
        
    Parameters
    ----------
    map : numpy array
        The temperatue values (in units of \muK).
    Tmin : float
        The lower bound for T (values lesser than Tmin will be painted with the
    minimun value of the colorscale)
    Tmax : float
        The upper bound for T (values grater than Tmax will be painted with the
    maximun value of the colorscale)
    output_file : str
        The output file (.png in general)
        
    Returns
    -------
    The figure with the map in Mollweide projection.
    '''
    
    colormap = 'viridis'
    plt.figure()
    hp.mollview(map, coord='C', unit='Events [counts]', xsize=800,
                     title=title, cbar=True, cmap=colormap,
                     min=Tmin, max=Tmax, badcolor='white')
    plt.savefig(output_file)
    plt.close()


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


# make the healpy map
nside     = 64
n_events  = len(df_Ev8)
th_uhe, phi_uhe = np.deg2rad( 90 - df_Ev8['dec'].to_numpy() ),  np.deg2rad( df_Ev8['RA'].to_numpy() )
ipix_uhe  = hp.ang2pix( nside, th_uhe, phi_uhe, lonlat=False )      # This associated to each direction a pixel
Auger_map = map_events( nside, n_events, ipix_uhe )

# plot of the healpy map
Tmin, Tmax = np.min(Auger_map), np.max(Auger_map)
output_file = graficos+'hpmap_AugerE8.png'
title = 'UHECRs incoming events with E'+r'$\geq$ 8 EeV'
hp_plot( Auger_map, title, Tmin, Tmax, output_file )

# sky map with the events
th_str, phi_str = 'Dec (deg)', 'RA (deg)'
coord_sys = 'Equatorial'
output_file = graficos+'skymap_AugerE8.png'
sky_plot( th_uhe, phi_uhe, th_str, phi_str, coord_sys, title, output_file )

##################################################################
##################################################################
# Difference between the observations and the model...
diff_map = Auger_map - uhemap


# plot - healpy map of the difference...
Tmin, Tmax = np.min(diff_map), np.max(diff_map)
output_file = graficos+'hpmap_diff_AugerE8-model.png'
title = 'Difference between measurements and model'
hp_plot( diff_map, title, Tmin, Tmax, output_file )


# We compute the power spectrum of the map to check
l_max = 3*nside - 1
it_e = 3
Cls_diff  = hp.anafast( diff_map, nspec=None, lmax=l_max, iter=it_e, alm=False)
Cls_Auger = hp.anafast( Auger_map, nspec=None, lmax=l_max, iter=it_e, alm=False)
Cls_model = hp.anafast( uhemap, nspec=None, lmax=l_max, iter=it_e, alm=False)

# plot of the power spectrum
title = 'Power spectrum of the map of differences'
output_file = graficos+'PowSpec_AugerE8-model.png'
plt.close()
plt.figure()
ell = np.arange( 0, len(Cls_diff), 1 )
plt.scatter( ell, Cls_diff, s=3.5, label='Auger - model')
plt.scatter( ell, Cls_Auger, s=3.5, label='Auger')
plt.scatter( ell, Cls_model, s=3.5, label='model')
plt.title(title)
plt.xlabel('$\ell$ (Multipole)', fontsize='x-large')
plt.ylabel('$C_\ell$',fontsize='x-large')
plt.xlim(-1,15)
plt.legend()
plt.savefig(output_file)
#plt.show()
