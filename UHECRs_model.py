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

import random


# Load the data files
cols_Auger = ['year', 'day', 'dec', 'RA', 'azimuth', 'weight']
df_Auger = pd.read_table(data_Events_a8, skiprows=33, names= cols_Auger, sep="\s+", index_col=False)

#df_gxs   = pd.read_table(data_2MRS, skiprows=10,\
#                         names=['RAdeg', 'DECdeg', 'l', 'b', 'k_c', 'h_c', 'j_c', 'k_tc', 'h_tc', 'j_tc',\
#                               'e_k', 'e_h', 'e_j', 'e_kt', 'e_ht', 'e_jt', 'e_bv', 'r_iso', 'r_ext',\
#                               'b/a', 'flgs', 'type', 'ts', 'v', 'e_v', 'c'], sep="\s+",\
#                         index_col=False)

cols_gxs = ['ID', 'RAdeg', 'DECdeg', 'cz', 'Ktmag_abs', 'l_deg', 'b_deg']
df_gxs = pd.read_table(data_2MRS_lowM, skiprows=1, names= cols_gxs, sep="\s+", index_col=False)

# Define the number of galaxies and the size of the map
n_evnt = 50000
nside  = 64

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

    cols_df_rdm = { 'l (rad)': lat_array,
                    'colat (rad)': colat_array,
                    'b (rad)': lon_array
                  }
    df_rdm = pd.DataFrame( data= cols_df_rdm )
    # Consider the cut in declinations > 45.0 deg
    #df_rdm = df_rdm[ df_rdm["DECdeg"] < 45.01 ]
    return df_rdm
    
df_rdm = rdm_sample( n_evnt )
df_rdm = df_rdm[ df_rdm['colat (rad)'] > np.deg2rad(45.01) ]

# Adjust the sample to have the number of observed UHECRs by Auger
n_CRs   = len(df_Auger)                                          # Number of observations

n_rdm   = 20000                                                  # Number of rdm event in our model
id_rdm  = df_rdm.index                                           # This variable contains all the index of the df_rdm
list_id = id_rdm.tolist()                                        # Here we transform it to a list for do iterations below...
id_pick = random.sample( list_id, n_rdm )                        # We choose the n_rdm indices of uhe_events from the df_rdm sample...
df_rdm_pick = df_rdm.loc[ id_pick ]                              # We choose those events from dr_rdm

th_rdm, phi_rdm = df_rdm_pick['colat (rad)'].to_numpy(), df_rdm_pick['b (rad)'].to_numpy()
ipix_uhe     = hp.ang2pix( nside, th_rdm, phi_rdm, lonlat=False )      # This associated to each direction a pixel

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
    
#    import numpy as np
#    import healpy as hp
#    import matplotlib.pyplot as plt

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
rdm_map = map_events( nside, n_rdm, ipix_uhe )

# plot of the healpy map
Tmin, Tmax = np.min(rdm_map), np.max(rdm_map)
output_file = graficos+'hpmap_rdm_contrib_model.png'
title = 'Random component to the model of UHECRs incoming events'
hp_plot( rdm_map, title, Tmin, Tmax, output_file )

# sky map with the events
th_str, phi_str = 'Dec (deg)', 'RA (deg)'
coord_sys = 'Equatorial'
output_file = graficos+'skymap_rdm_contrib_model.png'
sky_plot( th_rdm, phi_rdm, th_str, phi_str, coord_sys, title, output_file )



# 2) The gaussian contribution
# create Gaussian distribution around the position of galaxies
# Note: The generalization of the gaussian pdf into the unit 2-sphere can be done in several
#       ways. One of them is the von Mises-Fisher, other the Kent pdf. There exist others.
# Here we make a 2D Gaussian distribution on a flat disc and then project it onto the north pole
# of S^2. We finally rotate the points in order to make it centered at the position of the galaxies.

def generate_gaussian_points(n_samples, sigma):
    """
    Generates n_samples random points with a Gaussian distribution over the North Pole of the unit-2sphere.

    Parameters
    ----------
        n_samples : int
            Number of random points to generate.
        sigma : float
            Std of the distribution in degree
            
    Retunrs
    -------
        ndarray: Array of shape (n_samples, 3) containing the generated points.
    """
    points = []
    sigma = np.deg2rad(sigma) #np.sin( np.deg2rad(sigma) )
    while len(points) < n_samples:
        y = np.random.normal(0, sigma)
        z = np.random.normal(0, sigma)
        if y**2 + z**2 < 1:
            x = np.sqrt(1 - y**2 - z**2)
            points.append([x, y, z])
    return np.array(points)


def rotate_points(points, rot):
    """
    Rotates a set of points on the sphere.

    Args:
        points (ndarray): Array of shape (n_points, 3) containing the points to rotate.
        rot (tuple): Tuple containing the rotation angles (theta, phi, psi). (psi=0.0 below)

    Returns:
        ndarray: Array of shape (n_points, 3) containing the rotated points.
    """
    ph, th, ps = rot                                      # (RA, Dec, 90.0)
    c_ph = np.cos(ph)
    s_ph = np.sin(ph)
    c_th = np.cos(th)
    s_th = np.sin(th)
    c_ps = np.cos(ps)
    s_ps = np.sin(ps)
    r_11 = c_ph * c_ps - s_ph * c_th * s_ps
    r_12 = s_ph * c_ps + c_ph * c_th * s_ps #s_ph * c_ps + c_ph * c_th * s_ps
    r_13 = s_th * s_ps
    r_21 = - c_ph * s_ps - s_ph * c_th * c_ps
    r_22 = c_ph * c_th * c_ps - s_ph * s_ps
    r_23 = s_th * c_ps
    r_31 = s_ph * s_th
    r_32 = - c_ph * s_th
    r_33 = c_th
    rot_mat = [ [ r_11, r_12, r_13 ], [ r_21, r_22, r_23 ], [ r_31, r_32, r_33 ] ]
    # The convention her is that of Goldstein (1951) 'Classical Mechanics'.
    # In order to rotate a vector at x = [1,0,0] to the position of a galaxy in
    # (Dec, RA) the Euler angles are given by:
    #  phi = latitude - 90,      theta = latitude      and      psi = 90.0.
    
    # It return the coordinates of the rotated vector in the original coordinate system
    # Then, given the coordinate of the no-rotated points we multiply it by the transverse
    # of the rotation matrix previously defined.
    return np.dot( points, rot_mat )


df_sample   = df_gxs
th_gxs_str  = 'DECdeg'
phi_gxs_str = 'RAdeg'
points = []
sigma = 10.0                                              # std in deg
for i in range(0, len(df_sample)):
    vec_gauss_events = generate_gaussian_points( np.rint( (n_CRs - n_rdm)/len(df_sample) ), sigma )
    th_i  = df_sample[th_gxs_str][i]
    phi_i = df_sample[phi_gxs_str][i]
    rot_i    = np.deg2rad( np.mod(phi_i - 90, 360) ), np.deg2rad( th_i ), np.deg2rad(90.0)
    for j in range(0, len(vec_gauss_events)):
        points_i = rotate_points(vec_gauss_events[j], rot_i)
        points.append(points_i)


x_point = [item[0] for item in points]
y_point = [item[1] for item in points]
z_point = [item[2] for item in points]

# healpy map of the gaussian contribution
n_gsn = int( np.rint( (n_CRs - n_rdm)/len(df_sample) ) * len(df_sample) )
ipix_gsn = hp.vec2pix( nside, x_point, y_point, z_point )
gsn_map  = map_events( nside, n_gsn, ipix_gsn )

# The plot with the healp map
Tmin, Tmax = np.min(gsn_map), np.max(gsn_map)
output_file = graficos+'hpmap_gsn_contrib_model.png'
title = 'Gaussian component to the model of UHECRs incoming events'
hp_plot( gsn_map, title, Tmin, Tmax, output_file )

# The sky plot with the events for the Gaussian component of the model
points = np.array(points)
th_gsn, phi_gsn = hp.vec2ang( points )
th_str, phi_str = 'Dec (deg)', 'RA (deg)'
coord_sys = 'Equatorial'
output_file = graficos+'skymap_gsn_contrib_model.png'
sky_plot( th_gsn, phi_gsn, th_str, phi_str, coord_sys, title, output_file )


# The sky plot with the position of the galaxy sample
th_gxs  = np.deg2rad( 90 - df_sample[th_gxs_str].to_numpy() )           # This is co-latitude as required below
phi_gxs = np.deg2rad(df_sample[phi_gxs_str].to_numpy())
title = 'Distribution of the galaxy sample'
output_file = graficos+'skymap_gxs_smaple.png'
sky_plot( th_gxs, phi_gxs, th_str, phi_str, coord_sys, title, output_file )



# 3) The final model with the sum: rnd + gaussian
uhemap  = rdm_map + gsn_map

# plot of the healpy map
Tmin, Tmax = np.min(uhemap), np.max(uhemap)
output_file = graficos+'hpmap_model.png'
title = 'Model of UHECRs incoming events'
hp_plot( uhemap, title, Tmin, Tmax, output_file )

# plot of the scatter plot
th_CRs, phi_CRs = np.concatenate((th_rdm, th_gxs), axis=0), np.concatenate((phi_rdm, phi_gxs), axis=0)
output_file = graficos+'skymap_model.png'
sky_plot( th_CRs, phi_CRs, th_str, phi_str, coord_sys, title, output_file )
