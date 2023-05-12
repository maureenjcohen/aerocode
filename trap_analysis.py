#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:45:37 2023

@author: Mo Cohen
"""

import argparse, os
from aeropipe import *

""" Usage: python trap_analysis.py --input rot_space --ref reftrap"""


def init_trap(args):
    """ Instantiate Planet object for each TRAPPIST-1e simulation"""
    
    top_dir = '/exports/csce/datastore/geos/users/s1144983/exoplasim_data/'
    input_dir = args.input
    infiles = os.listdir(top_dir + input_dir)
    print(infiles)        
    
    planet_list = []
    for item in infiles:
        pl = Planet(trapdict)
        # Instantiate a Planet class object with info from the planet dict
        # This is stuff like planet radius, star temperature, etc.
        rot = item[5:]
        # Extract rotation period from filename
        print(rot)
        pl.rotperiod = rot
        # Overwrite default TRAP-1e rotation period with period used in sim
        pl.load_data(top_dir + item)
        # Load the file containing simulation data
        pl.savepath = '/home/s1144983/aerosim/trapspace/'
        # Change default directory to save plots to
        pl.add_rhog()
        # Calculate and add air density
        pl.area_weights()
        # Calculate and add area weights for area-weighted meaning
        planet_list.append(pl)
        
    return planet_list

def init_ref(args):
    """ Instantiate Planet object for reference TRAPPIST-1e simulations,
    with and without haze"""
    
    top_dir = '/exports/csce/datastore/geos/users/s1144983/exoplasim_data/'    
    ref_dir = args.ref
    infiles = sorted(os.listdir(ref_dir))
    print(infiles)
    
    ref_list = []    
    for item in infiles:
        pl = Planet(trapdict)
        # Instantiate Planet object with TRAPPIST-1e dictionary
        pl.load_data(top_dir + item)
        # Load data using filename
        pl.savepath = '/home/s1144983/aerosim/reftrap/'
        # Change default directory to save plots to
        pl.add_rhog()
        # Calculate and add air density
        pl.area_weights()
        # Calculate and add area weights for area-weighted meaning
        planet_list.append(pl)
        
    return ref_list

def winds(objlist, select = np.arange(1,31,1), savearg=False, sformat='png'):
    
    for plobject in objlist:
        if plobject.rotperiod in list(select):
            wind_vectors(plobject,
                         level=5,
                         savename='winds_lev_' + str(level) + '_' + 
                         str(plobject.rotperiod) + '.' + sformat, 
                         save=savearg, saveformat=sformat)
            
def mmr_maps(objlist, select = np.arange(1,31,1), savearg=False, sformat='png'):
    
    for plobject in objlist:
        if plobject.rotperiod in list(select):
            mmr_map(plobject, item='mmr',
                    level=5, cpower=7,
                    savename='mmrlev_' + str(level) + '_' + 
                    str(plobject.rotperiod) + '.' + sformat, 
                    save=savearg, saveformat=sformat)
            
def columns(objlist, select = np.arange(1,31,1), savearg=False, sformat='png'):
    
    for plobject in objlist:
        if plobject.rotperiod in list(select):
            haze_column = mass_column(plobject, mass_loading(plobject, 'mmr'))
            titlestr = f'Rotation period = {plobject.rotperiod} days'
            plot_lonlat(plobject, haze_column, title = titlestr, 
                        unit = 'kg m$^{-2}$',
                        savename='column_' + str(plobject.rotperiod) + 
                        '.' + sformat, 
                        save=savearg, saveformat=sformat)
            
def profiles(objlist, select = np.arange(1,31,1), savearg=False, sformat='png'):
    
    for plobject in objlist:
        if plobject.rotperiod in list(select):
            vert_profile(plobject, select='mmr',
                         savename = 'profiles_' + str(plobject.rotperiod) +
                         '.' + sformat,
                         save=savearg, saveformat=sformat)
            
def taus(objlist, select = np.arange(1,31,1), savearg=False, sformat='png'):
    
    for plobject in objlist:
        if plobject.rotperiod in list(select):
            tau(plobject, item='mmr', qext=plobject.sw1[-2], prad=1272,
                save=savearg, savename='tau_' + str(plobject.rotperiod) + 
                '.' + sformat, 
                saveformat=sformat)
            

def compare_refs(objlist, savearg=False, sformat='png'):
    
    compare_planets(objlist, ndata=2, level=5, n=2, qscale=10, fsize=14,
                    save=savearg, savename='', saveformat=sformat)
    
      


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Automated analysis pipeline for TRAPPIST-1e haze \
                    simulations')

    parser.add_argument(
        '--input',
        nargs=1,
        help='Path to parameter space input directory')
    
    parser.add_argument(
        '--ref',
        nargs=1,
        help='Path to directory containing reference TRAP-1e simulations \
            for climatology')
            
    args = parser.parse_args()
    # Parameter space sims
    all_traps = init_trap(args)
    winds(all_traps, savearg=False, sformat='png')
    mmr_maps(all_traps, savearg=False, sformat='png')
    columns(all_traps, savearg=False, sformat='png')
    taus(all_traps, savearg=False, sformat='png')
    
    # Reference sims
    ref_traps = init_ref(args) 
    compare_refs(ref_traps, savearg=False, sformat='png')
