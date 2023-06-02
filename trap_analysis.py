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
    input_dir = args.input[0]
    infiles = sorted(os.listdir(top_dir + input_dir))
    
    planet_list = []
    for item in infiles:
        pl = Planet(trapdict)
        # Instantiate a Planet class object with info from the planet dict
        # This is stuff like planet radius, star temperature, etc.
        rot = item[5:7]
        print(rot)
        # Extract rotation period from filename
        pl.rotperiod = rot
        # Overwrite default TRAP-1e rotation period with period used in sim
        pl.load_data(top_dir + input_dir + '/' + item, pspace=False)
        # Load the file containing simulation data
        pl.savepath = '/home/s1144983/aerosim/trapcontrol/'
        # Change default directory to save plots to
        pl.add_rhog()
        # Calculate and add air density
        pl.area_weights()
        # Calculate and add area weights for area-weighted meaning
        pl.add_vterm()
        # Calculate and add viscosity and terminal velocity
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
        pl.savepath = '/home/s1144983/aerosim/trapref/'
        # Change default directory to save plots to
        pl.add_rhog()
        # Calculate and add air density
        pl.area_weights()
        # Calculate and add area weights for area-weighted meaning
        planet_list.append(pl)
        
    return ref_list

def winds(objlist, select = [0.2]+[0.5]+list(np.arange(1,31,1)), level=3, savearg=False, sformat='png'):

    for plobject in objlist:
        if float(plobject.rotperiod) in list(select):
            print(float(plobject.rotperiod))
            wind_vectors(plobject,
                         level=level,
                         savename='winds_lev_' + str(level) + '_' + 
                         str(plobject.rotperiod) + '.' + sformat, 
                         save=savearg, saveformat=sformat, fsize=14)
            
def mmr_maps(objlist, select = [0.2]+[0.5]+list(np.arange(1,31,1)), level=5, savearg=False, sformat='png'):
    
    for plobject in objlist:
        if float(plobject.rotperiod) in list(select):
            mmr_map(plobject, item='mmr',
                    level=5, cpower=7,
                    savename='mmrlev_' + str(level) + '_' + 
                    str(plobject.rotperiod) + '.' + sformat, 
                    save=savearg, saveformat=sformat, fsize=14)
            
def columns(objlist, select = [0.2]+[0.5]+list(np.arange(1,31,1)), savearg=False, sformat='png'):
    
    for plobject in objlist:
        if float(plobject.rotperiod) in list(select):
            haze_column = mass_column(plobject, mass_loading(plobject, 'mmr'))
            titlestr = f'Rotation period = {plobject.rotperiod} days'
            plot_column(plobject, haze_column, title = titlestr, 
                        unit = '$10^{-5}$ kg m$^{-2}$', cpower=5,
                        savename='column_' + str(plobject.rotperiod) + 
                        '.' + sformat, 
                        save=savearg, saveformat=sformat, fsize=14)
            
def mmrprofiles(objlist, select = [0.2]+[0.5]+list(np.arange(1,31,1)), savearg=False, sformat='png'):
    
    for plobject in objlist:
        if float(plobject.rotperiod) in list(select):
            vert_profile(plobject, select='mmr',
                         savename = '_profiles_' + str(plobject.rotperiod) +
                         '.' + sformat,
                         save=savearg, saveformat=sformat, fsize=14)
            
def windprofiles(objlist, select = [0.2]+[0.5]+list(np.arange(1,31,1)), savearg=False, sformat='png'):

    for plobject in objlist:
        if float(plobject.rotperiod) in list(select):
            vert_profile(plobject, select='vterm',
                         savename = '_profiles_' + str(plobject.rotperiod) +
                         '.' + sformat,
                         save=savearg, saveformat=sformat, fsize=14)
            vert_profile(plobject, select='w',
                         savename = '_profiles_' + str(plobject.rotperiod) +
                         '.' + sformat,
                         save=savearg, saveformat=sformat, fsize=14)
            
def vterms(objlist, select = [0.2]+[0.5]+list(np.arange(1,31,1)), level=0, savearg=False, sformat='png'):

    for plobject in objlist:
        if float(plobject.rotperiod) in list(select):
            vterm(plobject, level=level, 
                  savename='vterm_' + str(plobject.rotperiod) + '.' + sformat,
                  save=savearg, saveformat=sformat, fsize=14)
            
def taus(objlist, select = [0.2]+[0.5]+list(np.arange(1,31,1)), savearg=False, sformat='png'):
    
    for plobject in objlist:
        if float(plobject.rotperiod) in list(select):
            tau(plobject, item='mmr', qext=plobject.sw1[-2], prad=5e-07, 
                pdens=1272,
                save=savearg, savename='tau_' + str(plobject.rotperiod) + 
                '.' + sformat, 
                saveformat=sformat, fsize=14, pplot=True)
            
def zmzws(objlist, select = [0.2]+[0.5]+list(np.arange(1,31,1)), savearg=False, sformat='png'):
    
    for plobject in objlist:
        if float(plobject.rotperiod) in list(select):
            zmzw(plobject, save=savearg, savename='zmzw_' + 
                 str(plobject.rotperiod) + '.' + sformat,
                 saveformat=sformat, fsize=14)
                 
def bulk_mass(objlist, savearg=False, sformat='png'):

    tlist = []
    elist = []
    wlist = []
    plist = []
    for plobject in objlist:
        prot = float(plobject.rotperiod)
        plist.append(prot)
        
        haze_column = mass_column(plobject, mass_loading(plobject, cube='mmr'))
        west_total, east_total, limb_total = limb_mass(plobject, haze_column)
        wlist.append(west_total)
        elist.append(east_total)
        tlist.append(limb_total)
    mlist = [wlist, elist, tlist]
    mass_distribution(objlist[0], plist, mlist, save=savearg,
                      savename='limb_mass' + '.' + sformat,
                      saveformat=sformat, fsize=14)
                      
def bulk_tau(objlist, savearg=False, sformat='png'):

    t3 = [] # Percentage where tau > 3
    t2 = [] # Percentage where tau > 2
    t1 = [] # Percentage where tau > 1
    plist = []
    for plobject in objlist:
        prot = float(plobject.rotperiod)
        plist.append(prot)
        west, east, limb, levs = tau(plobject, item='mmr', qext=plobject.sw1[-2], 
            prad=5e-07, pdens=1272,
            save=savearg, savename='tau_' + str(plobject.rotperiod) + 
            '.' + sformat, 
            saveformat=sformat, pplot=False)
        area3 = []
        area2 = []
        area1 = []
        for index, element in np.ndenumerate(west[1,:]):
            if element > 3.0:
                ar3 = plobject.area[index,16]
                area3.append(ar3)
            if element > 2.0:
                ar2 = plobject.area[index,16]
                area2.append(ar2)
            if element > 1.0:
                ar1 = plobject.area[index,16]
                area1.append(ar1)
        for index, element in np.ndenumerate(east[1,:]):
            if element > 3.0:
                ar3 = plobject.area[index,48]
                area3.append(ar3)
            if element > 2.0:
                ar2 = plobject.area[index,16]
                area2.append(ar2)
            if element > 1.0:
                ar1 = plobject.area[index,16]
                area1.append(ar1)
        areacov3 = np.sum(np.array(area3))/(2*np.sum(plobject.area[:,16]))
        areacov2 = np.sum(np.array(area2))/(2*np.sum(plobject.area[:,16]))
        areacov1 = np.sum(np.array(area1))/(2*np.sum(plobject.area[:,16]))        
        t3.append(areacov3)
        t2.append(areacov2)
        t1.append(areacov1)
    ttop = [t1, t2, t3]
    tau_distribution(objlist[0], plist, ttop, save=savearg,
                      savename='limb_tau' + '.' + sformat,
                      saveformat=sformat, fsize=14)

def tau_map(objlist, savearg=False, sformat='png'):

    tmax = []
    plist = []
    for plobject in objlist:
        prot = float(plobject.rotperiod)
        plist.append(prot)
        west, east, limb, levs = tau(plobject, item='mmr', qext=plobject.sw1[-2], 
            prad=5e-07, pdens=1272,
            save=savearg, savename='tau_' + str(plobject.rotperiod) + 
            '.' + sformat, 
            saveformat=sformat, pplot=False)
        hlist = []
        for h in range(0,levs.shape[0]):
            maxtau = np.max(limb[h,:])
            hlist.append(maxtau)
        tmax.append(hlist)
    tauplot = np.array(tmax)
    hsplot = np.mean(objlist[0].flpr, axis=(0,2,3))/100
    tau_contour(objlist[0], plist, hsplot,
                tauplot.T, save=savearg,
                savename='tau_contour' + '.' + sformat,
                saveformat=sformat, fsize=14)

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
    winds(all_traps, level=0, savearg=True, sformat='png')
    zmzws(all_traps, savearg=True, sformat='png')
    mmr_maps(all_traps, savearg=True, sformat='png')
    columns(all_traps, savearg=True, sformat='png')
    mmrprofiles(all_traps, savearg=True, sformat='png')
    windprofiles(all_traps, savearg=True, sformat='png')
    vterms(all_traps, savearg=True, sformat='png')
    taus(all_traps, savearg=True, sformat='png')
    bulk_mass(all_traps, savearg=True, sformat='png')
    bulk_tau(all_traps, savearg=True, sformat='png')
    tau_map(all_traps, savearg=True, sformat='png')
    
    # Reference sims
#    ref_traps = init_ref(args) 
#    compare_refs(ref_traps, savearg=False, sformat='png')
