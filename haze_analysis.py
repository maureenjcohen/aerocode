#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 09:59:59 2023

@author: Mo Cohen
"""
import argparse
from aeropipe import *


def main(args):
    """The MVC Controller of the haze analysis system.

    The Controller is responsible for:
    - Instantiating a Planet class object for the input planet(s)
    - Loading the simulation data for the planet(s)
    - Generating a standard set of plots for the planet(s)
    - (Optional) Generating a comparative set of plots for the planets
    """    
    dict_of_dicts = {'wolf' : wolfdict, 'trap' : trapdict, 'gj667' : gj667dict}
    top_dir = '/home/s1144983/aerosim/parameter_space/'
    # Where our data is located
    
    InFiles = args.infiles
    if not isinstance(InFiles, list):
        InFiles = [args.infiles]
        
    planet_list = []   
    for item in InFiles:
        planet = Planet(dict_of_dicts[item])
        # Instantiate a Planet class object with info from the planet dict
        # This is stuff like planet radius, star temperature, etc.
        planet.load_data(planet.datapath)
        # Load the file containing simulation data
        planet.add_rhog()
        # Calculate and add air density
        planet.area_weights()
        # Calculate and add area weights for area-weighted meaning
        planet.mmr_list()
        # Create list containing only names of mmr cubes for easy iteration
        self.rhop_lists()
        # Creates with simulations grouped by particle density
        planet_list.append(planet)
        
    if args.compare == True:
        print('Call comparative climatology here')
        compare_planets(planet_list, ndata=3)
        
    return planet_list

def mass_main(plobjects, showplot=False):
    """ Plot all haze mass colums and planetary limb mass versus particle size"""
    for planet in plobjects:
        data_1000 = []
        data_1262 = []
        data_1328 = []
        for item in planet.mmrs:
            print(item)
            particle_den = float(item[-10:-6])
            titlestr = f'{planet.longname}, r={item[-5]}e-{item[-1:]}m, rho={item[-10:-6]} kg/m3'
            haze_column = mass_column(planet, mass_loading(planet, item))
            if showplot == True:            
                plot_lonlat(planet, haze_column, title = titlestr, 
                            unit = 'kg m$^{-2}$', save = False,
                            savename = 'column_' + experiment + '.png', 
                            saveformat='png')
            limb_total = limb_mass(planet, haze_column)
            if particle_den == 1000:
                data_1000.append(limb_total)
            elif particle_den == 1262:
                data_1262.append(limb_total)
            elif particle_den == 1328:
                data_1328.append(limb_total)
            else:
                print('Particle density does not match parameter space')
        distribution(planet, [data_1000, data_1262, data_1328])
        distribution_scatter(planet, [data_1000, data_1262, data_1328])
        distribution_norm(planet, [data_1000, data_1262, data_1328], dens=1)
        
def profiles_main(plobjects,save=False, saveformat='png'):
    """ Plot vertical haze profiles"""
    for planet in plobjects:
        compare_profiles(planet, pdensity=1262, proflat=16, proflon=48,
                        titleloc='eastern terminator', save=save,
                        saveformat=saveformat, savename='east_term_profiles.png')
        compare_profiles(planet, pdensity=1262, proflat=16, proflon=16,
                        titleloc='western terminator', save=save,
                        saveformat=saveformat, savename='west_term_profiles.png')
        compare_profiles(planet, pdensity=1262, proflat=16, proflon=32,
                        titleloc='substellar point', save=save,
                        saveformat=saveformat, savename='sub_profiles.png')
        compare_profiles(planet, pdensity=1262, proflat=16, proflon=0,
                        titleloc='antistellar point', save=save,
                        saveformat=saveformat, savename='anti_profiles.png')
                        
def maps_main(plobjects):
    """ Plot loads of contourfill maps of mmr at various levels """
    for planet in plobjects:
        for listitem in planet.mmrs[0:1]:
            print(listitem)
            for l in [3,6,7]:
                mmr_map(planet, listitem, level=l, cpower=15,
                        save=False, 
                        savename=f'map_{listitem}_{l}.png',
                        saveformat='png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Automated analysis pipeline for 3-D haze simulations')

    parser.add_argument(
        'infiles',
        nargs='+',
        help='Name(s) input planet files. Options are trap, wolf, and gj667')

    parser.add_argument(
        '--compare',
        default=False,
        action='store_true',
        help='Whether to generate comparative plots')

    args = parser.parse_args()    
    all_planets = main(args)
    
    ### Now run bulk analysis functions
#    mass_main(all_planets, showplot=False)
    profiles_main(all_planets)
    maps_main(all_planets)
