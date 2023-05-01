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
        
    for item in InFiles:
        planet = Planet(dict_of_dicts[item])
        # Instantiate a Planet class object with info from the planet dict
        # This is stuff like planet radius, star temperature, etc.
        planet.load_data(planet.datapath)
        # Load the data file containing simulation data
        planet.add_rhog()
        # Calculate and add air density
        planet.area_weights()
        # Calculate and add area weights for area-weighted meaning
        planet.mmr_list()
        # Create list containing only names of mmr cubes for easy iteration
        dist_axis = []
        data_list = []
        for item in planet.mmrs:
            print(item)
            coeff, power = item[-5], item[-2:]
            particle_rad = float(coeff + 'e-' + power)
            particle_den = float(item[-10:-6])
            dist_axis.append(particle_rad)
            titlestr = f'{planet.longname}, r={item[-5]}e-{item[-1:]}m, rho={item[-10:-6]} kg/m3'
            haze_column = mass_column(planet, mass_loading(planet, item))            
            # plot_lonlat(planet, haze_column,title = titlestr, 
            #             unit = 'kg m$^{-2}$', save = False,
            #             savename = 'column_' + experiment + '.png', 
            #             saveformat='png')
            limb_total = limb_mass(planet, haze_column)
            data_list.append(limb_total)
        distribution(planet, dist_axis, data_list)
        


    if args.compare == True:
        print('Comparative climatology here')
        
    
    


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

    main(args)
