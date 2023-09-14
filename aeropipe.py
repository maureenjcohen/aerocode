#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:06:43 2023

@author: Mo Cohen
"""
import numpy as np
import iris, os, glob, re, warnings
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter


warnings.filterwarnings('ignore')


# path1 = '/home/s1144983/aerosim/parameter_space/trap/'
# path2 = '/home/s1144983/aerosim/parameter_space/gj667/'
# path3 = '/home/s1144983/aerosim/parameter_space/wolf'


#top_dir = '/home/s1144983/aerosim/parameter_space/'
top_dir = '/exports/csce/datastore/geos/users/s1144983/exoplasim_data'

### Dictionaries for instantiating Planet objects ###

trapdict = {'radius' : 0.92, 'solcon' : 889, 'startemp' : 2709, 'N2' : 0.988,
            'CO2' : 0.01, 'CH4' : 0.002, 'H2' : 0.0, 'rotperiod' : 6.1, 'gravity' : 9.1,
            'starrad' : 0.12, 'starspec' : top_dir + 'stellarspectra/trap.dat', 
            'eccentricity': 0.0, 'bulk' : 1, 'msource' : 1e-07,
            'longname' : 'TRAPPIST-1e', 'name' : 'trap',
            'datapath' : '/exports/csce/datastore/geos/users/s1144983/exoplasim_data/trap.npz',
            'savepath' : '/home/s1144983/aerosim/parameter_space/trap/',
            'radpath' : top_dir + '/rad_space/trap_1262_5e-07_output/trap_1262_5e-07.npz',
            'sw1' : [0.00012240086441818405,
                    0.0006151318210905076,
                    0.001266000139194225,
                    0.006642325172355838,
                    0.02872566227427993,
                    0.0539588602959086,
                    0.15077745238699491,
                    0.3255354876370522,
                    3.4199725580951332,
                    2.7214575629384403],
            'sw2' : [4.4971745688298625e-05,
                    0.00022504315367610582,
                    0.0004520967690817146,
                    0.0015091181121235078,
                    0.0034382457616139862,
                    0.0051471296491337115,
                    0.011279554859380902,
                    0.023113095691854366,
                    2.031283471591182,
                    2.824994947652728],
            'radii' : [1e-09, 5e-09, 10e-09, 30e-09, 50e-09, 60e-09, 80e-09, 100e-09, 
                      500e-09, 1000e-09]}

wolfdict = {'radius' : 1.66, 'solcon' : 1777, 'startemp' : 3408, 'N2' : 0.988,
            'CO2' : 0.01, 'CH4' : 0.002, 'H2' : 0.0, 'rotperiod' : 17.9, 'gravity' : 12.1,
            'starrad' : 0.32, 'starspec' : top_dir + 'stellarspectra/wolf.dat',
            'longname' : 'Wolf-1061c', 'eccentricity': 0.0, 'bulk' : 1,
            'datapath' : '/exports/csce/datastore/geos/users/s1144983/exoplasim_data/wolf.npz',
            'msource' : 1e-07,'name' : 'wolf',
            'radpath' : top_dir + '/rad_space/wolf_1262_5e-07_output/wolf_1262_5e-07.npz',
            'savepath' : '/home/s1144983/aerosim/parameter_space/wolf/',
            'sw1' : [0.00014163704621179726,
                    0.0007125536179985144,
                    0.001475234352691232,
                    0.008438008128846446,
                    0.03907271356057308,
                    0.0744091034180985,
                    0.20828586457191473,
                    0.44489249993152424,
                    3.003450192354858,
                    2.6171906269584215],
            'sw2' : [4.7678258206987163e-05,
                    0.00023865651537159285,
                    0.00048023440127956753,
                    0.0016646900643897474,
                    0.00413257853165313,
                    0.00646407480908696,
                    0.015127530538967664,
                    0.03212686712249229,
                    2.3010643626807075,
                    2.8295767694273164],
            'radii' : [1e-09, 5e-09, 10e-09, 30e-09, 50e-09, 60e-09, 80e-09, 100e-09, 
                      500e-09, 1000e-09]}

gj667dict = {'radius' : 1.77, 'solcon' : 1202, 'startemp' : 3594, 'N2' : 0.988,
            'CO2' : 0.01, 'CH4': 0.002, 'H2' : 0.0, 'rotperiod' : 28.1, 'gravity' : 11.9,
            'starrad' : 0.42, 'starspec' : top_dir + 'stellarspectra/gj667.dat',
            'longname' : 'GJ667Cc', 'eccentricity': 0.0, 'bulk' : 1, 
            'msource' : 1e-07, 'name' : 'gj667',
            'datapath' : '/exports/csce/datastore/geos/users/s1144983/exoplasim_data/gj667.npz',
            'savepath' : '/home/s1144983/aerosim/parameter_space/gj667/',
            'radpath' : top_dir + '/rad_space/gj667_1262_5e-07_output/gj667_1262_5e-07.npz',
            'sw1' : [0.00014466752688943883,
                    0.0007278859924406788,
                    0.0015079424133715008,
                    0.008698959267629749,
                    0.040524385984645375,
                    0.07726669481717197,
                    0.2163266723091629,
                    0.46048919424375906,
                    2.9543005330015393,
                    2.5997034760047293],
            'sw2' : [4.679939883498154e-05,
                    0.00023425616254302335,
                    0.0004713667608222034,
                    0.0016329419369405414,
                    0.0040484391881861,
                    0.006328478171784975,
                    0.014796642286503207,
                    0.031406455470575016,
                    2.2779839722279585,
                    2.8847760487288356],
            'radii' : [1e-09, 5e-09, 10e-09, 30e-09, 50e-09, 60e-09, 80e-09, 100e-09, 
                      500e-09, 1000e-09]}


### Planet class object definition ###
### Manages planet configuration data and simulation output data.
### Also adds a few useful supplementary values like area weights

class Planet:
    """ A Planet object which contains the model output data for a planet,
    plus all mmr cubes for the parameter space labelled by the name of each
    simulation in the parameter space."""
    
    def __init__(self, planetdict):
        self.longname = planetdict['longname']
        print(f'Welcome to {self.longname}!')
        for key, value in planetdict.items():
            setattr(self, key, value) 
        
    def load_files(self, datadir):
        """ Load files in a data directory and combine into one dictionary"""
        filelist = sorted(os.listdir(datadir))       
        allmmr = {}
        simnames = []
        for file in filelist:
            simname = file[:-4]
            simname = simname.replace('-','_')
            simnames.append(simname)
            filedata = dict(np.load(datadir+file))
            filedata[simname] = filedata['mmr']
            allmmr[simname] = filedata[simname]
        
        otherdata = dict(np.load(datadir+filelist[0]))
        del otherdata['mmr']
        othernames = list(otherdata.keys())
        
        planetdata = {**otherdata, **allmmr}
        self.data = planetdata
        self.contents = othernames + simnames
        for key, value in planetdata.items():
            setattr(self, key, value)
            
    def save_files(self):
        """ Save planetdata into a single npz file"""
        np.savez(top_dir + self.name + '.npz', 
                **{name:value for name,value in zip(self.contents, self.data.values())})
    
    def load_data(self, filepath, pspace=False):
        """ Load a single preprocessed npz file with all data"""
        planetdata = dict(np.load(filepath))
        planetnames = list(planetdata.keys())
        if pspace == True:
            for key in planetnames:
                if key == 'mmr':
                    simfile = filepath.split('/')[-1]
                    simname = simfile[:-4]
                    simname = simname.replace('-','_')
                    planetdata[simname] = planetdata.pop('mmr')
                    planetnames = [simname if item == 'mmr' else item for item in
                                   planetnames]
        self.data = planetdata
        self.contents = planetnames
        for key, value in planetdata.items():
            setattr(self, key, value)
            
    def add_rhog(self):   
        """ Calculate density of air and add to planet sim data"""
        rhog = (self.flpr*0.028)/(8.314*self.ta) # (time, height, lat, lon)
        self.rhog = rhog
        
    def add_eta(self):
        kb = 1.3806e-23 # m2 kg s-2 K-1, Boltzmann's constant
        mair = 4.652e-26 # kg, molecular mass of N2
        dair = 3.64e-10 # m molecular diameter of N2
        eps = 95.5 # kb*K, Lennard-Jones potential well of N2
        
        rt_temp = np.sqrt(self.ta)
        rt_mair = np.sqrt(mair)
        rt_pi = np.sqrt(np.pi)
        rt_kb = np.sqrt(kb)
        sq_dair = dair*dair
        temp_eps = (self.ta/eps)**0.16
        coeff = (5/16)*(1/1.22)*(1/np.pi)*rt_pi*rt_mair*rt_kb/sq_dair
        
        eta = coeff*rt_temp*temp_eps
        self.eta = eta
        
    def add_vterm(self):
        self.add_rhog()
        self.add_eta()
        
        kb = 1.3806e-23 # J/K
        dm = 3.64e-10 # m, molecular diameter of N2
        coeff = (kb/dm)*(1/(np.sqrt(2)*np.pi*dm))
        frpath = coeff*(self.ta/self.flpr)
        
        kn = frpath/500e-09 # Knudsen number
        pwer = -1.1/kn
        beta = 1 + kn*(1.256 + 0.4*np.exp(pwer))
        
        vel = -2*beta*((500e-09)**2)*self.gravity*(1262 - self.rhog)/(9*self.eta)
        self.vterm = vel
        
        
    def area_weights(self):
        """ Calculate array of area of each grid cell for use in averaging"""
        xlon, ylat = np.meshgrid(self.lon, self.lat)
        dlat = np.deg2rad(np.abs(np.gradient(ylat, axis=0)))
        dlon = np.deg2rad(np.abs(np.gradient(xlon, axis=1)))
        rad = self.radius*6371000
        dy = dlat*rad
        dx = dlon*rad*np.cos(np.deg2rad(ylat))
        area = dy*dx
        self.area = area
        self.dy = dy
        self.dx = dx
        
    def mmr_list(self):
        """ Make list of mmr data cubes only"""
        mmrs = [item for item in self.contents if 
                item.split('_')[0] == self.name]            
        self.mmrs = mmrs
        if len(self.mmrs) == 0:
            self.mmrs = self.mmr
        
    def rhop_lists(self):
        """ Make lists containing the simulations
        grouped by density"""
        rhop1000 = [item for item in self.contents 
                   if item.split('_')[0] == self.name 
                   if item.split('_')[1] == 1000]
        rhop1262 = [item for item in self.contents 
                   if item.split('_')[0] == self.name 
                   if item.split('_')[1] == 1262]
        rhop1328 = [item for item in self.contents 
                   if item.split('_')[0] == self.name 
                   if item.split('_')[1] == 1328]
        self.rhop1000 = rhop1000
        self.rhop1262 = rhop1262
        self.rhop1328 = rhop1328
                
        
### Functions

def normalise(array, aymax=None, aymin=None):
    """ Normalise input array between 0 and 1"""
    if aymax == None:
        aymax = np.max(array)
    if aymin == None:
        aymin = np.min(array)
    normed_array = (array - aymin)/(aymax - aymin)    
    return normed_array

def mass_loading(plobject, cube):
    """ Calculate haze mass loading in kg(of haze)/m3"""
    datacube = plobject.data[cube]
    mass_load = datacube*plobject.rhog        
      
    return mass_load

def mass_column(plobject, cube, month=-1, meaning=True):
    """ Calculate vertically integrated haze mass column (kg/m2)
    
    We exclude the top level from the integration because it contains the
    fixed haze source, which is of greater magnitude than the lower level
    mass loading."""
    if meaning == True:
        dz = np.diff(np.mean(plobject.hlpr,axis=0),axis=0)/(plobject.gravity*np.mean(plobject.rhog,axis=0))
        column = np.sum(np.mean(cube[:,1:,:,:],axis=0)*dz[1:,:,:],axis=0)
    else:
        dz = np.diff(plobject.hlpr[month,:,:,:],axis=0)/(plobject.gravity*plobject.rhog[month,:,:,:])
        column = np.sum(cube[month,1:,:,:]*dz[1:,:,:],axis=0)
    return column    
    
def plot_column(plobject, cube, title = 'Planet', unit='$10^{-5}$ kg m$^{-2}$',
                cpower=5,
                colors='plasma', save=False, savename='plotname.png', 
                saveformat='png', fsize=14):
    """ Make a lon-lat contourfill plot of the haze column"""
    coeff = 10**cpower
    # if plobject.name == 'trap':
    #     if float(plobject.rotperiod) < 1:
    #         clevs = np.arange(0.15, 5.0, 0.1)
    #     else:
    #         clevs = np.arange(0.15, 2.26, 0.1)
    # elif plobject.name == 'wolf':
    #     clevs = np.arange(0.15, 5.3, 0.1)
    # else:
    #     print('Check Planet object name')
    if float(plobject.rotperiod) < 2.0:
        clevs = np.arange(0.15, 5.0, 0.1)
    elif (float(plobject.rotperiod) >= 2.0) and (float(plobject.rotperiod) < 4.0):
        clevs = np.arange(0.15, 1.6, 0.05)
    elif (float(plobject.rotperiod) >= 4.0) and (float(plobject.rotperiod) < 13.0):
        clevs = np.arange(0.15, 5.6, 0.1)
    elif float(plobject.rotperiod) >= 13.0:
        clevs = np.arange(0.15, 2.6, 0.05)
        
    fig, ax = plt.subplots(figsize=(8,5))
    plt.contourf(plobject.lon, plobject.lat, cube*coeff, 
                 levels=clevs, cmap=colors)
    plt.title('Total haze column excluding source',
              fontsize=fsize)
    plt.xlabel('Longitude [deg]', fontsize=fsize)
    plt.ylabel('Latitude [deg]', fontsize=fsize)
    plt.xticks((plobject.lon[0], plobject.lon[15], plobject.lon[31], 
                plobject.lon[47], plobject.lon[63]), 
                ('180W', '90W', '0','90E','180E'))
    plt.yticks((plobject.lat[0], plobject.lat[7], plobject.lat[15], 
                plobject.lat[23], plobject.lat[31]),
                ('90N', '45N', '0', '45S', '90S'))
    cbar = plt.colorbar(orientation='vertical', fraction=0.05)
    cbar.set_label(unit, fontsize=fsize-2)
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat, 
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
def limb_mass(plobject, inputcol):
    """ Calculate total haze mass at planetary limb
        From vertically integrated haze mass loading column"""
    westsum = (np.sum(inputcol[:,15]*plobject.area[:,15]))
    eastsum = (np.sum(inputcol[:,47]*plobject.area[:,47]))
    limbsum = westsum + eastsum
       
    return westsum, eastsum, limbsum
    
def mass_distribution(plobject, inputaxis, inputdata, save=False, 
                     savename='massdistribution.png', 
                     saveformat='png', fsize=14):
    """ Plot total integrated haze at terminator against rotation rate"""
    limb_area = np.sum(plobject.area[:,16]) + np.sum(plobject.area[:,48])
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(np.array(inputaxis), np.array(inputdata[0])/(0.5*limb_area),
                color='b', label='Western terminator')
    plt.plot(np.array(inputaxis), np.array(inputdata[1])/(0.5*limb_area),
                color='r', label='Eastern terminator')
    plt.plot(np.array(inputaxis), np.array(inputdata[2])/(limb_area),
                color='k', label='Total')
    plt.title('Haze mass at planetary limb', fontsize=fsize)
    plt.xlabel('Planet rotation period [days]', fontsize=fsize)
    plt.ylabel('Haze mass [$kg m^{-2}$]', fontsize=fsize)
    plt.ylim((0.0, 3.6e-05))
    plt.legend(loc='upper right')
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat, 
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def tau_distribution(plobject, inputaxis, inputdata, save=False, 
                     savename='taudistribution.png', 
                     saveformat='png', fsize=14):
    """ Plot total integrated haze at terminator against rotation rate"""
    
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(np.array(inputaxis), np.array(inputdata[0])*100, 
                color='k', label=r'$\frac{d\tau}{ds}$ > 1')
    plt.plot(np.array(inputaxis), np.array(inputdata[1])*100, 
                color='b', label=r'$\frac{d\tau}{ds}$ > 2')
    plt.plot(np.array(inputaxis), np.array(inputdata[2])*100, 
                color='r', label=r'$\frac{d\tau}{ds}$ > 3')
    plt.title('Percent of tropopause where limb exceeds differential optical thickness of 1, 2 and 3', fontsize=fsize)
    plt.xlabel('Planet rotation period [days]', fontsize=fsize)
    plt.ylabel('Percent [%]', fontsize=fsize)
    plt.legend(loc='lower right')
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat, 
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def tau_contour(plobject, xaxis, yaxis, inputdata, save=False,
                savename='taucontour.png', saveformat='png',
                fsize=14):
    """ Contour fill of max limb tau vs height and rotation rate"""
    fig, ax = plt.subplots(figsize=(8,5))
    plt.contourf(np.array(xaxis), np.array(yaxis), np.array(inputdata),
                levels=np.arange(0,21,1), cmap='plasma')
    plt.title('Max differential optical depth at planetary limb', fontsize=fsize)
    plt.xlabel('Planet rotation period [days]', fontsize=fsize)
    plt.ylabel('Pressure [mbar]', fontsize=fsize)
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(orientation='vertical', fraction=0.05)
    cbar.set_label(r'$\frac{d\tau}{ds}$', fontsize=fsize-2)
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat, 
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
        
def surf_space(plobject, inputaxis, radlist, conlist, save=False,
                savename='surftemp_dist.png', saveformat='png',
                fsize=14):
    
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(np.array(inputaxis), np.array(radlist), 
                color='r', label=r'With haze')
    plt.plot(np.array(inputaxis), np.array(conlist), 
                color='b', label=r'Without haze')

    plt.title('Global mean surface temperature', fontsize=fsize)
    plt.xlabel('Planet rotation period [days]', fontsize=fsize)
    plt.ylabel('Temperature [K]', fontsize=fsize)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.legend(loc='upper right')   
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat, 
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def vapour_space(plobject, inputaxis, radlist, conlist, save=False,
                savename='vapour_dist.png', saveformat='png',
                fsize=14):
    
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(np.array(inputaxis), np.array(radlist), 
                color='r', label=r'With haze')
    plt.plot(np.array(inputaxis), np.array(conlist), 
                color='b', label=r'Without haze')

    plt.title('Global mean water vapour column', fontsize=fsize)
    plt.xlabel('Planet rotation period [days]', fontsize=fsize)
    plt.ylabel('Water vapour [kg/m$^2$]', fontsize=fsize)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.legend(loc='upper right')   
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat, 
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def prw_space(plobject, inputaxis, radlist, conlist, save=False,
                savename='prw_dist.png', saveformat='png',
                fsize=14):
    
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(np.array(inputaxis), np.array(radlist), 
                color='r', label=r'With haze')
    plt.plot(np.array(inputaxis), np.array(conlist), 
                color='b', label=r'Without haze')

    plt.title('Fraction of water vapour on nightside', fontsize=fsize)
    plt.xlabel('Planet rotation period [days]', fontsize=fsize)
    plt.ylabel('Nightside vapour fraction', fontsize=fsize)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.legend(loc='upper right')   
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat, 
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    
    


def distribution(plobject, inputaxis, inputlists, save=False,
                savename='plotname.png', saveformat='png', fsize=14):
    """ Plot total integrated haze at terminator against particle size"""
    limb_area = np.sum(plobject.area[:,15]) + np.sum(plobject.area[:,47])    
    fig, ax = plt.subplots(figsize=(8,5))    
    plt.scatter(np.array(inputaxis), np.array(inputlists[0])/limb_area, 
            c='g', label='1000 kg/m$^3$')
    plt.scatter(np.array(inputaxis), np.array(inputlists[1]/limb_area), 
            c='r', label='1262 kg/m$^3$')
    plt.scatter(np.array(inputaxis), np.array(inputlists[2]/limb_area), 
            c='b', label='1328 kg/m$^3$')
    plt.title('Total haze mass at the planetary limb', fontsize=fsize)
    plt.xlabel('Particle radius [m]', fontsize=fsize)
    plt.ylabel('Mass [kg/m2]', fontsize=fsize)
    plt.xscale('log')
    plt.legend()
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat, 
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
def distribution_scatter(plobject, inputaxis, inputlists, save=False,
                savename='plotname.png', saveformat='png'):
    """ Plot total integrated haze at terminator against particle size"""
    
    fig, ax1 = plt.subplots(figsize=(6,4))
    line1 = ax1.scatter(np.array(inputaxis), np.array(inputlists[0]), 
            c='g', label='1000 kg/m$^3$')
    line2 = ax1.scatter(np.array(inputaxis), np.array(inputlists[1]), 
            c='r', label='1262 kg/m$^3$')
    line3 = ax1.scatter(np.array(inputaxis), np.array(inputlists[2]), 
            c='b', label='1328 kg/m$^3$')
    ax1.set_ylabel('kg')
    ax1.set_xlabel('Particle radius [m]')
#        plt.yscale('log')
    ax1.set_xscale('log')
    
    ax2 = ax1.twinx()
    line4 = ax2.scatter(np.array(plobject.radii), np.array(plobject.sw1), 
            marker='s', color='k', label='Band 1')
    line5 = ax2.scatter(np.array(plobject.radii), np.array(plobject.sw2), 
            marker='s', color='m', label='Band 2')
    ax2.set_ylabel('Extinction efficiency')
    lns = [line1,line2,line3,line4,line5]
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs)
    plt.title('Total haze mass at planetary limb \n and extinction efficiency')

    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat,
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def distribution_norm(plobject, inputaxis, inputlists, dens=1, save=False,
                savename='plotname.png', saveformat='png'):
    """ Plot total integrated haze at terminator against particle size"""
    
    normed_sw = normalise(np.array(plobject.sw1))
    normed_kg = normalise(np.array(inputlists[dens]))

    fig, ax = plt.subplots(figsize=(6,4))
    plt.plot(np.array(plobject.radii), 
             normed_kg*normed_sw, c='b')
    plt.ylabel('Effect size')
    plt.xlabel('Particle radius [m]')
    plt.xscale('log')
    plt.title('Effect size of haze by particle size')
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat,
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()    
    
def compare_profiles(plobject, pdensity=1262, proflat=15, proflon=47,
                    titleloc='eastern terminator', save=False,
                    savename='plotname.png', saveformat='png'):
    
    """ Plot vertical profiles of haze mmr at input lat and lon. 
        Args for proflon:
        0: Antistellar point
        16: Western terminator
        32: Substellar point
        48: Eastern terminator
        63: Antistellar point again """
    mmrs_toplot = []
    psizes = []
    for item in plobject.mmrs:
        particle_den = float(item[-10:-6])        
        if particle_den == pdensity:
            data = np.mean(plobject.data[item], axis=0)
            mmrs_toplot.append(data)            
            coeff, power = item[-5], item[-2:]
            particle_rad = float(coeff + 'e-' + power)
            psizes.append(particle_rad)
        
    titlelat = int(np.round(plobject.lat[proflat],0))
    titlelon = int(np.round(plobject.lon[proflon],0))
    pressure = np.mean(plobject.flpr[:,:,proflat,proflon],axis=0)/100
    
    fig, ax = plt.subplots(figsize=(7,5))
    for i in range(0,len(mmrs_toplot)):
        mmr = mmrs_toplot[i]
        plabel = psizes[i]
        plt.plot(mmr[:,proflat,proflon], pressure,
                 label=f'{plabel}')
    plt.gca().invert_yaxis()
    plt.title(f'Mean haze profile at {titleloc}',
              fontsize=14)
    plt.xlabel('Mass mixing ratio [kg/kg]', fontsize=14)
    plt.ylabel('Pressure [mbar]', fontsize=14)
#    plt.xlim(0,plobject.msource)
    plt.legend(fontsize='small')
    fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat,
                    bbox_inches='tight')
    plt.show()
    
    return
    
def mmr_map(plobject, item, level=5, cpower=9, 
            save=False, savename='plotname.png', 
            saveformat='png', fsize=14):
    
    coeff = 10**cpower
    
    # if plobject.name == 'trap':
    #     clevs = np.arange(0.0, 0.021, 0.001)
    # elif plobject.name == 'wolf':
    #     clevs = np.arange(0.0, 0.06, 0.002)
    # else:
    #     print('Check Planet object name')
#    clevs = np.arange(0.0, 1.1, 0.1)
    if float(plobject.rotperiod) < 2.0:
        clevs = np.arange(0.0, 20, 0.1)
    elif (float(plobject.rotperiod) >= 2.0) and (float(plobject.rotperiod) < 4.0):
        clevs = np.arange(0.0, 6.1, 0.01)
    elif (float(plobject.rotperiod) >= 4.0) and (float(plobject.rotperiod) < 13.0):
        clevs = np.arange(0.0, 6.1, 0.01)
    elif float(plobject.rotperiod) >= 13.0:
        clevs = np.arange(0.0, 6.1, 0.01)
    fig, ax = plt.subplots(figsize=(8,5))    
    plt.contourf(plobject.lon, plobject.lat, 
                np.mean(plobject.data[item][:,level,:,:],axis=0)*coeff,
                 levels=clevs,
                 cmap='plasma')
    plt.title('Mass mixing ratio at %s mbar' % 
              (np.round(np.mean(plobject.flpr[:,level,15,31], axis=0)/100)),
              fontsize=fsize)
    plt.xlabel('Longitude [deg]', fontsize=fsize)
    plt.ylabel('Latitude [deg]', fontsize=fsize)
    plt.xticks((plobject.lon[0], plobject.lon[15], plobject.lon[31], 
                plobject.lon[47], plobject.lon[63]), 
                ('180W', '90W', '0','90E','180E'))
    plt.yticks((plobject.lat[0], plobject.lat[7], plobject.lat[15], 
                plobject.lat[23], plobject.lat[31]),
                ('90N', '45N', '0', '45S', '90S'))
    cbar = plt.colorbar(orientation='vertical', fraction=0.05)
    cbar.set_label('$10^{-9}$ kg/kg', loc='center', fontsize=fsize-2)
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat,
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()  
    
    return
    
def compare_planets(plobjects, ndata=3, level=5, n=2, qscale=10, fsize=14,
                    whichlevs = 'trap', save=False, savename='compclim.png',
                    saveformat='png'):
    """ Compare climatology of all planets in parameter space"""
    redblu = mpl_cm.get_cmap('coolwarm')
    hot = mpl_cm.get_cmap('hot')
    
    if whichlevs == 'trap':
        surflevs = np.arange(140, 300, 10) # Surf temp
        zlevs = np.arange(-60, 81, 10) # ZMZW
        wlevs = np.arange(-0.08, 0.61, 0.01) # Vertical wind
        tlims = (140, 300) # T profile
        vlims = (-0.001, 0.008) # V profile
    elif whichlevs == 'wolf':
        surflevs = np.arange(230, 350, 10) # Surf temp
        zlevs = np.arange(-20, 100, 10) # ZMZW
        wlevs = np.arange(-0.15, 0.61, 0.01) # Vertical wind 
        tlims = (240, 330) # T profile
        vlims = (-0.001, 0.1) # V profile
    else:
        print(f'{whichlevs} is not a valid keyword')

    fig, ax = plt.subplots(figsize=(14, 20), nrows=5, ncols=ndata)
    fig.tight_layout(h_pad=4, w_pad=4)
    for i in range(ndata):
        plobject = plobjects[i]
        #Surface temperature
        surf = ax[0, i].contourf(plobject.lon, plobject.lat,
               np.mean(plobject.ts, axis=0), 
               levels=surflevs,
               cmap=hot)
        ax[0,0].set_title(f'{plobject.longname}', fontsize=fsize+2)
        ax[0,1].set_title(f'{plobject.longname}, hazy', fontsize=fsize+2)
        sbar = fig.colorbar(surf, ax=ax[0, i], orientation='vertical', fraction=0.05) 
        sbar.set_label('K', loc='center', fontsize=fsize)
        ax[0, i].set_xlabel('Longitude [deg]', fontsize=fsize)
        ax[0, 0].set_ylabel('Surface temperature [K] \n Latitude [deg]', fontsize=fsize)
        # Air temperature profile
        ax[1, i].plot(np.mean(plobject.ta[:,:,16,32], axis=0),
                      np.mean(plobject.flpr[:,:,16,32]/100,axis=0),
                      color='r', label='Substellar')
        ax[1, i].plot(np.mean(plobject.ta[:,:,16,0], axis=0),
                      np.mean(plobject.flpr[:,:,16,0]/100,axis=0),
                      color='b', label='Antistellar')
        ax[1, i].invert_yaxis()
        ax[1, i].set_xlim(tlims[0], tlims[1])
        ax[1, i].legend(fontsize=fsize)
        ax[1, i].set_xlabel('Temperature [K]', fontsize=fsize)
        ax[1, 0].set_ylabel('Temperature profile [K] \n Pressure [mbar]', fontsize=fsize)
        # Water vapour profile
        ax[2, i].plot(np.mean(plobject.hus[:,:,16,32], axis=0),
                      np.mean(plobject.flpr[:,:,16,32]/100,axis=0),
                      color='r', label='Substellar')
        ax[2, i].plot(np.mean(plobject.hus[:,:,16,0], axis=0),
                      np.mean(plobject.flpr[:,:,16,0]/100,axis=0),
                      color='b', label='Antistellar')
        ax[2, i].invert_yaxis()
        ax[2, i].set_xlim(vlims[0], vlims[1])
        ax[2, i].legend(fontsize=fsize)
        ax[2, i].set_xlabel('Specific humidity [kg/kg]', fontsize=fsize)
        ax[2, 0].set_ylabel('Vapour profile [kg/kg] \n Pressure [mbar]', fontsize=fsize)
        # ZMZW
        cont = ax[3, i].contourf(plobject.lat, np.mean(plobject.flpr[:,:,16,32]/100, axis=0), 
               np.mean(plobject.ua, axis=(0,3)), levels=zlevs, 
               cmap=redblu, norm=TwoSlopeNorm(0))
        ax[3, i].invert_yaxis()
        ax[3, 0].set_ylabel('Zonal mean zonal wind [m/s] \n Pressure [mbar]', fontsize=fsize)
        ax[3, i].set_xlabel('Latitude [deg]', fontsize=fsize)
        zbar = fig.colorbar(cont, ax=ax[3, i], orientation='vertical', fraction=0.05)
        zbar.set_label('m/s', loc='center', fontsize=fsize)
        # Winds
        X, Y = np.meshgrid(np.arange(0, len(plobject.lon)), np.arange(0, len(plobject.lat)))
        w = ax[4, i].contourf(np.mean(plobject.wa[level,:,:]*1e03,axis=0), 
                   cmap='coolwarm', levels=wlevs, norm=TwoSlopeNorm(0))
        wbar = fig.colorbar(w, ax=ax[4, i], orientation='vertical', fraction=0.05)
        wbar.set_label('$10^{-3}$ m/s', loc='center', fontsize=fsize)
        
        q1 = ax[4, i].quiver(X[::n, ::n], Y[::n, ::n], 
                            np.mean(plobject.ua[:, level, ::n, ::n], axis=0),
                            -np.mean(plobject.va[:, level, ::n, ::n], axis=0),
                            scale_units='xy', scale=qscale)
        ax[4, i].quiverkey(q1, X=0.9, Y=1.05, U=qscale*2, label='%s m/s' %str(qscale*2),
                     labelpos='E', coordinates='axes')
        ax[4,i].set_title(f'{int(np.mean(plobject.flpr[:,level,:,:])/100)} mbar', fontsize=fsize)
        ax[4, i].set_xticks((0, 15, 31, 47, 63), 
                    ('180W', '90W', '0','90E','180E'))
        ax[4, i].set_yticks((0, 7, 15, 23, 31 ),
                    ('90S', '45S', '0', '45N', '90N'))
        ax[4, i].set_xlabel('Longitude [deg]', fontsize=fsize)
        ax[4, 0].set_ylabel('Horizontal and vertical winds [m/s] \n Latitude [deg]', fontsize=fsize)
        ax[4, i].set_xlabel('Longitude [deg]', fontsize=fsize)
    if save == True:
        plt.savefig(plobjects[0].savepath + savename, format=saveformat,
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    
def mmr2n(plobject, item='mmr', partrad=5e-07, pdens=1272):
    """ Convert mass mixing ratio (kg/kg) to number density (particles/m3)"""
    mmr_raw = plobject.data[item] # in kg/kg
    
    if item == 'mmr':
        particle_rad = partrad
        particle_den = pdens
    else:
        coeff, power = item[-5], item[-2:]
        particle_rad = float(coeff + 'e-' + power)
        particle_den = float(item[-10:-6])
    sphere_vol = (4/3)*np.pi*(particle_rad**3)
    particle_mass = sphere_vol*particle_den
    
    outcube = mmr_raw*(1/particle_mass)*plobject.rhog # particles/m3
    nsource = plobject.msource*(1/particle_mass)*np.mean(plobject.rhog[:,0,15,31], axis=0)
    return outcube, nsource
    
def tau(plobject, item, qext, prad, pdens, time_slice=-1, meaning=True,
        save=False, savename='plotname.png', saveformat='png', fsize=14, 
        pplot=True):
    """ Calculate optical depth in the line of sight when viewed during transit
        Considers only the actual terminator and not the 3-D geometry   """
    if meaning == True:
        nrho, nsource = mmr2n(plobject, item=item)
        nrho = np.mean(nrho, axis=0)
        heights = np.mean(plobject.flpr, axis=0)
    else:
        nrho, nsource = mmr2n(plobject, item=item)
        nrho = nrho[time_slice,:,:,:]
        heights = plobject.flpr[time_slice,:,:,:]
        
    west = nrho[:,:,15]*qext*np.pi*(prad**2)*1000000
    east = nrho[:,:,47]*qext*np.pi*(prad**2)*1000000
    limb = np.concatenate((west, east[:,::-1]), axis=1)
    heights = np.mean(heights, axis=(1,2))/100
    
    # if plobject.name == 'trap':
    #     if float(plobject.rotperiod) < 1.0:
    #         clevs = np.arange(0,12.6,0.2)
    #     else:
    #         clevs = np.arange(0,9.1,0.2)
    # elif plobject.name == 'wolf':
    #     clevs = np.arange(0,20.1, 0.2)
    # else:
    #     print('Check Planet object name')
    if float(plobject.rotperiod) < 2.0:
        clevs = np.arange(0.0, 14.1, 1.0)
    elif (float(plobject.rotperiod) >= 2.0) and (float(plobject.rotperiod) < 4.0):
        clevs = np.arange(0.0, 3.1, 0.2)
    elif (float(plobject.rotperiod) >= 4.0) and (float(plobject.rotperiod) < 13.0):
        clevs = np.arange(0.0, 18.1, 1.0)
    elif float(plobject.rotperiod) >= 13.0:
        clevs = np.arange(0.0, 10.1, 1.0)

    if pplot == True:

        [r, th] = np.meshgrid(heights, np.radians(np.arange(0,64)*5.75))
        fig = plt.figure(figsize=(4,5))
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(+1)
        ax.set_rorigin(1200)
        
        contf = ax.contourf(th, r, limb.T, levels=clevs, cmap='plasma')
        ax.set_xticklabels(['90N','45N','eq.','45S','90S','45S','eq.','45N'])
        ax.set_rlim(bottom=heights[-1], top=heights[0]+1)
        ax.set_title('Differential optical depth per 1000 km \n at planetary limb',
                     fontsize=fsize)
        cbar = plt.colorbar(contf, pad=0.15, orientation='vertical', 
                            fraction=0.05)
        cbar.set_label(r'$\frac{d\tau}{ds}$', loc='center',fontsize=fsize-2)
        rlabels = ax.get_ymajorticklabels()
        for label in rlabels:
            label.set_color('white')
        if saveformat == 'eps':
            fig.tight_layout()
        if save == True:
            plt.savefig(plobject.savepath + savename, format=saveformat,
                        bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return west, east, limb, heights    
    
### Old functions, maybe delete after fully cannibalised    

def import_data(path):
    
    filelist = sorted(os.listdir(path))
    filetype = os.path.splitext(filelist[0])[-1]
    file_ext = filetype[1:]
    
    if file_ext=='npz':
        print('Processing npz files')
        allfiles = []
        for file in filelist:
            filedata = np.load(path+file)
            allfiles.append(filedata)
        print(allfiles)
        
    elif file_ext == 'nc':
        print('Processing nc files')
        
        allfiles = []
        
        for file in filelist:
            fullpath = os.path.join(path, file)
            allfiles.append(fullpath)
        print(allfiles)
    
    return allfiles

def import_planet(path, cubename ='mmr'):
    
    filelist = sorted(os.listdir(path))
    nameslist = [file[:-4] for file in filelist]
    print(nameslist)
    
    alldata = []
    for file in filelist:
        filedata = np.load(path+file)
        cube = filedata[cubename]
        alldata.append(cube)
    
    return alldata

    
      


def compare_years(yearlist, startyear=0, proflat=16, proflon=32, clevel=8):
    
    """ Args for proflon:
        0: Antistellar point
        15: Western terminator
        31: Substellar point
        47: Eastern terminator
        63: Antistellar point again """
    
    ref_year = yearlist[0]
    levels = ref_year['lev']
    plevels = ref_year['levp']
    lats = ref_year['lat']
    lons = np.roll(ref_year['lon'],32)
    surfpress = np.mean(ref_year['ps'], axis=0)
    
    mmrs = []
    for year in yearlist:       
        data = np.mean(year['mmr'], axis=0)
        mmrs.append(data)    
        
    titlelat = int(np.round(lats[proflat],0))
    titlelon = int(np.round(lons[proflon],0))
    titlestart = int(startyear+1)
    titleend = int(startyear + len(yearlist))
    
    fig, ax = plt.subplots(figsize=(7,5))
    counter = startyear
    for year in mmrs:
        counter +=1
        plt.plot(year[:,proflat,proflon], levels*surfpress[proflat,proflon],
                 label=f'Year {counter}')

    plt.gca().invert_yaxis()
    plt.title(f'Annual mean aerosol profile at lat {titlelat}, lon {titlelon}, year {titlestart}-{titleend}',
              fontsize=14)
    plt.xlabel('Mass mixing ratio [kg/kg]', fontsize=14)
    plt.ylabel('Pressure [mbar]', fontsize=14)
#    plt.xlim(0,1e-13)
    plt.legend(fontsize='small')
    plt.show()

    return

def surface_temp(plobject, fsize=14):
    
    if plobject.name == 'trap':
        clevs = np.arange(160,320,5)
    elif plobject.name == 'wolf':
        clevs = np.arange(225,346,5)

    
    temperature = np.mean(plobject.data['ts'],0)
    
    plt.contourf(plobject.lon, plobject.lat, temperature, levels=clevs,
                 cmap='hot')
    plt.title('Surface temperature')
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.xticks((plobject.lon[0], plobject.lon[15], plobject.lon[31], 
                plobject.lon[47], plobject.lon[63]), 
                ('180W', '90W', '0','90E','180E'))
    plt.yticks((plobject.lat[0], plobject.lat[7], plobject.lat[15], 
                plobject.lat[23], plobject.lat[31]),
                ('90N', '45N', '0', '45S', '90S'))
    cbar = plt.colorbar(orientation='vertical', fraction=0.05)
    cbar.ax.set_title('K', fontsize=fsize-2)
    plt.show()   
    
    return
    


def wind_vectors(plobject, time_slice=-1, level=5, n=2, qscale=10, meaning=True,
                 save=False, savename='wind_vectors', saveformat='png', fsize=14):
    
    data = plobject.data
    if meaning==True:
        u = np.mean(data['ua'],axis=0) # Eastward wind
        v = np.mean(data['va'],axis=0) # Northward wind
        w = np.mean(data['wa'],axis=0) # Upward wind
        titletime = 'long-term mean'

    else:
        u = data['ua'][time_slice,:,:,:]
        v = data['va'][time_slice,:,:,:]
        w = data['wa'][time_slice,:,:,:]
        titletime = 'final'

    nlon = len(data['lon'])
    nlat = len(data['lat'])
    levels = data['lev']
    surfpress = np.mean(data['ps'], axis=0)
    heights = levels*surfpress[15,31]
    titleterm = 'Horizonal and vertical wind'
    
    X, Y = np.meshgrid(np.arange(0, len(plobject.lon)), np.arange(0, len(plobject.lat)))
    #np.meshgrid(np.arange(0, nlon), np.arange(0, nlat))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    w = ax.contourf(w[level,:,:]*1e03, cmap='coolwarm', 
                 levels=np.arange(-0.25, 1.26, 0.05), norm=TwoSlopeNorm(0))
    cbar = plt.colorbar(w, orientation='vertical', fraction=0.05)
    cbar.set_label('Vertical wind [$10^{-3}$ m/s]', loc='center')
    
    q1 = ax.quiver(X[::n, ::n], Y[::n, ::n], u[level, ::n, ::n],
                   -v[level, ::n, ::n], scale_units='xy', scale=qscale)
    ax.quiverkey(q1, X=0.9, Y=1.05, U=qscale*2, label='%s m/s' %str(qscale*2),
                 labelpos='E', coordinates='axes')
    plt.title('%s, %s \n %s mbar' %
              (titleterm, titletime, np.round(heights[level],0)), fontsize=fsize)
    plt.xticks((0, 15, 31, 47, 63), 
                ('180W', '90W', '0','90E','180E'))
    plt.yticks((0, 7, 15, 23, 31 ),
                ('90S', '45S', '0', '45N', '90N'))
    plt.xlabel('Longitude [deg]', fontsize=fsize)
    plt.ylabel('Latitude [deg]', fontsize=fsize)
    ax.axis('tight')
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat,
                    bbox_inches='tight')   
        plt.close()
    else:
        plt.show()
    
    return


    
def cloud_cover(data, time_slice=-1, level=5, meaning=False, total_cloud=False):
    
    lats = data['lat']
    lons = data['lon']
    levels = data['lev']
    
    surfpress = np.mean(data['ps'], axis=0)
    heights = levels*surfpress[15,31]
    
    if total_cloud == True:
        cloud_data = data['clt']
        titleterm = 'total'
    else:
        cloud_data = data['cl'][:,level,:,:]
        titleterm = str(np.round(heights[level],0)) + ' mbar'
        
    if (meaning == True and total_cloud == True):
        cloud_data = np.mean(cloud_data,axis=0)
    else:
        cloud_data = cloud_data[time_slice,:,:]            
    
    plt.contourf(lons, lats, cloud_data, 
                 cmap='Blues')
    plt.title(f'Cloud area fraction, {titleterm}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    cbar = plt.colorbar()
#    cbar.ax.set_ylim(0,1)
    plt.show()    
    
    return


def vert_profile(plobject, time_slice=-1, select='t', meaning=True,
                 save=False, savename='profiles.png', saveformat='png',
                 fsize=14):
    plobject.add_rhog()
    data = plobject.data
    lats = data['lat']
    lons = data['lon']
    levels = data['lev']
    
    surfpress = np.mean(data['ps'], axis=0)
    heights = levels*surfpress[15,31]
    
    if select == 't':
        cube = data['ta']
        titleterm = 'Temperature'
        unit = 'K'
        limits = (0,320)
    elif select == 'v':
        cube = data['hus']
        titleterm = 'Vapour'
        unit = 'kg/kg'
        if plobject.name == 'trap':
            limits = (0, 0.008)
        elif plobject.name == 'wolf':
            limits = (0, 0.2)
        else:
            print('Check Planet object name')
    elif select == 'mmr':
        cube = data['mmr']*plobject.rhog*1e09
        titleterm = 'Haze mass loading'
        unit = '$10^{-9}$ kg/m$^3$'
        if plobject.name == 'trap':
            limits = (-0.8, 5)
        elif plobject.name == 'wolf':
            limits = (-0.8, 5)
        else:
            print('Check Planet object name')
    elif select == 'no':
        cube, nsource = mmr2n(plobject, item='mmr')
        titleterm = 'Haze number density \n'
        unit = 'particles/m$^3$'
        if plobject.name == 'trap':
            limits = (-1e06, 5e06)
        elif plobject.name == 'wolf':
            limits = (-1e06, 5e06)
        else:
            print('Check Planet object name')
    elif select == 'dtdt':
        cube = data['dtdt']
        titleterm = 'Radiative heating rate'
        unit = 'K/s'
        limits = (-8e-05, 1.0e-05)
    elif select == 'vterm':
        cube = (data['wa'] + plobject.vterm)*1e04
        titleterm = 'Net particle velocity'
        unit = '$10^{-4}$ m/s'
        if plobject.name == 'trap':
            limits = (-3.5, 1.0)
        elif plobject.name == 'wolf':
            limits = (-3.5, 1.0)
    elif select == 'w':
        cube = (data['wa'])*1e04
        titleterm = 'Vertical velocity'
        unit = '$10^{-4}$ m/s'
        if plobject.name == 'trap':
            limits = (-3.5, 1.0)
        elif plobject.name == 'wolf':
            limits = (-3.5, 1.0)
    else:
        print('Select t for air temperature, v for specific humidity, or \
              mmr for mass mixing ratio.')
    
    if meaning == True:
        plotme = np.mean(cube, axis=0)
    else:
        plotme = cube[time_slice,:,:,:]
        
    daylist = []
    nightlist = []
    limblist = []
    westlist = []
    eastlist = []
    for l in range(0, len(levels)):
        dayside = np.sum(plotme[l,:,15:48]*plobject.area[:,15:48]) /    \
                  np.sum(plobject.area[:,15:48])
        daylist.append(dayside)
        nightside = (np.sum(plotme[l,:,48:]*plobject.area[:,48:]) +     \
                     np.sum(plotme[l,:,:16]*plobject.area[:,:16])) /    \
                    (np.sum(plobject.area[:,48:]) + np.sum(plobject.area[:,:16]))
        nightlist.append(nightside)
        limb = (np.sum(plotme[l,:,15]*plobject.area[:,15]) +             \
                np.sum(plotme[l,:,47]*plobject.area[:,47])) /            \
                (np.sum(plobject.area[:,15]) + np.sum(plobject.area[:,47]))
        limblist.append(limb)
        west = np.sum(plotme[l,:,15]*plobject.area[:,15])/              \
              (np.sum(plobject.area[:,15]))
        westlist.append(west)
        east = np.sum(plotme[l,:,47]*plobject.area[:,47])/              \
              (np.sum(plobject.area[:,47]))
        eastlist.append(east)

    fig, ax = plt.subplots(figsize=(4,5))    
#    plt.plot(plotme[:,16,32], heights, color='k', label='Substellar')
#    plt.plot(plotme[:,16,0], heights, color='m', label='Antistellar')
    plt.plot(np.array(daylist), heights, color='k', linestyle='dashed', label='Dayside mean')
    plt.plot(np.array(nightlist), heights, color='m', linestyle='dashed', label='Nightside mean')
    plt.plot(np.array(limblist), heights, color='r', linestyle='dashed', label='Terminator mean')
    plt.plot(np.array(westlist), heights, color='g', label='West terminator mean')
    plt.plot(np.array(eastlist), heights, color='b', label='East terminator mean')
    plt.gca().invert_yaxis()
    plt.xlim(limits)

    plt.title(f'{titleterm} profiles', fontsize=fsize) 
    plt.ylabel('Pressure [mbar]', fontsize=fsize)
    plt.xlabel(f'{titleterm} [{unit}]', fontsize=fsize)
    plt.legend(fontsize='small')
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + select + '_' + savename, 
                    format=saveformat, 
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()  
    
    return

def zmzw(plobject, time_slice=-1, meaning=True, save=False, 
         savename='zmzw.png', saveformat='png', fsize=14):
    data = plobject.data
    lats = data['lat']
    levels = data['lev']
    surfpress = np.mean(data['ps'], axis=0)
    heights = levels*surfpress[15,31]
    
    cube = data['ua']
    
    if meaning == True:
        plotme = np.mean(cube, axis=0)
    else:
        plotme = cube[time_slice,:,:,:]
        
    zmu = np.mean(plotme, axis=2)
    
    # if plobject.name == 'trap':
    #     clevs = np.arange(-25, 75, 5)
    # elif plobject.name == 'wolf':
    #     clevs = np.arange(-40, 135, 5)
    if float(plobject.rotperiod) < 2.0:
        clevs = np.arange(-20, 41)
    elif (float(plobject.rotperiod) >= 2.0) and (float(plobject.rotperiod) < 4.0):
        clevs = np.arange(-40, 71)
    elif (float(plobject.rotperiod) >= 4.0) and (float(plobject.rotperiod) < 13.0):
        clevs = np.arange(-10, 131)
    elif float(plobject.rotperiod) >= 13.0:
        clevs = np.arange(-10, 81)
    
    fig, ax = plt.subplots(figsize=(5,5))
    plt.contourf(lats, heights, zmu, cmap='RdBu_r', 
                 levels=clevs, norm=TwoSlopeNorm(0))
    plt.gca().invert_yaxis()
    plt.title('Zonal mean zonal wind', fontsize=fsize)
    plt.xlabel('Latitude [degrees]', fontsize=fsize)
    plt.ylabel('Pressure [mbar]', fontsize=fsize)
    cbar = plt.colorbar(orientation='vertical', fraction=0.05, ticks=clevs[::10])
    cbar.ax.set_title('m/s')
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat, 
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def zmzwdiff(plobject, cobject, time_slice=-1, meaning=True, save=False, 
         savename='zmzw.png', saveformat='png', fsize=14):
    data = plobject.data
    cdata = cobject.data
    lats = data['lat']
    levels = data['lev']
    surfpress = np.mean(data['ps'], axis=0)
    heights = levels*surfpress[15,31]
    
    cube = data['ua']
    ccube = cdata['ua']
    
    if meaning == True:
        plotme = np.mean(cube, axis=0)
        plotcontour = np.mean(ccube, axis=0)
    else:
        plotme = cube[time_slice,:,:,:]
        plotcontour = ccube[time_slice,:,:,:]
        
    zmu = np.mean(plotme, axis=2)
    zmuc = np.mean(plotcontour, axis=2)
    zmudiff = np.abs(zmu) - np.abs(zmuc)
    
    # if plobject.name == 'trap':
    #     clevs = np.arange(-25, 75, 5)
    # elif plobject.name == 'wolf':
    #     clevs = np.arange(-40, 135, 5)
    if float(plobject.rotperiod) < 2.0:
        clevs = np.arange(-20, 41)
    elif (float(plobject.rotperiod) >= 2.0) and (float(plobject.rotperiod) < 4.0):
        clevs = np.arange(-40, 71)
    elif (float(plobject.rotperiod) >= 4.0) and (float(plobject.rotperiod) < 13.0):
        clevs = np.arange(-10, 131)
    elif float(plobject.rotperiod) >= 13.0:
        clevs = np.arange(-10, 81)
    
    fig, ax = plt.subplots(figsize=(5,5))
    CF = plt.contourf(lats, heights, zmu, cmap='RdBu_r', 
                 levels=clevs, norm=TwoSlopeNorm(0))
    CL = plt.contour(lats, heights, zmudiff, 
#                 levels=np.arange(np.floor(np.min(zmudiff)), np.ceil(np.max(zmudiff)), 1),
                 levels=[-30, -20, -10, -5, -3, -1, 1, 3, 5, 10, 20, 30],
                 colors='k', negative_linestyles='dashed', 
                 linewidths=0.5, alpha=0.5)
    plt.gca().invert_yaxis()
    plt.title('Zonal mean zonal wind', fontsize=fsize)
    plt.xlabel('Latitude [degrees]', fontsize=fsize)
    plt.ylabel('Pressure [mbar]', fontsize=fsize)
    cbar = plt.colorbar(CF, orientation='vertical', fraction=0.05, ticks=clevs[::10])
    cbar.ax.set_title('m/s')
    ax.clabel(CL, fontsize=9, inline=True)
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + savename, format=saveformat, 
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        

def vterm(plobject, time_slice=-1, level=0, meaning=True, fsize=14,
          saveformat='png', save=False, savename='_vterm.png'):
    
    data = plobject.data
    levels = data['lev']
    surfpress = np.mean(data['ps'], axis=0)
    heights = np.round(levels*surfpress[15,31], 0)
    vel = plobject.vterm
    
    if meaning==True:
        w = np.mean(data['wa'],axis=0) # Upward wind
        t = np.mean(vel, axis=0) # Terminal velocity
        titletime = 'long-term mean'

    else:
        w = data['wa'][time_slice,:,:,:]
        t = np.mean(vel, axis=0)
        titletime = 'final'
      
    if plobject.name == 'trap':
        clevs = np.arange(-6.0, 12.0, 0.2)
    elif plobject.name == 'wolf':
        clevs = np.arange(-6.0, 12.0, 0.2)
        
    balance = (w + t)*1e04
    fig, ax = plt.subplots(figsize=(8,5))
    plt.contourf(plobject.lon, plobject.lat, balance[level,:,:], cmap='RdBu_r', 
                 levels=clevs, norm=TwoSlopeNorm(0))
    plt.title('Net particle velocity at %s mbar, %s' %(heights[level],titletime),
              fontsize=fsize)
    cbar = plt.colorbar(orientation='vertical', fraction=0.05)
    cbar.ax.set_title('$10^{-4} $m/s') 
    plt.xlabel('Longitude [deg]', fontsize=fsize)
    plt.ylabel('Latitude [deg]', fontsize=fsize)
    plt.xticks((plobject.lon[0], plobject.lon[15], plobject.lon[31], 
                plobject.lon[47], plobject.lon[63]), 
                ('180W', '90W', '0','90E','180E'))
    plt.yticks((plobject.lat[0], plobject.lat[7], plobject.lat[15], 
                plobject.lat[23], plobject.lat[31]),
                ('90N', '45N', '0', '45S', '90S'))
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + 'vmap_' + savename, format=saveformat, 
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show() 
        

    fig, ax = plt.subplots(figsize=(5,5))    
    plt.plot(balance[:,15,31], heights, color='k', label='Substellar')
    plt.plot(balance[:,15,0], heights, color='m', label='Antistellar')
    plt.plot(balance[:,15,15], heights, color='r', label='West terminator eq.')
    plt.plot(balance[:,15,47], heights, color='b', label='East terminator eq.')
    plt.plot(balance[:,3,47], heights, color='r', linestyle='dashed', 
                                                     label='South gyre')
    plt.plot(balance[:,28,48], heights, color='b', linestyle='dashed', 
                                                     label='North gyre')
    plt.gca().invert_yaxis()
    plt.xlim((-2.5, 12.0))

    plt.title('Net particle velocity profiles', fontsize=fsize) 
    plt.ylabel('Pressure [mbar]', fontsize=fsize)
    plt.xlabel('$10^{-4} $m/s', fontsize=fsize)
    plt.legend()
    if saveformat == 'eps':
        fig.tight_layout()
    if save == True:
        plt.savefig(plobject.savepath + 'vprof_' + savename, format=saveformat, 
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show() 
    
    return

def wxsection(plobject, time_slice=-1, meaning=True, save=False, 
         savename='zmzw.png', saveformat='png', fsize=14):
    
    levels = plobject.data['lev']
    surfpress = np.mean(plobject.data['ps'], axis=0)
    heights = np.round(levels*surfpress[15,31], 0)
    
    if meaning == True:
        w = np.mean(plobject.data['wa'], axis=0)*1e04
        titleterm = 'long-term mean'
    else:
        w = plobject.data['wa'][time_slice,:,:,:]*1e04
        
    clevs = np.arange(-5,15,1)    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8,5))
    fig.suptitle('Vertical wind cross-sections', fontsize=fsize)
    im1 = ax1.contourf(plobject.lon, heights, w[:,15,:], levels=clevs,
                       cmap='RdBu_r', norm=TwoSlopeNorm(0))
    ax1.set_xlabel('Longitude [deg]')
    ax1.set_ylabel('Pressure [mbar]')
    im2 = ax2.contourf(plobject.lat, heights, w[:,:,31], levels=clevs,
                       cmap='RdBu_r', norm=TwoSlopeNorm(0))
    ax2.set_xlabel('Latitude [deg]')
    plt.gca().invert_yaxis()
    cbar = fig.colorbar(im2, ax=(ax1, ax2), orientation='vertical', 
                        fraction=0.05)
    cbar.ax.set_title('$10^{-4} $m/s')
    plt.show()

    