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

# k2dict = {'radius' : 2.37, 'solcon' : 1722, 'startemp' : 3518, 'N2' : 0.0,
#             'CO2' : 0.0, 'H2' : 1.0, 'rotperiod' : 33., 'gravity' : 16.3,
#             'starrad': 0.42, 'starspec' : top_dir + 'stellarspectra/k2.dat',
#             'name' : 'K2-18b', 'eccentricity': 0.0, 'bulk' : 2, 'msource' : 1e-13}

# teedict = {'radius' : 1.02, 'solcon' : 1572, 'startemp' : 2997, 'N2' : 0.9996,
#             'CO2' : 0.000400, 'H2' : 0.0, 'rotperiod' : 4.91, 'gravity' : 9.9,
#             'starrad' : 0.11, 'starspec' : top_dir + 'stellarspectra/teegarden.dat',
#             'name' : 'Teegardens Star b', 'eccentricity': 0.0, 'bulk' : 1, 'msource' : 1e-13}

# kep452dict = {'radius' : 1.63, 'solcon' : 1504, 'startemp' : 5635, 'N2' : 0.9996,
#             'CO2' : 0.000400, 'H2' : 0.0, 'rotperiod' : 385, 'gravity' : 12.2,
#             'starrad' : 1.0, 'starspec' : top_dir + 'stellarspectra/kep452.dat',
#             'name' : 'Kepler-452b', 'eccentricity': 0.0, 'bulk' : 1, 'msource' : 1e-13}

# kep1649dict =  {'radius' : 1.06, 'solcon' : 1025, 'startemp' : 3383, 'N2' : 0.9996,
#             'CO2' : 0.000400, 'H2' : 0.0, 'rotperiod' : 19.5, 'gravity' : 10.5,
#             'starrad' : 0.24, 'starspec' : top_dir + 'stellarspectra/kep1649.dat',
#             'name' : 'Kepler-1649c', 'eccentricity': 0.0, 'bulk' : 1, 'msource' : 1e-13}

wolfdict = {'radius' : 1.66, 'solcon' : 1777, 'startemp' : 3408, 'N2' : 0.988,
            'CO2' : 0.01, 'CH4' : 0.002, 'H2' : 0.0, 'rotperiod' : 17.9, 'gravity' : 12.1,
            'starrad' : 0.32, 'starspec' : top_dir + 'stellarspectra/wolf.dat',
            'longname' : 'Wolf-1061c', 'eccentricity': 0.0, 'bulk' : 1,
            'datapath' : '/exports/csce/datastore/geos/users/s1144983/exoplasim_data/wolf.npz',
            'msource' : 1e-07,'name' : 'wolf',
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
            setattr(Planet, key, value) 
        
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
            setattr(Planet, key, value)
            
    def save_files(self):
        """ Save planetdata into a single npz file"""
        np.savez(top_dir + self.name + '.npz', 
                **{name:value for name,value in zip(self.contents, self.data.values())})
    
    def load_data(self, filepath):
        """ Load a single preprocessed npz file with all data"""
        planetdata = dict(np.load(filepath))
        planetnames = list(planetdata.keys())
        self.data = planetdata
        self.contents = planetnames
        for key, value in planetdata.items():
            setattr(Planet, key, value)
            
    def add_rhog(self):   
        """ Calculate density of air and add to planet sim data"""
        rhog = (self.flpr*0.028)/(8.314*self.ta) # (time, height, lat, lon)
        self.rhog = rhog
        
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
        
    def mmr_list(self):
        """ Make list of mmr data cubes only"""
        mmrs = [item for item in self.contents if 
                item.split('_')[0] == self.name]
        self.mmrs = mmrs
        
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
    
def plot_lonlat(plobject, cube, title = 'Planet', unit='kg m$^{-2}$', 
                colors='plasma', save=False, savename='plotname.png', 
                saveformat='png'):
    """ Make a lon-lat contourfill plot of the input data cube"""
    plt.contourf(plobject.lon, plobject.lat, cube, cmap=colors)
    plt.title(title + '\n' + 'Total haze column excluding source',
              fontsize=14)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    cbar = plt.colorbar()
    cbar.set_label(unit, fontsize=10)
    if save:
        plt.savefig(plobject.savepath + savename, format=saveformat)
    plt.show()
    
def limb_mass(plobject, inputcol):
    """ Calculate total haze mass at planetary limb"""
    
    limbsum = (np.sum(inputcol[:,16]*plobject.area[:,16]) + \
                np.sum(inputcol[:,48]*plobject.area[:,48]))
       
    return limbsum

def distribution(plobject, inputlists, save=False,
                savename='plotname.png', saveformat='png'):
    """ Plot total integrated haze at terminator against particle size"""
    
    fig, ax = plt.subplots(figsize=(6,4))    
    plt.scatter(np.array(plobject.radii), np.array(inputlists[0]), 
            c='g', label='1000 kg/m$^3$')
    plt.scatter(np.array(plobject.radii), np.array(inputlists[1]), 
            c='r', label='1262 kg/m$^3$')
    plt.scatter(np.array(plobject.radii), np.array(inputlists[2]), 
            c='b', label='1328 kg/m$^3$')
    plt.title('Total haze column at the planetary limb')
    plt.xlabel('Particle radius [m]')
    plt.ylabel('kg')
#        plt.yscale('log')
    plt.xscale('log')
    if save:
        plt.savefig(plobject.savepath + savename, format=saveformat)
    plt.show()
    
def distribution_scatter(plobject, inputlists, save=False,
                savename='plotname.png', saveformat='png'):
    """ Plot total integrated haze at terminator against particle size"""
    
    fig, ax1 = plt.subplots(figsize=(6,4))
    line1 = ax1.scatter(np.array(plobject.radii), np.array(inputlists[0]), 
            c='g', label='1000 kg/m$^3$')
    line2 = ax1.scatter(np.array(plobject.radii), np.array(inputlists[1]), 
            c='r', label='1262 kg/m$^3$')
    line3 = ax1.scatter(np.array(plobject.radii), np.array(inputlists[2]), 
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
    lns = line1+line2+line3+line4+line5
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs)
    plt.title('Total haze column at planetary limb \n and extinction efficiency')

    if save:
        plt.savefig(plobject.savepath + savename, format=saveformat)
    plt.show()

def distribution_norm(plobject, inputlist, dens=1, save=False,
                savename='plotname.png', saveformat='png'):
    """ Plot total integrated haze at terminator against particle size"""
    
    sw = np.array(plobject.sw1 + plobject.sw2)
    normed_sw = normalise(sw)
    normed_kg = normalise(np.array(inputlists[dens]))

    fig, ax = plt.subplots(figsize=(6,4))
    plt.plot(np.array(plobject.radii), 
             normed_kg*normed_sw, c='b')
    plt.set_ylabel('Effect size')
    plt.set_xlabel('Particle radius [m]')
    plt.set_xscale('log')
    plt.title('Effect size of haze by particle size')
    if save:
        plt.savefig(plobject.savepath + savename, format=saveformat)
    plt.show()    
    
def compare_profiles(plobject, pdensity=1262, proflat=16, proflon=48,
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
    if save:
        plt.savefig(plobject.savepath + savename, format=saveformat)
    plt.show()
    
    return
    
def mmr_map(plobject, item, level=4, cpower=7, 
            save=False, savename='plotname.png', 
            saveformat='png'):
    
    coeff = 10**cpower
    
    fig, ax = plt.subplots(figsize=(7,5))    
    plt.contourf(plobject.lon, plobject.lat, 
                np.mean(plobject.data[item][:,level,:,:],axis=0)*coeff,
#                 np.arange(0,1.05,0.1),
                 cmap='plasma')
    plt.title('Mass mixing ratio at %s mbar' % 
              (np.round(np.mean(plobject.flpr[:,level,16,32], axis=0)/100)))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    cbar = plt.colorbar()
    cbar.ax.set_title('$10^{-7}$ kg/kg', size=10)
    fig.tight_layout()
    plt.show()  
    
    return
    
def compare_planets(plobjects, ndata=3, level=5, n=2, qscale=10):
    """ Compare climatology of all planets in parameter space"""
    names = []
    for plobject in plobjects:
        plname = plobject.longname
        names.append(plname)
    redblu = mpl_cm.get_cmap('coolwarm')
    hot = mpl_cm.get_cmap('hot')
    
    fig, ax = plt.subplots(figsize=(16, 22), nrows=5, ncols=ndata)
    for i in range(ndata):
        plobject = plobjects[i]
        #Surface temperature
        surf = ax[0, i].contourf(plobject.lon, plobject.lat,
               np.mean(plobject.ts, axis=0), 
               levels=np.arange(100, 400, 20),
               cmap=hot)
        ax[0,i].set_title(f'{plobject.longname}', fontsize=14)
        fig.colorbar(surf, orientation='vertical')#        
        ax[0, i].set_xlabel('Longitude [deg]', fontsize=14)
        ax[0, i].set_ylabel('Surface temperature [K] \n Latitude [deg]', fontsize=14)
        # Air temperature profile
        ax[1, i].plot(np.mean(plobject.ta[:,:,16,32], axis=0),
                      np.mean(plobject.flpr[:,:,16,32]/100,axis=0),
                      color='r', label='Substellar')
        ax[1, i].plot(np.mean(plobject.ta[:,:,16,0], axis=0),
                      np.mean(plobject.flpr[:,:,16,0]/100,axis=0),
                      color='b', label='Antistellar')
        ax[1, i].legend()
        ax[1, i].set_xlabel('Temperature [K]', fontsize=14)
        ax[1, i].set_ylabel('Temperature profile \n Pressure [mbar]', fontsize=14)
        # Water vapour profile
        ax[2, i].plot(np.mean(plobject.hus[:,:,16,32], axis=0),
                      np.mean(plobject.flpr[:,:,16,32]/100,axis=0),
                      color='r', label='Substellar')
        ax[2, i].plot(np.mean(plobject.hus[:,:,16,0], axis=0),
                      np.mean(plobject.flpr[:,:,16,0]/100,axis=0),
                      color='b', label='Antistellar')
        ax[2, i].legend()
        ax[2, i].set_xlabel('Specific humidity [kg/kg]', fontsize=14)
        ax[2, i].set_ylabel('Vapour profile \n Pressure [mbar]', fontsize=14)
        # ZMZW
        cont = ax[3, i].contourf(plobject.lat, np.mean(plobject.flpr[:,:,16,32], axis=0), 
               np.mean(plobject.ua, axis=(0,3)), levels=np.arange(-60, 140, 20), 
               cmap=redblu, norm=TwoSlopeNorm(0))
        ax[3, i].set_ylabel('Zonal mean zonal wind [m/s] \n Pressure [mbar]', fontsize=14)
        ax[3, i].set_xlabel('Latitude [degrees]', fontsize=14)
        fig.colorbar(cont, orientation='vertical')
        # Winds
        X, Y = np.meshgrid(np.arange(0, len(plobject.lon)), np.arange(0, len(plobject.lat)))
        q1 = ax[4, i].quiver(X[::n, ::n], Y[::n, ::n], 
                            np.mean(plobject.ua[:, level, ::n, ::n], axis=0),
                            -np.mean(plobject.va[:, level, ::n, ::n], axis=0),
                            scale_units='xy', scale=qscale)
        ax[4, i].quiverkey(q1, X=0.9, Y=1.05, U=qscale*2, label='%s m/s' %str(qscale*2),
                     labelpos='E', coordinates='axes')
        ax[4, i].set_xticks((0, 16, 32, 48, 64), 
                    ('180W', '90W', '0','90E','180E'))
        ax[4, i].set_yticks((0, 8, 16, 24, 32 ),
                    ('90S', '45S', '0', '45N', '90N'))
        ax[4, i].set_xlabel('Longitude [deg]', fontsize=14)
        ax[4, i].set_ylabel('Horizontal and vertical winds \n Latitude [deg]', fontsize=14)
    plt.show()






    
def mmr2n(plobject, item):
    """ Convert mass mixing ratio (kg/kg) to number density (particles/m3)"""
    mmr_raw = plobject.data[item] # in kg/kg
    
    coeff, power = item[-5], item[-2:]
    particle_rad = float(coeff + 'e-' + power)
    particle_den = float(item[-10:-6])
    sphere_vol = (4/3)*np.pi*(particle_rad**3)
    particle_mass = sphere_vol*particle_den
    
    outcube = mmr_raw*(1/particle_mass)*plobject.rhog # particles/m3
    nsource = plobject.msource*(1/particle_mass)*np.mean(plobject.rhog[:,0,16,32], axis=0)
    return outcube, nsource
    


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
        16: Western terminator
        32: Substellar point
        48: Eastern terminator
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

def surface_temp(data):
    
    # zmin, zmax, zstep = 260,420,20
    
    levels = data['lev']
    lats = data['lat']
    lons = data['lon']
    
    temperature = np.mean(data['ts'],0)
    
    plt.contourf(lons, lats, temperature,
                 cmap='hot')
    plt.title('Surface temperature')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    cbar = plt.colorbar()
    cbar.ax.set_title('K')
    plt.show()   
    
    return


def wind_vectors(data, time_slice=-1, level=5, n=2, qscale=10, meaning=True):
    
    if meaning==True:
        u = np.mean(data['ua'],axis=0) # Eastward wind
        v = np.mean(data['va'],axis=0) # Northward wind
        w = np.mean(data['wa'],axis=0) # Upward wind
    else:
        u = data['ua'][time_slice,:,:,:]
        v = data['va'][time_slice,:,:,:]
        w = data['wa'][time_slice,:,:,:]

    nlon = len(data['lon'])
    nlat = len(data['lat'])
    levels = data['lev']
    surfpress = np.mean(data['ps'], axis=0)
    heights = levels*surfpress[16,32]
    titleterm = 'Horizonal wind'
    titletime = 'long-term mean'
    
    X, Y = np.meshgrid(np.arange(0, nlon), np.arange(0, nlat))
    
    fig, ax = plt.subplots(figsize=(8.5, 5))
    plt.imshow(w*1e02, cmap='RdBu_r', origin='lower')
    cbar = plt.colorbar()
    cbar.set_label('$10^{-2$ m/s', loc='center')
    
    q1 = ax.quiver(X[::n, ::n], Y[::n, ::n], u[level, ::n, ::n],
                   -v[level, ::n, ::n], scale_units='xy', scale=qscale)
    ax.quiverkey(q1, X=0.9, Y=1.05, U=qscale*2, label='%s m/s' %str(qscale*2),
                 labelpos='E', coordinates='axes')
    plt.title('%s, %s, h=%s mbar' %
              (titleterm, titletime, np.round(heights[level],0)), fontsize=14)
    plt.xticks((0, 16, 32, 48, 64), 
                ('180W', '90W', '0','90E','180E'))
    plt.yticks((0, 8, 16, 24, 32 ),
                ('90S', '45S', '0', '45N', '90N'))
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.show()
    
    return


    
def cloud_cover(data, time_slice=-1, level=5, meaning=False, total_cloud=False):
    
    lats = data['lat']
    lons = data['lon']
    levels = data['lev']
    
    surfpress = np.mean(data['ps'], axis=0)
    heights = levels*surfpress[16,32]
    
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


def tvprofile(data, time_slice=-1, select='t', meaning=True):
    
    lats = data['lat']
    lons = data['lon']
    levels = data['lev']
    
    surfpress = np.mean(data['ps'], axis=0)
    heights = levels*surfpress[16,32]
    
    if select == 't':
        cube = data['ta']
        titleterm = 'Temperature'
        unit = 'K'
    elif select == 'v':
        cube = data['hus']
        titleterm = 'Vapour'
        unit = 'kg/kg'
    else:
        print('Select t for air temperature or v for specific humidity.')
    
    if meaning == True:
        plotme = np.mean(cube, axis=0)
    else:
        plotme = cube[time_slice,:,:,:]
        
    plt.plot(plotme[:,16,32], heights, label='Substellar')
    plt.plot(plotme[:,16,0], heights, label='Antistellar')
    plt.plot(plotme[:,16,16], heights, label='Western terminator')
    plt.plot(plotme[:,16,48], heights, label='Eastern terminator')
    plt.gca().invert_yaxis()

    plt.title(f'{titleterm} profile', fontsize=14) 
    plt.ylabel('Height [mbar]', fontsize=14)
    plt.xlabel(f'{titleterm} [{unit}]', fontsize=14)
    plt.legend()
    plt.show()  
    
    return

def zmzw(data, time_slice=-1, meaning=True):
    
    lats = data['lat']
    levels = data['lev']
    surfpress = np.mean(data['ps'], axis=0)
    heights = levels*surfpress[16,32]
    
    cube = data['ua']
    
    if meaning == True:
        plotme = np.mean(cube, axis=0)
    else:
        plotme = cube[time_slice,:,:,:]
        
    zmu = np.mean(plotme, axis=2)
    
    plt.contourf(lats, heights, zmu, cmap='RdBu_r', norm=TwoSlopeNorm(0))
    plt.gca().invert_yaxis()
    plt.title('Zonal mean zonal wind', fontsize=14)
    plt.xlabel('Latitude [degrees]', fontsize=14)
    plt.ylabel('Pressure [mbar]', fontsize=14)
    cbar = plt.colorbar()
    cbar.ax.set_title('m/s')
    plt.show()

    