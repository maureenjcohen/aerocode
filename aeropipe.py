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


path1 = '/home/s1144983/aerosim/parameter_space/trapdata/'
path2 = '/home/s1144983/aerosim/parameter_space/k2data/'
path3 = '/home/s1144983/aerosim/parameter_space/wolfdata/'


top_dir = '/home/s1144983/aerosim/parameter_space/'

trapdict = {'radius' : 0.92, 'solcon' : 889, 'startemp' : 2709, 'N2' : 0.9996,
            'CO2' : 0.000400, 'H2' : 0.0, 'rotperiod' : 6.1, 'gravity' : 9.1,
            'starrad' : 0.12, 'starspec' : top_dir + 'stellarspectra/trap.dat', 
            'eccentricity': 0.0, 'bulk' : 1, 'msource' : 1e-13,
            'name' : 'TRAPPIST-1e'}

k2dict = {'radius' : 2.37, 'solcon' : 1722, 'startemp' : 3518, 'N2' : 0.0,
            'CO2' : 0.0, 'H2' : 1.0, 'rotperiod' : 33., 'gravity' : 16.3,
            'starrad': 0.42, 'starspec' : top_dir + 'stellarspectra/k2.dat',
            'name' : 'K2-18b', 'eccentricity': 0.0, 'bulk' : 2, 'msource' : 1e-13}

teedict = {'radius' : 1.02, 'solcon' : 1572, 'startemp' : 2997, 'N2' : 0.9996,
            'CO2' : 0.000400, 'H2' : 0.0, 'rotperiod' : 4.91, 'gravity' : 9.9,
            'starrad' : 0.11, 'starspec' : top_dir + 'stellarspectra/teegarden.dat',
            'name' : 'Teegardens Star b', 'eccentricity': 0.0, 'bulk' : 1, 'msource' : 1e-13}

kep452dict = {'radius' : 1.63, 'solcon' : 1504, 'startemp' : 5635, 'N2' : 0.9996,
            'CO2' : 0.000400, 'H2' : 0.0, 'rotperiod' : 385, 'gravity' : 12.2,
            'starrad' : 1.0, 'starspec' : top_dir + 'stellarspectra/kep452.dat',
            'name' : 'Kepler-452b', 'eccentricity': 0.0, 'bulk' : 1, 'msource' : 1e-13}

kep1649dict =  {'radius' : 1.06, 'solcon' : 1025, 'startemp' : 3383, 'N2' : 0.9996,
            'CO2' : 0.000400, 'H2' : 0.0, 'rotperiod' : 19.5, 'gravity' : 10.5,
            'starrad' : 0.24, 'starspec' : top_dir + 'stellarspectra/kep1649.dat',
            'name' : 'Kepler-1649c', 'eccentricity': 0.0, 'bulk' : 1, 'msource' : 1e-13}

wolfdict = {'radius' : 1.66, 'solcon' : 1777, 'startemp' : 3408, 'N2' : 0.9996,
            'CO2' : 0.000400, 'H2' : 0.0, 'rotperiod' : 17.9, 'gravity' : 12.1,
            'starrad' : 0.32, 'starspec' : top_dir + 'stellarspectra/wolf.dat',
            'name' : 'Wolf-1061c', 'eccentricity': 0.0, 'bulk' : 1,
            'datapath' : '/home/s1144983/aerosim/parameter_space/Wolf-1061c.npz',
            'msource' : 1e-13}

gj667dict = {'radius' : 1.77, 'solcon' : 1202, 'startemp' : 3594, 'N2' : 0.9996,
            'CO2' : 0.000400, 'H2' : 0.0, 'rotperiod' : 28.1, 'gravity' : 11.9,
            'starrad' : 0.42, 'starspec' : top_dir + 'stellarspectra/gj667.dat',
            'name' : 'GJ667Cc', 'eccentricity': 0.2, 'bulk' : 1, 'msource' : 1e-13}



class Planet:
    """ A Planet object which contains the model output data for a planet,
    plus all mmr cubes for the parameter space labelled by the name of each
    simulation in the parameter space."""
    
    def __init__(self, planetdict):
        self.name = planetdict['name']
        print(f'Welcome to {self.name}!')
        for key, value in planetdict.items():
            setattr(Planet, key, value) 
            
        self.load_data(self.datapath)
            
        self.area_weights()
        
        p10 = np.transpose(self.lev*self.ps[...,None], (0, 3, 1, 2))
        rhog = (p10*0.028)/(8.314*self.ta) # (time, height, lat, lon)
        self.rhog = rhog
        
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
            
    def plot_lonlat(self, cube, title, unit, colors='plasma'):
        """ Make a lon-lat contourfill plot of the input data cube"""
        plt.contourf(self.lon, self.lat, cube, cmap=colors)
        plt.title(title + '\n' + 'Total haze column excluding source', fontsize=14)
        plt.xlabel('Longitude', fontsize=14)
        plt.ylabel('Latitude', fontsize=14)
        cbar = plt.colorbar()
        cbar.set_label(unit, fontsize=10)
        plt.show()
        
    def mmr2n(self, item):
        """ Convert mass mixing ratio (kg/kg) to number density (particles/m3)"""
        mmr_raw = self.data[item] # in kg/kg
        
        coeff, power = item[-5], item[-2:]
        particle_rad = float(coeff + 'e-' + power)
        particle_den = float(item[-10:-6])
        sphere_vol = (4/3)*np.pi*(particle_rad**3)
        particle_mass = sphere_vol*particle_den
        
        outcube = mmr_raw*(1/particle_mass)*self.rhog # particles/m3
        nsource = self.msource*(1/particle_mass)*np.mean(self.rhog[:,0,16,32], axis=0)
        return outcube, nsource

    def mmr2kg(self, item):
        """ Convert mass mixing ratio (kg/kg) to mass density (kg/m3)"""
        mmr_raw = self.data[item] # in kg/kg
        rhcube = mmr_raw*self.rhog
        return rhcube
        
    def haze_column(self, item, month=-1, meaning=True, display=True, cubetype='num'):
        """ Calculate total integrated vertical haze column and plot """
        if cubetype=='num':            
            ncube, nsource = self.mmr2n(item=item)
            barunit = 'particles/m2'
            if meaning==True:
                p11 = np.transpose(self.levp*self.ps[...,None], (0, 3, 1, 2))
                dz = np.diff(np.mean(p11,axis=0),axis=0)/(self.gravity*np.mean(self.rhog,axis=0))
                column = np.sum(np.mean(ncube[:,1:,:,:],axis=0)*dz[1:,:,:],axis=0)/nsource
            else:
                p11 = np.transpose(self.levp*self.ps[...,None], (0, 3, 1, 2))
                dz = np.diff(p11[month,:,:,:],axis=0)/(self.gravity*self.rhog[month,:,:,:])
                column = np.sum(ncube[month,1:,:,:]*dz[1:,:,:],axis=0)/nsource
                
        elif cubetype=='mass':
            mcube = self.mmr2kg(item=item)
            barunit = 'kg/m2'
            if meaning==True:           
                pressure = np.transpose(self.levp * \
                            np.mean(self.ps, axis=0)[...,None], (2, 0, 1))
                pressure_thickness = np.diff(pressure, axis=0)
                column = np.sum(np.mean(mcube[:,1:,:,:], axis=0) * \
                          pressure_thickness[1:,:,:], axis=0)
                column /= self.gravity
            else:
                pressure = self.levp[month,:,:,:] * \
                            self.ps[month,:,:,:]
                pressure_thickness = np.diff(pressure, axis=1)
                column = np.sum(mcube[month,:,:,2:] * \
                          pressure_thickness[:,:,2:], axis=0)
                column /= self.gravity
            
        titlestr = f'{self.name}, r={item[-5]}e-{item[-1:]}m, rho={item[-10:-6]} kg/m3'
        if display==True:
            self.plot_lonlat(column, title=titlestr, unit=barunit)
        return column
        
    def all_columns(self, display=True):
        """ Plot total haze column for all simulations"""
        all_keys = self.data.keys()
        mmr_keys = [key for key in all_keys if len(key) > 6]
        mmr_cols = {}
        for key in mmr_keys:
            keycol = self.haze_column(item=key, display=display)       
            case = {str(key) : keycol}
            mmr_cols.update(case)
        return mmr_cols
    
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
        
    def distribution(self):
        """ Plot total integrated haze at terminator against particle size"""
        self.area_weights()
        haze_cols = self.all_columns(display=False)
        
        x_axis = []
        data_list = []
        for key in haze_cols.keys():
            coeff, power = key[-5], key[-2:]
            size_string = coeff + 'e-' + power
            x_axis.append(float(size_string))

            haze_data = haze_cols[key]
            limb_wsum = (np.sum(haze_data[:,16]*self.area[:,16]) + \
                        np.sum(haze_data[:,48]*self.area[:,48]))
            limb_area = np.sum(np.sum(self.area[:,16]) + np.sum(self.area[:,48]))
            limb_mean = limb_wsum/limb_area
            data_list.append(limb_mean)
        color1 = [[1,0,0] for x in range(10)]
        color2 = [[0,1,0] for x in range(10)]
        color3 = [[0,0,1] for x in range(10)]
        colors = np.array(color1+color2+color3)
        
        plt.scatter(np.array(x_axis), np.array(data_list), c=colors)
        plt.title('Mean total haze column at the planetary limb')
        plt.xlabel('Particle radius [m]')
        plt.ylabel('Mean particles/m2')
#        plt.yscale('log')
        plt.xscale('log')
        plt.show()
        
        
    

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
    plt.xlim(0,1e-13)
    plt.legend(fontsize='small')
    plt.show()

    return


def compare_profiles(datalist, proflat=16, proflon=32):
    
    """ Args for proflon:
        0: Antistellar point
        16: Western terminator
        32: Substellar point
        48: Eastern terminator
        63: Antistellar point again """
    
    return


def mmr_map(data, level=2, cpower=7):
    
    coeff = 10**cpower
    
    levels = data['lev']
    lats = data['lat']
    lons = data['lon']
    surfpress = np.mean(data['ps'], axis=0)
    
    mixing_ratio = np.mean(data['mmr'],0)
    
    plt.contourf(lons, lats, mixing_ratio[level,:,:]*coeff,
#                 np.arange(0,1.05,0.1),
                 cmap='plasma')
    plt.title('Mass mixing ratio at %s mbar' % 
              (np.round(levels[level]*surfpress[16,32],0)))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    cbar = plt.colorbar()
    cbar.ax.set_title('$10^{-7}$ kg/kg', size=10)
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
    else:
        u = data['ua'][time_slice,:,:,:]
        v = data['va'][time_slice,:,:,:]

    nlon = len(data['lon'])
    nlat = len(data['lat'])
    levels = data['lev']
    surfpress = np.mean(data['ps'], axis=0)
    heights = levels*surfpress[16,32]
    titleterm = 'Horizonal wind'
    titletime = 'long-term mean'
    
    X, Y = np.meshgrid(np.arange(0, nlon), np.arange(0, nlat))
    
    fig, ax = plt.subplots(figsize=(8.5, 5))
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
    

def plot_lonlat(data, cube, unit, colors='plasma'):
    """ Make a lon-lat contourfill plot of the input data cube"""
    plt.contourf(data.lon, data.lat, cube, cmap=colors)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    cbar = plt.colorbar()
    cbar.set_label(unit, fontsize=10)
    plt.show()
    
def haze_column(data, cubename, month=-1, meaning=True):
    """ Calculate total integrated vertical haze column and plot """
    if meaning==True:           
        pressure = np.transpose(data.levp * \
                   np.mean(data.ps, axis=0)[...,None], (2, 0, 1))
        pressure_thickness = np.diff(pressure, axis=0)
        column = np.sum(np.mean(cubename[:,1:,:,:], axis=0) * \
                 pressure_thickness[1:,:,:], axis=0)
        column /= data.gravity
    else:
        pressure = data.levp[month,:,:,:] * \
                   data.ps[month,:,:,:]
        pressure_thickness = np.diff(pressure, axis=1)
        column = np.sum(cubename[month,1:,:,:] * \
                 pressure_thickness[1:,:,:], axis=0)
        column /= data.gravity
    
    plot_lonlat(data=data, cube=column, unit='kg/kg')
    
    
    
    
    


    
    
    
    