import pandas as pd
import argparse
import miepython
import numpy as np


def main(args):

    # For embedded spheres, we need to divide the refractive index by the medium
    # refractive index and multiply the size parameter by the medium refractive index.
    # N2 has a refractive index of 1.0003 or 1.00029 in our wavelength range so we
    # approximate this as 1 and omit in the calculations below.

    radius = 0.05 # in micrometers
    radiusnm = int(radius*1000) # in nanometers

    coeffs = pd.read_csv(args.hazedata[0], encoding='ISO-8859-1', sep=' ')
    spectrum = pd.read_csv(args.stellarspectrum[0], sep=' ')
    
    coeffs = coeffs.drop(columns='index')

    coeffs['wave'] = coeffs['wave'].round(2)
    subset = coeffs[coeffs['wave'].isin(spectrum['Wavelength'])]
    subset = subset.reset_index(drop=True)
    
    spectrum = spectrum[spectrum['Wavelength'].isin(subset['wave'])]
    spectrum = spectrum.reset_index(drop=True)
    
    size_parameter = 2*np.pi*radius/spectrum['Wavelength']
    r_index = subset['n'] - 1.0j*subset['k']

    subset['flux'] = spectrum['Flux']
    subset['r_index'] = r_index
    subset['size_parameter'] = size_parameter
    
    qext, qscat, qback, g = miepython.mie(subset['r_index'], subset['size_parameter'])
    subset['qext'] = qext
    subset['qscat'] = qscat
    subset['qback'] = qback
    subset['g'] = g
    print(subset)
    
    subset.to_csv(str(args.stellarspectrum[0][:-4]) + '_data.csv', index=False)
    
    sw1 = subset.loc[subset['wave'] < 0.75].copy()
    sw2 = subset.loc[subset['wave'] > 0.75].copy()
    sw1_qext = np.average(sw1['qext'], weights=sw1['flux'])
    sw1_qscat = np.average(sw1['qscat'], weights=sw1['flux'])
    sw1_qback = np.average(sw1['qback'], weights=sw1['flux'])
    sw1_g = np.average(sw1['g'], weights=sw1['flux'])
    sw2_qext = np.average(sw2['qext'], weights=sw2['flux'])
    sw2_qscat = np.average(sw2['qscat'], weights=sw2['flux'])
    sw2_qback = np.average(sw2['qback'], weights=sw2['flux'])
    sw2_g = np.average(sw2['g'], weights=sw2['flux']) 
    
    with open(str(args.stellarspectrum[0][:-4]) + f'_constants_{radiusnm}' + '.dat', 'w') as datafile:
        for value in [sw1_qext, sw1_qscat, sw1_qback, sw1_g, sw2_qext, sw2_qscat, sw2_qback, sw2_g]:
            print(value)
            datafile.write(str(value) +'\n')
    




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generates haze optical constants for ExoPlaSim based on He et al. 2023')
        
    parser.add_argument('stellarspectrum',nargs=1,help='Spectrum of star')
    parser.add_argument('hazedata',nargs=1,help='CSV of complex refractive index of haze for wavelengths')
    
    args = parser.parse_args()

    main(args)
