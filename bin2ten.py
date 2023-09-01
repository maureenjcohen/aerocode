#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:34:58 2023

@author: Mo Cohen
"""
import numpy as np
import pandas as pd

unbinned = pd.read_csv('/home/s1144983/aerosim/corrales/CG22_nk/CG22_CO-0.625_nk.dat', delimiter=' ')

unbinned['wavel(um)'] = unbinned['wavel(um)'].astype(float).apply(lambda x: np.trunc(x*100)/100)
binned_n = unbinned.groupby(['wavel(um)'])['n_opt'].mean()
binned_k = unbinned.groupby(['wavel(um)'])['k_opt'].mean()
binned_w = unbinned['wavel(um)'].unique().tolist()

d = {'wave': pd.Series(binned_w, index=np.arange(0,554)),
        'n': pd.Series(binned_n.values, index=np.arange(0,554)),
        'k': pd.Series(binned_k.values, index=np.arange(0,554))}

binned = pd.DataFrame(data=d, index=np.arange(0,554), columns=['wave','n','k'])

binned.to_csv('/home/s1144983/aerosim/corrales/binned_corrales.txt', index=False, sep=' ')
