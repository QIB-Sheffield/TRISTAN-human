import mdreg
import os
from utilities import helper
import numpy as np

def map_molli(info, checkpoint):
    
    array_path = os.path.join(info['output_path'], 'arrays')
    subject_path = info['subject_path']
    output_path = info['output_path']
    fig_path = os.path.join(output_path, 'figures')
    
    scan_path = os.path.join(array_path, subject_path)

    file_list = helper.list_files(scan_path)

    for filename in file_list:
        if 'LL' in filename and checkpoint in filename:
            molli_array = np.load(filename)
        
        if 'LL' in filename and 'inversion_times' in filename:
            inv_times = np.load(filename)['inversion_times']
            spacing = np.load(filename)['spacing']
    
    molli = {
    # The function to fit the data
    'func': mdreg.abs_exp_recovery_2p,
    # The keyword arguments required by the function
    'TI': inv_times/1000,
    }
    
    coreg_params = {
        'package': 'elastix',
        'FinalGridSpacingInPhysicalUnits': 50.0,
        'spacing': spacing
    }

    plot_settings = {
        'path' : fig_path,
        'vmin' : np.percentile(molli_array, 1),
        'vmax' : np.percentile(molli_array, 99),
        'show' : False,
    }

    coreg, defo, fit, pars = mdreg.fit(molli_array,
                                    fit_image=molli, 
                                    fit_coreg=coreg_params,
                                    plot_params=plot_settings,
                                    maxit=5, 
                                    verbose=2)

    np.savez_compressed(f'{output_path}\\coreg\\{checkpoint}_molli.npz', coreg=coreg, pars=pars)
    mdreg.plot_series(molli_array, fit, coreg, defo, plot_settings)

    return