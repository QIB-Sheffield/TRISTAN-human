import mdreg
import os
from utilities import helper
import numpy as np

def map_molli(info, scan_type):

    logger = info['logger']    
    subject, visit, scan = info['subject'], info['visit'], info['scan']

    array_path = os.path.join(info['output_path'], 'arrays')
    subject_path = info['subject_path']
    output_path = info['output_path']
    fig_path = os.path.join(output_path, 'figures')
    
    scan_path = os.path.join(array_path, subject_path)

    file_list = helper.list_files(scan_path)


    if scan_type == 'pre_coreg':
        dicom_name = ''
        scan_tag = 'prec'
    elif scan_type == 'post_coreg':
        dicom_name = 'POST'
        scan_tag = 'postc'

    molli_output_path = os.path.join(output_path, 'molli')
    helper.check_dirs_exist(molli_output_path)

    molli_file = os.path.join(molli_output_path, f'S{subject}_V{visit}_S{scan}_{scan_type}_molli.npz')

    # Load the correct MOLLI array and corresponding inversion times
    for filename in file_list:
        if 'LL' in filename and dicom_name in filename:
            molli_array = np.load(filename)
        
        if 'inversion_times' in filename and scan_tag in filename:
            inv_times = np.load(filename)['inversion_times']
            spacing = np.load(filename)['spacing']
    
    molli = {
    # The function to fit the data
    'func': mdreg.abs_exp_recovery_2p,
    # The keyword arguments required by the function
    'TI': inv_times/1000, # Convert to seconds from ms
    }
    
    # Set the coregistration parameters to use mdreg
    coreg_params = {
        'package': 'elastix',
        'FinalGridSpacingInPhysicalUnits': 50.0,
        'spacing': spacing,
    }

    plot_settings = {
        'path' : fig_path,
        'vmin' : np.percentile(molli_array, 1),
        'vmax' : np.percentile(molli_array, 99),
        'show' : False,
    }

    coreg, defo, fit, pars = mdreg.fit(molli_array[:,:,0,:],
                                    fit_image=molli, 
                                    fit_coreg=coreg_params,
                                    plot_params=plot_settings,
                                    maxit=5, 
                                    verbose=2)

    # Save the coregistration and fit parameters
    # Fit parameters are the T1 and M0 values

    np.savez(molli_file, coreg=coreg, S0=pars[..., 0], T1=pars[..., 1], fit=fit)
    logger.info(f'{scan_type} MOLLI mapping complete')

    logger.info('TO DO: implement export of ROI T1 values to .npz files as needed for XLSX export')
    logger.info('requires affine and vreg/dbdicom to select reiggion from dce mask')

    # TO DO implement export of ROI T1 values to .npz files as needed for XLSX export
    # requires affine and vreg/dbdicom to select reiggion from dce mask
    # Or similiar if changing the export method in csv.py

    # example:
    # some segmentation extraction here
    # np.savez_compressed(os.path.join(output_path, 'molli', f'S{subject}_V{visit}_S{scan}_{scan_type}_molli_ROIvalues.npz'), time=timepoints, liver=liver_value, aorta=aorta_value)

    return