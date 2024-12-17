import numpy as np
import mdreg
import os
import time

import os
import numpy as np
import itk
import mdreg
import dcmri
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nibabel as nib
from utils import get_subject_visits, check_dirs_exist, save_as_nifti, orient_for_imshow
import pickle
import traceback

def save_coreg_output(coreg_output, subject_details, main_path, series_no, acq_type, show_coreg_output):
    '''
    Save coregistration output to folder and display output
    
    Args:
        coreg_output (tuple): Coregistered output consisting of:
            coreg (np.array): Coregistered DCE data
            def_max (np.array): Maximum deformation
            model_fit (np.array): Model fit
        subject_details (tuple): Subject, visit, scan
        main path (str): Path to folder to save coregistration output
        series_no (str): Series number
        acq_type (str): Acquisition type
        show_coreg_output (bool): Display coregistration output

    Returns:
        None
    '''
    
    subject, visit, scan = subject_details
    coreg, def_max, model_fit = coreg_output
    
    coreg_path = os.path.join(main_path, '..\\outputs\\coreg\\')
    check_dirs_exist(coreg_path)
    
    file_coreg = os.path.join(coreg_path, f'Subject{subject}_Visit{visit}_Scan{scan}_coreg.npz')
    np.savez(file_coreg, coreg=coreg, def_max=def_max, model_fit=model_fit)
    
    save_as_nifti(coreg, coreg_path, f'Subject{subject}_Visit{visit}_Scan{scan}_coreg.nii.gz')
    save_as_nifti(def_max, coreg_path, f'Subject{subject}_Visit{visit}_Scan{scan}_defmax.nii.gz')
    save_as_nifti(model_fit, coreg_path, f'Subject{subject}_Visit{visit}_Scan{scan}_modelfit.nii.gz')

    if show_coreg_output:
        show_coreg_output(coreg, subject_details, series_no, acq_type, main_path)
    
    return

def show_coreg_output(coreg, subject_details, series_no, acq_type, main_path):
    '''
    Display coregistration output

    Args:
        coreg (np.array): Coregistered DCE data
        subject_details (tuple): Subject, visit, scan
        series_no (str): Series number
        acq_type (str): Acquisition type
        main path (str): Path to folder containing dynamic data

    Returns:
        None
    
    '''

    subject, visit, scan = subject_details
    data = np.load(f"{main_path}..\data\\np_arrays\\Subject_{subject}\\Visit_{visit}\\Scan_{scan}\\{series_no}{acq_type}.npy",allow_pickle=True)
    
    # Orient the data for imshow
    data_disp = orient_for_imshow(data)
    coreg_disp = orient_for_imshow(coreg)

    # Identify central slices for display
    slice_no = data_disp.shape[2]//2

    # Plot the original and coregistered slices
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # Adjust as needed for the number of slices

    # Original data
    imgs_orig = [axs[0, i-16].imshow(data_disp[:,:,i,0], cmap='gray', vmax=np.percentile(data_disp, 99)) for i in range(slice_no-2, slice_no+1)]
    for i, ax in enumerate(axs[0, :]):
        ax.set_title(f'Original slice {i+1}')

    # Coregistered data
    imgs_coreg = [axs[1, i-16].imshow(coreg_disp[:,:,i,0], cmap='gray', vmax=np.percentile(coreg_disp, 99) ) for i in range(slice_no-2, slice_no+1)]
    for i, ax in enumerate(axs[1, :]):
        ax.set_title(f'Coregistered slice {i+1}')

    # Create the slider
    time_slider_ax  = plt.axes([0.1, 0.05, 0.8, 0.02])
    time_slider = Slider(time_slider_ax, 'Time', 0, len(coreg_disp[0,0,0,:])-1, valinit=0, valstep=1)

    def update_image(val):
        time_index = int(time_slider.val)
        for i in range(slice_no-2,slice_no+1):  # Adjust as needed for the number of slices
            imgs_orig[i-16].set_data(data_disp[:,:,i,time_index])
            imgs_coreg[i-16].set_data(coreg_disp[:,:,i,time_index])
        fig.canvas.draw_idle()

    time_slider.on_changed(update_image)

    plt.show()

    return


def main(run_tag, start_point, end_point, subject_details):
    '''Main function for DCE model-driven registration
        
    Returns:
        None
    
    '''
    
    rel_path = os.path.dirname(__file__)
    output_path = os.path.join(f"..\\outputs\\{run_tag}\\coreg\\")
    full_path = os.path.join(rel_path, output_path)
    check_dirs_exist(full_path)

    subject, visit, scan = subject_details

    if end_point is None:
        print("End point not specified, fitting full dynamic data")

    print(f"Running coregistration for Subject {subject}, Visit {visit}, Scan {scan}")
    print(f"Starting aif and dynamic data loading")
    
    aif = np.load(f"{rel_path}\\..\\outputs\\aif\\ciclosporin\\s{subject}_v{visit}_s{scan}_aif.npz",allow_pickle=True)['aif'] 
    
    print(f"Loaded aif for Subject {subject}, Visit {visit}, Scan {scan}")
    
    data = np.load(f"{rel_path}\\..\\outputs\\np_tristan_twocomp\\Subject_{subject}\\Visit_{visit}\\Scan_{scan}\\combined_dynamic.npz",allow_pickle=True)
    
    print(f"Loaded dynamic data for Subject {subject}, Visit {visit}, Scan {scan}")
    
    spacing = np.load(f"{rel_path}\\..\\outputs\\np_tristan_twocomp\\Subject_{subject}\\Visit_{visit}\\Scan_{scan}\\dyn_spacing_15.npz",allow_pickle=True)['arr_0']
    
    print(f"Loaded spacing for Subject {subject}, Visit {visit}, Scan {scan}")
    
    array = data['data'][:,:,:,start_point:end_point]
    aif = aif[start_point:end_point]
    time_array = data['timepoints'][start_point:end_point]
    time_array = (time_array - time_array[0])/1000

    print("Adjusted time array")
    baseline = int(np.floor(60/time_array[1]))
    # if scan == 1:
    #     baseline = int(np.floor(60/time_array[1]))
    # if scan == 2:
    #     baseline = int(np.floor(300/time_array[1]))
    
    print(f"Calculated baseline: timepoint {baseline}")

    tristan_fit = {
    'func': mdreg.array_2cfm_lin,  # The function to fit the data
    'aif': aif,  # The arterial input function
    'time': time_array,  # The time points
    'baseline': baseline,  # The header information
    'Hct': 0.45
}

    coreg_params = {
        'package': 'elastix',
        'FinalGridSpacingInPhysicalUnits': 50.0,
        'spacing': spacing
    }

    plot_settings = {
        'path' : full_path,
        'vmin' : np.percentile(array, 1),
        'vmax' : np.percentile(array, 99),
        'show' : False,
    }

    start_time = time.time()

    coreg, defo, fit, pars = mdreg.fit(array,
                                    fit_image=tristan_fit, 
                                    fit_coreg=coreg_params,
                                    plot_params=plot_settings,
                                    maxit=5, 
                                    verbose=2)
    
    total_time = time.time() - start_time

    np.savez_compressed('{}\\Sub{}_V{}_S{}_coreg.npz'.format(full_path, subject, visit, scan), coreg=coreg, defo=defo, fit=fit, pars=pars, total_time=total_time)
        
    plot_settings = {
        'path' : full_path,
        'filename': f'Sub{subject}_V{visit}_S{scan}',
        'vmin' : np.percentile(array, 1),
        'vmax' : np.percentile(array, 99),
        'show' : False,
    }

    mdreg.plot_series(array, fit, coreg, **plot_settings)

    return  

if __name__ == '__main__':

    run_name = 'ciclosporin'

    try:
        subject_details = (2, 1, 1)
        start_point = 0
        end_point = 75
        main(run_name, start_point, end_point, subject_details)
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        pass

    try:
        subject_details = (2, 1, 2)
        start_point = 100
        end_point = 175
        main(run_name, start_point, end_point, subject_details)
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        pass




