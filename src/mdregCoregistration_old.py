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

def load_aif(subject_details, main_path):
    '''Load AIF (a.u.) and time (s) from folder containing AIF.csv file

    Args:
        folder (str): Path to folder containing AIF.csv
    
    Returns:
        aif (np.array): Arterial input function
        time (np.array): Time points

    '''
    subject, visit, scan = subject_details

    aif_path = os.path.join(main_path,'..\\outputs\\aifs')

    file_aif = os.path.join(aif_path,f"Subject{subject}_Visit{visit}_Scan{scan}_aif.npz")
    average_aif = np.load(file_aif, allow_pickle=True)
    
    aif = average_aif['aif_signal']
    time = average_aif['adj_time']
    
    plot_aif = False
    if plot_aif:
        plt.plot(time, aif)
        plt.show()

    baseline = average_aif['baseline']
    pixel_spacing = average_aif['pixel_spacing']
    slice_thickness = average_aif['slice_thickness'].tolist()

    spacing = pixel_spacing.tolist()
    spacing.append(slice_thickness)

    header_info = {'baseline': int(baseline), 'spacing': spacing}

    return aif, time, header_info

def load_dynamic_data(subject_details, main_path, series_no, acq_type):
    '''
    Load dynamic data from folder containing dynamic data
    
    Args:
        subject_details (tuple): Subject, visit, scan
        main path (str): Path to folder containing dynamic data
        series_no (str): Series number
        acq_type (str): Acquisition type
        
    Returns:
        array (np.array): 4D array of DCE data
    '''
    
    subject, visit, scan = subject_details
    
    data_path = f'\\..\\outputs\\np_arrays\\Subject_{subject}\\Visit_{visit}\\Scan_{scan}\\'

    file_dynamic = os.path.join(main_path+data_path, f'{series_no}{acq_type}.npy')
    
    array = np.load(file_dynamic, allow_pickle=True)

    return array

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

def dce_setup(subject_details, main_path, series_no, acq_type, show_coreg_output):
    '''
    Setup for DCE model-driven registration.
    
    Args:
        subject_details (tuple): Subject, visit, scan
        main path (str): Path to folder containing dynamic data
        series_no (str): Series number
        acq_type (str): Acquisition type
        show_coreg_output (bool): Display coregistration output
        
    Returns:
        _mdr() (function): Model-driven registration function output

    '''
    
    aif, time, header_info = load_aif(subject_details, main_path)
    array = load_dynamic_data(subject_details, main_path, series_no, acq_type)

    print('Setting up MDR..')
    signal_pars = {'aif':aif[:50], 'time':time[:50], 'baseline':header_info['baseline']}
    signal_model = dcmri.pixel_2cfm_linfit

    coreg_output = _mdr(array[:,:,:,:50], header_info, signal_model, signal_pars)

    save_coreg_output(coreg_output, subject_details, main_path, series_no, acq_type, show_coreg_output)

    return

def default_elastix_parameters():
    '''Default elastix parameters for model-driven registration
    
    Returns:
        param_obj (itk.ParameterObject): Elastix parameter object
    '''
    # See here for default bspline settings and explanation of parameters
    # https://github.com/SuperElastix/ElastixModelZoo/tree/master/models%2Fdefault
    param_obj = itk.ParameterObject.New()
    parameter_map_bspline = param_obj.GetDefaultParameterMap('bspline')
    param_obj.AddParameterMap(parameter_map_bspline) 
    param_obj.SetParameter("FixedImagePyramid", "FixedRecursiveImagePyramid") # "FixedSmoothingImagePyramid"
    param_obj.SetParameter("MovingImagePyramid", "MovingRecursiveImagePyramid") # "MovingSmoothingImagePyramid"
    param_obj.SetParameter("Metric", "AdvancedMeanSquares")
    param_obj.SetParameter("FinalGridSpacingInPhysicalUnits", "50.0")
    param_obj.SetParameter("ErodeMask", "false")
    param_obj.SetParameter("ErodeFixedMask", "false")
    #param_obj.SetParameter("PassiveEdgeWidth", "4") #added
    #param_obj.SetParameter("NumberOfResolutions", "4") 
    param_obj.SetParameter("MaximumNumberOfIterations", "500") # down from 500 MaximumNumberOfIterations = 256
    param_obj.SetParameter("MaximumStepLength", "0.1") 
    #param_obj.SetParameter("NumberOfSpatialSamples", "2048")
    param_obj.SetParameter("BSplineInterpolationOrder", "1")
    param_obj.SetParameter("FinalBSplineInterpolationOrder", "3")
    #param_obj.SetParameter("DefaultPixelValue", "0")
    param_obj.SetParameter("WriteResultImage", "false")
    return param_obj


def _mdr(array, header_info, signal_model, signal_pars, downsample=2, elastix_parameters=default_elastix_parameters()):
    '''
    Model-driven registration for DCE data.
    
    Args:
        array (np.array): 4D array of DCE data
        header_info (dict): Header information
        signal_model (function): Signal model
        signal_pars (list): Signal parameters
        downsample (int): Downsample factor
        elastix_parameters (itk.ParameterObject): Elastix parameter object
        
    Returns:
        tuple: Coregistered output consisting of:
            coreg (np.array): Coregistered DCE data
            def_max (np.array): Maximum deformation
            model_fit (np.array): Model fit
    '''
    # Define 3D output arrays
    model_fit = np.zeros(array.shape)
    coreg = np.zeros(array.shape)
    def_max = np.zeros(array.shape[:3])
    
    mdr = mdreg.MDReg()
    mdr.log = False # True for debugging
    mdr.parallel = False # Parallellization is too slow
    mdr.max_iterations = 5
    mdr.downsample = downsample
    mdr.signal_model = signal_model
    mdr.signal_parameters = signal_pars
    mdr.elastix = elastix_parameters
    mdr.set_array(array)
    mdr.pixel_spacing = header_info['spacing']
    #mdr.set_elastix(MaximumNumberOfIterations = 256, PassiveEdgeWidth=1)
    mdr.precision = 1

    mdr.fit() 
    model_fit = mdr.model_fit
    coreg = mdr.coreg
    def_max = mdr.deformation

    return (coreg, def_max, model_fit)

def setTRISTANdynamic(subject_details):
    '''
    Set dynamic type for TRISTAN data as collection was not consistent

    Args:
        subject_details (tuple): Subject, visit, scan

    Returns:
        dynamic_type (list): Dynamic type for each subject
        
    '''
    subject = subject_details[0]

    if subject == 3:
        acq_type = '[SPGR_cor_fb_fa12_dynamic_4.5_os]'
    else:
        acq_type = '[None]'
    
    series_no = [ [['Series 1101 ', 'Series 901 '],['Series 901 ', 'Series 1001 ']] , [['Series 1001 ', 'Series 1101 '],['Series 901 ', 'Series 1001 ']] , [['Series 1001 ','Series 901 '], ['Series 901 ', 'Series 901 ']]]

    return acq_type, series_no

def main():
    '''Main function for DCE model-driven registration
        
    Returns:
        None
    
    '''
    
    rel_path = os.path.dirname(__file__)
    subject_visits_path = os.path.join(rel_path, '..\\data\\subjects_visits.txt')

    subjects, visits, scans = get_subject_visits(subject_visits_path)   

    subjects = [1]
    visits = [1,2]
    scans = [1,2]

    show_coreg_output = False

    for subject in subjects:
        for visit in visits:
            for scan in scans:
                subject_details = (subject, visit, scan)

                acq_type, series_no = setTRISTANdynamic(subject_details)
                
                dce_setup(subject_details, rel_path, series_no[subject-1][visit-1][scan-1], acq_type, show_coreg_output)

                print(f'Coregistration complete for Subject {subject}, Visit {visit}, Scan {scan}')



if __name__ == '__main__':
    main()

