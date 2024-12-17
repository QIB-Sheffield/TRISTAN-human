import os
import numpy as np
import dbdicom as db
import pandas as pd
from utilities import helper
import nibabel as nib

def arrays_from_dicom(info):
    # Extract the subject, visit and scan numbers
    subject, visit, scan = info['subject'], info['visit'], info['scan']
    scan_path = os.path.join(info['output_path'], 'arrays',f'Subject_{subject}', f'Visit_{visit}', f'Scan_{scan}')
    
    helper.check_dirs_exist(scan_path)

    database = db.database(info['data_path'])

    series = database.studies()[0].series()

    for instance in series:
                
        instance.SeriesDescription = format_series_description(str(instance.SeriesDescription))
        sorted_np_export(instance, scan_path)
        
        if 'T1Map_LL' in instance.SeriesDescription:
            export_inversion_times(instance, scan_path)
    
    combine_dynamic(scan_path)

    return

def export_inversion_times(instance, path):
    '''
    Export inversion times as a Numpy file.

    Parameters:
        instance (dbdicom series): The DICOM series to be exported
        path (str): The path to the directory where the Numpy file will be saved

    '''
    try:
        inversion_times = instance.InversionTime
        spacing = np.append(instance.PixelSpacing, instance.SliceThickness)
        if 'POST' in instance.label():
            np.save(f'{path}\\inversion_times_postc.npz', inversion_times = inversion_times, spacing = spacing)
        else:
            np.save(f'{path}\\inversion_times_prec.npz', inversion_times = inversion_times, spacing = spacing)
    except Exception:
        pass

    return

def nifti_from_arrays(path_to_arrays, path_to_nifti, file_name):
    '''
    Save the data as a NIfTI file.

    Parameters:
        array (numpy.ndarray): The data to be saved
        save_path (str): The path to the directory where the file will be saved
        file_name (str): The name of the file to be saved
    '''
    
    array = np.load(path_to_arrays)['data']
    
    # Create an affine matrix
    affine = np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])
    
    # Create an empty header
    empty_header = nib.Nifti1Header()
    empty_header.get_data_shape()

    # Create a NIfTI image
    nib.aff2axcodes(affine)
    new_image = nib.Nifti1Image(array, affine, header=empty_header)

    # Save the NIfTI image
    nib.save(new_image, os.path.join(path_to_nifti, file_name))

    return

def format_series_description(description):
    '''
    Format the series description to remove special characters.

    Parameters:
        series_description (str): The series description to be formatted

    Returns:
        series_description (str): The formatted series description
    '''
    if '*' in description:
        format_description = description.replace('*', 'star')
    elif ':' in description:
        format_description = description.replace(':', '')
    elif '-' in description:
        format_description = description.replace('-', '_')
    else:
        return description

    return format_description

def sorted_np_export(instance, path):
    '''
    Export DICOM pixel values as a Numpy file based on the manufacturer DICOM 
    tag usage.

    Parameters:
        instance (dbdicom series): The DICOM series to be exported
        path (str): The path to the directory where the Numpy file will be saved

    '''
    if instance.Manufacturer == 'GE MEDICAL SYSTEMS':
        sorted_np_export_ge(instance, path)
    elif instance.Manufacturer == 'SIEMENS' or instance.Manufacturer == 'Philips':
        sorted_np_export_siemens_philips(instance, path)
    
    return

def sorted_np_export_ge(instance, path):
    '''
    Export GE type DICOM pixel values as a Numpy file.

    Parameters:
        instance (dbdicom series): The DICOM series to be exported
        path (str): The path to the directory where the Numpy file will be saved

    '''
    try:
        if 'DISCO' in instance.label():
            print(f'starting timepoints for DISCO {instance.SeriesNumber}')
            timepoints = instance.TriggerTime + (instance.AcquisitionTime*1000)
            np.savez_compressed('{}\\dyn_timepoints_{}.npz'.format(path, instance.SeriesNumber), timepoints=timepoints)
            np.savez_compressed('{}\\dyn_spacing_{}.npz'.format(path, instance.SeriesNumber), spacing = np.append(instance.PixelSpacing, instance.SliceThickness))
    except Exception:
        pass

    try:
        instance.export_as_npy(path, ['SliceLocation','TriggerTime'])
    except Exception:
        try:
            instance.export_as_npy(path, ['SliceLocation','InstanceNumber'])
        except Exception:
            try:
                instance.export_as_npy(path, ['InstanceNumber', 'SliceLocation'])
            except Exception:
                instance.export_as_npy(path)

    return

def sorted_np_export_siemens_philips(instance, path):
    '''
    Export siemens type DICOM pixel values as a Numpy file.

    Parameters:
        instance (dbdicom series): The DICOM series to be exported
        path (str): The path to the directory where the Numpy file will be saved

    '''

    try:
        if len(instance.AcquisitionTime) > 100:
            if not instance.SeriesDescription:
                instance.SeriesDescription = 'dynamic'
                
            np.save('{}\\dyn_timepoints_{}.npy'.format(path, instance.SeriesNumber), instance.AcquisitionTime)
            np.save('{}\\dyn_spacing.npy_{}'.format(path, instance.SeriesNumber), np.append(instance.PixelSpacing, instance.SliceThickness))
    except Exception:
        pass

    try:
        instance.export_as_npy(path, ['SliceLocation','AcquisitionTime'])
    except Exception:
        try:
            instance.export_as_npy(path, ['SliceLocation','InstanceNumber'])
        except Exception:
            try:
                instance.export_as_npy(path, ['InstanceNumber', 'SliceLocation'])
            except Exception:
                instance.export_as_npy(path)
   
    return

def combine_dynamic(scan_path):
    
    def stitch_timeseries(file_paths):
        timeseries = []
        for file_path in file_paths:
            data = np.load(file_path)
            timeseries.append(data)
        return np.concatenate(timeseries, axis=-1)

    def stitch_timepoints(file_paths):
        timepoints = []
        for file_path in file_paths:
            data = np.load(file_path)['timepoints']
            timepoints.append(data)
        return np.concatenate(timepoints, axis=-1)
 
    def list_files(directory):
        return [os.path.join(directory, f) for f in os.listdir(directory)]

    file_paths = list_files(scan_path)

    dyn_time_paths = []
    dynamic_files = []

    for file_path in file_paths:
        if 'DISCO' in file_path:
            dynamic_files.append(file_path)

    for file_path in file_paths:
        if 'timepoints' in file_path and '.npz' in file_path:
            dyn_time_paths.append(file_path)

    long_timepoints = stitch_timepoints(dyn_time_paths)
    long_timeseries = stitch_timeseries(dynamic_files)
    spacing = np.load(f'{scan_path}\\dyn_spacing_15.npz')['arr_0']
    
    print('Now saving combined dynamic data')
    np.savez_compressed(f'{scan_path}\\combined_dynamic.npz', data=long_timeseries, timepoints=long_timepoints, spacing=spacing)
    print('Combined dynamic data saved')

    return