import os
import traceback

import numpy as np
import nibabel as nib

import dbdicom as db
from . import helper


def read_database(datapath):
    database = db.database(datapath)
    database.save()
    return database


def arrays_from_dicom(info):
    # Extract the subject, visit and scan numbers
    scan_path = info['scan_path']
    database = db.database(info['data_path'])
    all_series_in_study = database.studies()[0].series()
    
    for series in all_series_in_study:
        # odd bug where sometimes series description is a list for localiser scans
        # more so if new dbdicom is used
        # also some series descriptions have special characters that need to be removed
        # see format_series_description function below
        #
        # if isinstance(desc, list):
        #     desc = desc[0]
        # if '*' or ':' or '-' or '/' or "'" in desc:
        #     series.SeriesDescription = format_series_description(str(desc))
        
        sorted_np_export(series, scan_path)
        
        if 'T1Map_LL' in series.SeriesDescription:
            export_inversion_times(series, scan_path)
    
    combine_dynamic(scan_path)


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
            np.savez(os.path.join(path, 'inversion_times_postc.npz') , inversion_times = inversion_times, spacing = spacing)
        else:
            np.savez(os.path.join(path, 'inversion_times_prec.npz') , inversion_times = inversion_times, spacing = spacing)
    except Exception:
        pass


def export_affine(instance, path):

    try:
        affine = instance.affine()
        affine_path = os.path.join(path, 'affine')
        helper.check_dirs_exist(affine_path)
        affine_file = os.path.join(affine_path, f'{instance.label()}.npz')
        np.savez_compressed(affine_file, affine=affine)
        return
    except Exception:
        pass

    try:
        affine = instance.unique_affines()
        affine_path = os.path.join(path, 'affine')
        helper.check_dirs_exist(affine_path)
        affine_file = os.path.join(affine_path, f'{instance.label()}_unique.npz')
        np.savez_compressed(affine_file, affine=affine)
        return
    except Exception:
        pass


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
    elif '/' in description:
        format_description = description.replace('/', '')
    elif "'" in description:
        format_description = description.replace("'", '')
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
            np.savez_compressed(os.path.join(path, 'dyn_timepoints_{}.npz'.format(instance.SeriesNumber)) , timepoints = timepoints)
            np.savez_compressed(os.path.join(path, 'dyn_spacing_{}.npz'.format(instance.SeriesNumber)) , spacing = np.append(instance.PixelSpacing, instance.SliceThickness))
            np.savez_compressed(os.path.join(path, 'dyn_header_{}.npz'.format(instance.SeriesNumber)) , fa = instance.FlipAngle, Tr=instance.RepetitionTime, weight=instance.PatientWeight, study_date=instance.StudyDate)
    except Exception:
        pass

    try:
        instance.export_as_npy(path, ['SliceLocation','TriggerTime'])
    except Exception:
        error_message = traceback.format_exc()
        print(error_message)
        try:
            instance.export_as_npy(path, ['SliceLocation','InstanceNumber'])
        except Exception:
            error_message = traceback.format_exc()
            print(error_message)
            try:
                instance.export_as_npy(path, ['InstanceNumber', 'SliceLocation'])
            except Exception:
                error_message = traceback.format_exc()
                print(error_message)
                instance.export_as_npy(path)

    try:
        export_affine(instance, path)
    except Exception:
        pass



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
            
            np.savez_compressed(os.path.join(path, 'dyn_timepoints_{}.npz'.format(instance.SeriesNumber)) , timepoints = instance.AcquisitionTime)
            np.savez_compressed(os.path.join(path, 'dyn_spacing_{}.npz'.format(instance.SeriesNumber)) , spacing = np.append(instance.PixelSpacing, instance.SliceThickness))
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
    spacing = np.load(os.path.join(scan_path,'dyn_spacing_15.npz'))['spacing']
    header_info = np.load(os.path.join(scan_path,'dyn_header_15.npz'))
    fa = header_info['fa'] # degrees
    Tr = header_info['Tr'] # ms
    
    print('Now saving combined dynamic data')
    np.savez_compressed(os.path.join(scan_path,'combined_dynamic.npz'), data=long_timeseries, timepoints=long_timepoints, spacing=spacing, fa=fa, Tr=Tr)
    print('Combined dynamic data saved')