'''
Utility functions for the project

'''

import os
import numpy as np
import nibabel as nib

def get_subject_visits(subject_visits_path):
    '''
    Read the subjects and visits from the corresponding text file.
    
    The text file must have the following format:
        --------------------------------
        Subjects: 1,2,......,N
        Visits: 1,..,N
        Scans: 1,..,N
        --------------------------------

    Parameters:
        subject_visits_path (str): Path to the text file containing the volunteers and visits

    Returns:
        subjects (list): List of integers representing the volunteers
        visits (list): List of integers representing the visits
        scans (list): List of integers representing the scans
    '''
    with open(subject_visits_path, 'r') as file:
            lines = file.readlines()
            subjects = list(map(int, lines[0].strip().split(': ')[1].split(',')))
            visits = list(map(int, lines[1].strip().split(': ')[1].split(',')))
            scans = list(map(int, lines[2].strip().split(': ')[1].split(',')))

    return subjects, visits, scans

def unique_scan_ids(subjects, visits, scans):
     '''
     Generate list of unique subject, visit, scan numbers for each idividual scanning session

        Parameters:
            subjects (list): List of integers representing the volunteers
            visits (list): List of integers representing the visits
            scans (list): List of integers representing the scans

        Returns:
            scan_ids (list): List of unique scan ids
     '''

     scan_ids = []
     
     for subject in subjects:
        for visit in visits:
            for scan in scans:
                scan_ids.append(f"{subject}, {visit}, {scan}")
            
     return scan_ids

def check_dirs_exist(*args):
    '''
    Check if the directories exist and create them if they don't.

    Parameters:
        *args (str): Variable number of strings representing the paths to the directories
    '''
    for arg in args:
        os.makedirs(arg, exist_ok=True)

    return

def save_as_nifti(array, save_path, file_name):
    '''
    Save the data as a NIfTI file.

    Parameters:
        array (numpy.ndarray): The data to be saved
        save_path (str): The path to the directory where the file will be saved
        file_name (str): The name of the file to be saved
    '''
    
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
    nib.save(new_image, os.path.join(save_path, file_name))

    return

def orient_for_imshow(array):
    '''
    Orient the array for display using imshow.

    Parameters:
        array (numpy.ndarray): The data to be oriented

    Returns:
        array (numpy.ndarray): The oriented data
    '''

    array_reoriented = np.rot90(array, axes=(0,1),k=-1)
    array_reoriented = np.fliplr(array_reoriented)

    return array_reoriented

def load_headerinfo(subject_details, main_path):
    '''Load AIF (a.u.) and time (s) from folder containing AIF.csv file

    Args:
        folder (str): Path to folder containing AIF.csv
    
    Returns:
        header_info (np.array): baseline, spacing, slice_thickness

    '''
    subject, visit, scan = subject_details

    path = os.path.join(main_path,'..\\outputs\\aifs')

    file_aif = os.path.join(path,f"Subject{subject}_Visit{visit}_Scan{scan}_aif.npz")
    average_aif = np.load(file_aif, allow_pickle=True)

    baseline = average_aif['baseline']
    pixel_spacing = average_aif['pixel_spacing']
    slice_thickness = average_aif['slice_thickness'].tolist()

    spacing = pixel_spacing.tolist()
    spacing.append(slice_thickness)

    header_info = {'baseline': int(baseline), 'spacing': spacing}

    return header_info

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

def sorted_np_export_ge(instance, path):
    '''
    Export GE type DICOM pixel values as a Numpy file.

    Parameters:
        instance (dbdicom series): The DICOM series to be exported
        path (str): The path to the directory where the Numpy file will be saved

    '''
    # try:
    #     if 'DISCO' in instance.label():
    #         print(f'starting timepoints for DISCO {instance.SeriesNumber}')
    #         timepoints = instance.TriggerTime + (instance.AcquisitionTime*1000)
    #         np.savez_compressed('{}\\dyn_timepoints_{}.npz'.format(path, instance.SeriesNumber), timepoints=timepoints)
    #         np.savez_compressed('{}\\dyn_spacing_{}.npz'.format(path, instance.SeriesNumber), np.append(instance.PixelSpacing, instance.SliceThickness))
    # except Exception:
    #     pass

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

