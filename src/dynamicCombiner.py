import os
import pandas as pd
import numpy as np
from utils import get_subject_visits, check_dirs_exist

def combine_dynamic(dirpath, subject, visit, scan, identifier):
    '''
    Combine the dynamic scans for each subject and visit.

    Parameters:
        dirpath (str): Path to the directory containing the dynamic scans
        outputpath (str): Path to the output directory
        subjects (list): List of integers representing the volunteers
        visits (list): List of integers representing the visits
        scans (list): List of integers representing the scans
    '''

    # Construct the path for the dynamic scans
    dynamic_path_1 = '{}\\Subject_{}\\Visit_{}\\Scan_1'.format(dirpath, subject, visit)
    dynamic_path_2 = '{}\\Subject_{}\\Visit_{}\\Scan_2'.format(dirpath, subject, visit)

    # Scan the dynamic path for files with 'DISCO' in their name
    dynamic_files_1 = [f for f in os.listdir(dynamic_path_1) if 'DISCO' in f]
    dynamic_files_2 = [f for f in os.listdir(dynamic_path_2) if 'DISCO' in f]

    # Check if there are any DISCO files
    if not dynamic_files_1 and not dynamic_files_2:
        print(f"No dynamic files found in scan session")
        return
    elif len(dynamic_files_1) == len(dynamic_files_2) == 1:
        print(f"Only one dynamic file per scan session found")
        return
    else:

        scan1_dynamics = []
        scan2_dynamics = []

        for f in dynamic_files_1:
            pixel_data = np.load(os.path.join(dynamic_path_1, f)).shape
            if pixel_data[-1] > 100:
                print(f"Found a dynamic scan with {pixel_data[-1]} frames")
                scan1_dynamics.append(pixel_data)

        for f in dynamic_files_2:
            pixel_data = np.load(os.path.join(dynamic_path_2, f)).shape
            if pixel_data[-1] > 100:
                print(f"Found a dynamic scan with {pixel_data[-1]} frames")
                scan2_dynamics.append(pixel_data)

    # stack np arrays
    dynamic_np_1 = np.concatenate(scan1_dynamics, axis=-1)
    dynamic_np_2 = np.concatenate(scan2_dynamics, axis=-1)

    # Save the combined dynamic scan
    np.save(os.path.join(dynamic_path_1, f'combined_dynamic.npy'), dynamic_np_1)
    np.save(os.path.join(dynamic_path_2, f'combined_dynamic.npy'), dynamic_np_2)

    return

# Set relative path
rel_path = os.path.dirname(__file__)

# Construct all relevant paths
np_path =  os.path.join(rel_path,'..\\outputs\\test_philips')
subject_visits_path = os.path.join(rel_path, '..\\data\\subjects_visits.txt')

subjects, visits, scans = get_subject_visits(subject_visits_path)

# Please update the identifier to match the series description of the dynamic
# scans you want to combine. E.g. 'dynamic' or 'DISCO'
identifier = 'DISCO'

for subject in subjects:
    for visit_number in visits:

        # Construct the path for the output
        path1 = '{}\\Subject_{}\\Visit_{}\\Scan_1'.format(np_path, subject, visit_number)
        path2 = '{}\\Subject_{}\\Visit_{}\\Scan_2'.format(np_path, subject, visit_number)

        combine_dynamic(np_path, np_path, subject, visit_number, 1, identifier)



        
        