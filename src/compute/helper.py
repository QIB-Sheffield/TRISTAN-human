import os
import pandas as pd

def details_dict(datapath, resultspath, subject, visit, scan, logger):

    # This can simplify - read from directories rather than header.csv
    
    study_name = 'tristan_twocomp'
    data_finder_path = os.path.join(datapath, 'header.csv')
    df = pd.read_csv(data_finder_path)
    
    # Find the row in the dataframe corresponding to the volunteer number and visit number
    row = df.loc[(df['Subject No'] == subject) & (df['Visit'] == visit)]

    # Get the directory name corresponding to volunteer number and visit 
    dirname = row['Directory Name'].values[0]
    comp_name = row['Drug'].values[0]
    subject_id = row['Subject ID'].values[0]
   
    data_path = os.path.join(datapath, subject_id, dirname)
    output_path = os.path.join(resultspath, study_name, comp_name, 'outputs')
    results_path = os.path.join(resultspath, study_name, comp_name, 'results')
    subject_path = os.path.join(f'Subject_{subject}', f'Visit_{visit}', f'Scan_{scan}')
    
    if not os.path.exists(data_path):
        logger.error(f"Data path {data_path} does not exist")

    check_dirs_exist(output_path, results_path)

    scan_path = os.path.join(output_path, 'arrays', f'Subject_{subject}', f'Visit_{visit}', f'Scan_{scan}')
    
    check_dirs_exist(scan_path)

    info = {
        'subject': subject,
        'visit': visit,
        'scan': scan,
        'subject_id': subject_id,
        'study_name': study_name,
        'comp_name': comp_name,
        'data_finder_path': data_finder_path,
        'data_path': data_path,
        'output_path': output_path,
        'results_path': results_path,
        'subject_path': subject_path,
        'scan_path': scan_path,
        'logger':logger
    }

    return info

def check_dirs_exist(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
    

def list_files(directory):
        return [os.path.join(directory, f) for f in os.listdir(directory)]