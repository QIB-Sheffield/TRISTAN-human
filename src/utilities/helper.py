import os
import pandas as pd

def details_dict(subject_details, study_name, logger=None):
    rel_path = os.path.dirname(__file__)
    data_finder_path = os.path.join(rel_path, '..', '..', 'data', 'data_finder.csv')
    df = pd.read_csv(data_finder_path)

    subject, visit, scan = subject_details
    
    # Find the row in the dataframe corresponding to the volunteer number and visit number
    row = df.loc[(df['Subject No'] == subject) & (df['Visit'] == visit)]

    # Get the directory name corresponding to volunteer number and visit 
    dirname = row['Directory Name'].values[0]
    comp_name = row['Drug'].values[0]
    subject_id = row['Subject ID'].values[0]
    
    main_data_path_file = os.path.join(rel_path, '..', '..', 'data', 'data_paths.txt')
    with open(main_data_path_file, 'r') as file:
        main_data_path = file.readline().strip()
        main_output_path = file.readline().strip()
   
    data_path = os.path.join(main_data_path, subject_id, dirname)
    output_path = os.path.join(main_output_path, study_name, comp_name, 'outputs')
    results_path = os.path.join(main_output_path, study_name, comp_name, 'results')
    subject_path = os.path.join(f'Subject_{subject}', f'Visit_{visit}', f'Scan_{scan}')
    
    if not os.path.exists(data_path):
        logger.error(f"Data path {data_path} does not exist")

    if not os.path.exists(data_finder_path):
        logger.error(f"Data finder path {data_finder_path} does not exist")

    check_dirs_exist(output_path, results_path)

    info = {
        'subject': subject_details[0],
        'visit': subject_details[1],
        'scan': subject_details[2],
        'subject_id': subject_id,
        'study_name': study_name,
        'comp_name': comp_name,
        'data_finder_path': data_finder_path,
        'data_path': data_path,
        'output_path': output_path,
        'results_path': results_path,
        'subject_path': subject_path,
        'logger':logger
    }

    return info

def check_dirs_exist(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
    

def list_files(directory):
        return [os.path.join(directory, f) for f in os.listdir(directory)]