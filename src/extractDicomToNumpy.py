# Description: This script exports the DICOM files from the database to NPY files for Tristan experimental medicine study

import dbdicom as db
import os
import pandas as pd
import numpy as np
from utils import get_subject_visits, check_dirs_exist, format_series_description

# chose right scanner type to sort via DICOM tags
from utils import sorted_np_export

# Set relative path
rel_path = os.path.dirname(__file__)

# Construct all relevant paths
outputspath =  os.path.join(rel_path,'..\\outputs\\np_tristan_twocomp')
data_finder_path = os.path.join(rel_path,'..\\data\\data_finder.csv')

subject_visits_path = os.path.join(rel_path, '..\\data\\subjects_visits.txt')

subjects, visits, scans = get_subject_visits(subject_visits_path)
print()


# Read the CSV data
df = pd.read_csv(data_finder_path)
# Now df is a DataFrame containing the data from the CSV file

subjects = [2]
visits = [1]
scans = [1, 2]

for subject in subjects:
    for visit_number in visits:

        # Construct the path for the output
        path1 = '{}\\Subject_{}\\Visit_{}\\Scan_1'.format(outputspath, subject, visit_number)
        path2 = '{}\\Subject_{}\\Visit_{}\\Scan_2'.format(outputspath, subject, visit_number)

        check_dirs_exist(path1, path2)

        # Find the row in the dataframe corresponding to the volunteer number and visit number
        row = df.loc[(df['Subject No'] == subject) & (df['Visit'] == visit_number)]

        # Get the directory name corresponding to volunteer number and visit 
        dirname = row['Directory Name'].values[0]
        
        #database_path = os.path.join(rel_path, '..\\data\\{}'.format(dirname))

        database_path = os.path.join(rel_path, '..\\..\\..\\Data\\tristan_two_comp_use\\')
        # Load the database
        database = db.database(database_path)
        database.save()

        # Get the series for each visit
        series_study_1 = database.studies()[0].series()
        series_study_2 = database.studies()[1].series()

        # Export all the series in the visit
        # for instance in series_study_1[16:17]:

        #     instance.SeriesDescription = format_series_description(str(instance.SeriesDescription))
        #     sorted_np_export(instance, path1)
        
        print('starting second scan')
        for instance in series_study_2[15:]:
                
                instance.SeriesDescription = format_series_description(str(instance.SeriesDescription))
                sorted_np_export(instance, path2)

        print('All dicom converted for Volunteer {} Visit {}'.format(subject, visit_number))

