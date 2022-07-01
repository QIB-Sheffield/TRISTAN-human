import numpy as np
import wezel
import dbdicom as db
from mdreg import MDReg
from wezel.apps.dicom import Windows

import menu as tristan

wsl = wezel.app(Windows)
wsl.set_menu(tristan.dev)
wsl.show()
exit()

attr = [
    'SliceLocation', 'AcquisitionTime', 
    'FlipAngle', 'EchoTime', 'InversionTime',                           
]

# Get data
path = 'C:\\Users\\steve\\Dropbox\\Data\\TRISTAN_patient_examples_2\\v4_1\\data-rzp2'
folder = db.Folder(path, attributes=attr)
series = folder.series(SeriesDescription='dev')[0]
#array, dataset = series.array(['SliceLocation','AcquisitionTime'], pixels_first=True)
array, dataset = series.array('AcquisitionTime', pixels_first=True)

# Perform mdreg
mdr = MDReg()
mdr.export_path = 'C:\\Users\\steve\\Dropbox\\Data\\TRISTAN_patient_examples_2\\'
mdr.set_array(np.squeeze(array[...,0]))
mdr.fit()
mdr.export()

# Save results as dicom
array[...,0] = mdr.coreg
fit = series.new_cousin(SeriesDescription = series.SeriesDescription + '_coreg')
fit.set_array(array, dataset, pixels_first=True)

# Display results in wezel
wsl = wezel.app(Windows)
wsl.set_menu(tristan.dev)
wsl.set_data(folder)
wsl.show()






