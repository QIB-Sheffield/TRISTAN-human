import nibabel as nib
import numpy as np
from utils import save_as_nifti
import os

subjects = [1, 2, 3]
visits = [1, 2]
scans = [1, 2]

rel_path = os.path.dirname(__file__)
seg_path = os.path.join(rel_path, '..\\outputs\\segmentation\\liver')
coreg_path = os.path.join(rel_path, '..\\outputs\\coreg\\AUC')

os.makedirs(coreg_path, exist_ok=True)


for subject in subjects:
    for visit in visits:
        for scan in scans:
            # Read in the NIfTI file
            nifti_file = f"{coreg_path}/Subject{subject}_Visit{visit}_Scan{scan}_coreg.nii.gz"
            nifti_img = nib.load(nifti_file)
            data = nifti_img.get_fdata()

            # Calculate the area under the curve
            auc = np.trapz(data, axis=-1
                           )

            # Save the AUC image to a new NIfTI file
            save_as_nifti(auc, coreg_path, f'Subject{subject}_Visit{visit}_Scan{scan}_coreg_AUC.nii.gz')

            print(f"Saved AUC image for Subject {subject}, Visit {visit}, Scan {scan}")
