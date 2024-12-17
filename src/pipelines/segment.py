import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from utilities import helper
import shutil
from tqdm import tqdm
from totalsegmentator.python_api import totalsegmentator

def setup_segmentation(info, seg_type):

    subject, visit, scan = info['subject'], info['visit'], info['scan']

    if seg_type == 'pre_coreg':
        array_path = os.path.join(info['output_path'], 'arrays')
        data_4d =  np.load(os.path.join(array_path, f'Subject_{subject}', f'Visit_{visit}', f'Scan_{scan}', 'combined_dynamic.npz'))['data']
        dims = np.load(os.path.join(array_path, f'Subject_{subject}', f'Visit_{visit}', f'Scan_{scan}', 'combined_dynamic.npz'))['spacing']
    elif seg_type == 'post_coreg':
        array_path = os.path.join(info['output_path'], 'coreg')
        data_4d =  np.load(os.path.join(array_path, f'Sub{subject}_V{visit}_S{scan}_coreg.npz'))['coreg']
        #dims = np.load(os.path.join(array_path, f'Sub{subject}_V{visit}_S{scan}_coreg.npz'))['spacing']
        dims = np.load(os.path.join(info['output_path'], 'arrays', f'Subject_{5}', f'Visit_{visit}', f'Scan_{scan}', 'combined_dynamic.npz'))['spacing']

    aif_path = os.path.join(info['output_path'], f'aif')
    liver_path = os.path.join(info['output_path'], f'liver')
    helper.check_dirs_exist(aif_path, liver_path)

    dims_axial = [dims[1], dims[2], dims[0]]
    data_4d_axial = np.transpose(data_4d, (3, 1, 2, 0))
    
    data_axial_max = np.max(data_4d_axial, axis=0)

    if info['study_name'] == 'tristan_twocomp':
        data_axial_max = np.flip(data_axial_max, axis=1)

    output_folder = os.path.join(array_path, 'masks', f'S{subject}_v{visit}_s{scan}_max')
    aorta_mask, liver_mask = run_totalsegmentator(data_axial_max, dims_axial, output_folder, seg_type, format='3D',
                                    task="total_mr", return_mask=True,
                                    quiet=True, verbose=True)

    masked_data = data_4d_axial * aorta_mask[np.newaxis, ...]
    aif = np.sum(masked_data, axis=(1,2,3))/np.sum(aorta_mask)

    if seg_type == 'pre_coreg':
        np.savez_compressed(os.path.join(aif_path, f'S{subject}_v{visit}_s{scan}_aif_prec.npz'), aif=aif)
    if seg_type == 'post_coreg':
        masked_data_liver = data_4d_axial * liver_mask[np.newaxis, ...]
        liver = np.sum(masked_data_liver, axis=(1,2,3))/np.sum(liver_mask)
        np.savez_compressed(os.path.join(aif_path, f'S{subject}_v{visit}_s{scan}_aif_postc.npz'), aif=aif)
        np.savez_compressed(os.path.join(liver_path, f'S{subject}_v{visit}_s{scan}_liver_postc.npz'), liver=liver)  

    return


def save_volumes_as_nifti(volumes, voxel_dim, output_folder, output_file='volume', flip_axes=[2,1,0], rotate=180, vol_format='3D', vol_3d=0,
                         verbose=True, return_nifti=False): 
    #flip_axes: is used to order the axes (Heparim data's order is 2,1,0)
    #rotate: use to rotate image
    #vol_format: '3D' or '4D', '3D' saves individual nifti files of each volume, '4D' saves one nifit file of all volumes
    #vol_3d = volume to save in vol_format='3D_single'
    helper.check_dirs_exist(output_folder)
    #for time, Z, X, Y in volumes.shape: #(215, 72, 224, 224)
    affine = np.diag([voxel_dim[0], voxel_dim[1], voxel_dim[2], 1])
        # Stack slices into a 3D volume
    
    if vol_format == '3D':
        time, Z, X, Y = volumes.shape
        for i in tqdm(range(time), desc="...Processing Volumes..."):
            volume_data = volumes[i].transpose(flip_axes[0],flip_axes[1],flip_axes[2])
            volume_data = ndi.rotate(volume_data, rotate, axes=(0,1), reshape=False)
        
            nifti_img = nib.Nifti1Image(volume_data, affine)
            numb = int(i)
            output_path = os.path.join(output_folder, f"{output_file}_{numb}.nii.gz")
            nib.save(nifti_img, output_path)
        print(f"Saved {numb+1} NIFTI files to {output_folder}")
    elif vol_format == '3D_single':
        if vol_3d == None:
            #Z, X, Y = volumes.shape
            #print(f'Mask shape: {volumes.shape}')
            volume_data = volumes.transpose(flip_axes[0],flip_axes[1],flip_axes[2])
            volume_data = ndi.rotate(volume_data, rotate, axes=(0,1), reshape=False)
            
            nifti_img = nib.Nifti1Image(volume_data, affine=affine)
            #numb = int(vol_3d)
            output_path = os.path.join(output_folder, f"{output_file}.nii.gz")
            nib.save(nifti_img, output_path)
        else:
            #time, Z, X, Y = volumes.shape
            volume_data = volumes[vol_3d].transpose(flip_axes[0],flip_axes[1],flip_axes[2])
            volume_data = ndi.rotate(volume_data, rotate, axes=(0,1), reshape=False)
            
            nifti_img = nib.Nifti1Image(volume_data, affine=affine)
            numb = int(vol_3d)
            output_path = os.path.join(output_folder, f"{output_file}_slice{numb}.nii.gz")
            nib.save(nifti_img, output_path)
    else:
            print(f"{vol_format} format not supported.")
        
    #if verbose:    
            #print(f"...Saved 3D NIfTI file to {output_file}...")
    

    if return_nifti:
        return output_path

def run_totalsegmentator(dicom_4d_array, dims, output_folder, seg_type, format='4D',
                         ha_vol=0, task="total_mr", return_mask=True,
                         quiet=True, verbose=True):
    #Get temporary nifti files of desired volumes:
    labels = 'labels'
    img = 'img'
    
    labels_folder = os.path.join(output_folder, labels)
    img_folder = os.path.join(output_folder, img)

    helper.check_dirs_exist(labels_folder, img_folder)

    if format=='4D':
        nifti_path = save_volumes_as_nifti(dicom_4d_array, dims, img_folder, vol_format='3D_single', vol_3d=ha_vol,
                                           verbose=False, return_nifti=True)
    elif format=='3D':
        if len(dicom_4d_array.shape)==3:
            x, y, z = dicom_4d_array.shape
            dicom_4d_array = np.reshape(dicom_4d_array, (1, x, y, z ))
        
        nifti_path = save_volumes_as_nifti(dicom_4d_array, dims, img_folder, vol_format='3D_single', vol_3d=0,
                                           verbose=False, return_nifti=True)
    else:
        print('Format not supported!')
        
    if verbose:
        print("\n...<<Running TotalSegmentator>>...")
    
    totalsegmentator(nifti_path, labels_folder, task=task, roi_subset=['liver', 'aorta'], quiet=quiet, device='cpu')
    aorta_data = nib.load(os.path.join(labels_folder, 'aorta.nii.gz'))
    volume_aorta = aorta_data.get_fdata().transpose(1,0,2)
    volume_aorta = ndi.rotate(volume_aorta, 180, axes=(0,1), reshape=False)
    volume_aorta = np.transpose(volume_aorta, (2, 0, 1))

    liver_data = nib.load(os.path.join(labels_folder, 'liver.nii.gz'))
    volume_liver = liver_data.get_fdata().transpose(1,0,2)
    volume_liver = ndi.rotate(volume_liver, 180, axes=(0,1), reshape=False)
    volume_liver = np.transpose(volume_liver, (2, 0, 1))

    shutil.rmtree(labels_folder)
    shutil.rmtree(img_folder)

    if verbose:
        print("\n...<<Masks created>>...")

    volume_aorta[volume_aorta<1]=0
    volume_aorta[volume_aorta>1]=1

    volume_liver[volume_liver<1]=0
    volume_liver[volume_liver>1]=1

    if seg_type == 'pre_coreg':
        np.savez_compressed(os.path.join(output_folder, 'aorta_mask_precoreg.npz'), aorta_mask=volume_aorta)
    elif seg_type == 'post_coreg':
        np.savez_compressed(os.path.join(output_folder, 'masks_postcoreg.npz'), aorta_mask=volume_aorta, liver_mask=volume_liver)
    
    return volume_aorta, volume_liver

