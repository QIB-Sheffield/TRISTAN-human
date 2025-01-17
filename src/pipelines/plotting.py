import matplotlib.pyplot as plt
import math
import numpy as np
import os
from utilities import helper

def overlay_masks(info, checkpoint):
    output_path = info['output_path']
    subject, visit, scan = info['subject'], info['visit'], info['scan']
    subject_path = os.path.join(f'Subject_{subject}', f'Visit_{visit}', f'Scan_{scan}')
    data_path = os.path.join(output_path, 'arrays', subject_path, 'combined_dynamic.npz')

    if checkpoint == 'pre_coreg':
         scan_type = 'precoreg'
    elif checkpoint == 'post_coreg':
        scan_type = 'postcoreg'

    mask_path = os.path.join(output_path, 'masks', f'S{subject}_v{visit}_s{scan}_max',f'masks_{scan_type}.npz')
    masks = np.load(mask_path)
    aorta = masks['aorta_mask']
    liver = masks['liver_mask']

    aorta_coronal = np.transpose(aorta, (1,0,2))
    liver_coronal = np.transpose(liver, (1,0,2))

    data = np.load(data_path)['data']
    max_data = np.max(data, axis=-1)
    num_slices = data.shape[2]
    grid_size = math.ceil(math.sqrt(num_slices))
    titlesize = 10

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
    fig.subplots_adjust(wspace=0.5, hspace=0.01)
    plt.tight_layout()

    for i in range(grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            if i < num_slices:
                axes[row, col].imshow(max_data[:, :, i].T, cmap='gray', animated=True, vmin=0, vmax=np.percentile(max_data,95))
                axes[row, col].set_title('Slice {}'.format(i+1), fontsize=titlesize)
                axes[row, col].imshow(np.ma.masked_where(aorta_coronal[i] != 1, aorta_coronal[i]), cmap='jet', alpha=0.5)
                axes[row, col].imshow(np.ma.masked_where(liver_coronal[i] != 1, liver_coronal[i]), cmap='spring', alpha=0.5)
   
            else:
                axes[row, col].axis('off')  # Turn off unused subplots
            axes[row, col].set_xticks([])  # Remove x-axis ticks
            axes[row, col].set_yticks([])

    plt.suptitle(f'Subject {subject}, Visit {visit}, Scan {scan}', fontsize=20)
    helper.check_dirs_exist(os.path.join(output_path, 'figures'))
    fig.savefig(os.path.join(output_path, 'figures', f'S{subject}_v{visit}_s{scan}_overlay_{checkpoint}.png'))