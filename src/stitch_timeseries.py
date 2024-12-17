#%%
import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(file_path):
    # Replace this with actual data loading logic
    return np.load(file_path)

def stitch_timeseries(file_paths):
    timeseries = []
    for file_path in file_paths:
        data = load_data(file_path)
        timeseries.append(data)
    return np.concatenate(timeseries, axis=-1)

def stitch_timepoints(file_paths):
    timepoints = []
    for file_path in file_paths:
        data = load_data(file_path)
        data = data['timepoints']
        timepoints.append(data)
    return np.concatenate(timepoints, axis=-1)


# Example usage
rel_path = os.path.dirname(__file__)
main_path = f'{rel_path}\\..\\outputs\\np_tristan_twocomp\\Subject_2\\Visit_1\\Scan_2\\'

def list_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)]

file_paths = list_files(main_path)

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

corrected_time = long_timepoints - long_timepoints[0]
corrected_time = corrected_time / 1000 / 60
#np.savez_compressed(f'{main_path}\\combined_dynamic.npz', data=long_timeseries, timepoints=long_timepoints)

avg_signal = np.mean(long_timeseries[140:150,80:90,25,:], axis=(0,1))
avg_liver = np.mean(long_timeseries[75:85,105:110,25,:], axis=(0,1))
#%%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(np.max(long_timeseries[:,:,25,:],axis=-1).T, cmap='gray')
rect = plt.Rectangle((140, 80), 10, 10, edgecolor='r', facecolor='none')
ax[0].add_patch(rect)
rect2 = plt.Rectangle((75, 105), 10, 10, edgecolor='c', facecolor='none')
ax[0].add_patch(rect2)
ax[1].plot(corrected_time,avg_signal,'r.', markersize=3)
ax[2].plot(corrected_time,avg_liver,'c.', markersize=3)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xlabel('Time (min)')
ax[2].set_xlabel('Time (min)')
ax[1].set_ylabel('Signal intensity (a.u.)')
ax[2].set_ylabel('Signal intensity (a.u.)')
fig.suptitle('Average in region signals')
plt.tight_layout()
plt.show()


# %%
