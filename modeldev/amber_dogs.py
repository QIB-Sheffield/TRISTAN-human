import os

import fit_amber_aorta
import fit_amber_kidney

import plot
import calc
import amber_dogs_report

root = os.path.abspath(os.sep)
datapath = os.path.join(root, 'Users', 'steve', 'Dropbox')
datadir = os.path.join(datapath, 'Data', 'amber_dogs')
aorta_results = os.path.join(datapath, 'Results', 'amber_dogs', 'aorta')
kidney_results = os.path.join(datapath, 'Results', 'amber_dogs', 'kidneys')

# Calculate
#fit_amber_aorta.main(datadir, aorta_results)
fit_amber_kidney.main(datadir, aorta_results, kidney_results)

# Aorta results.
calc.derive_aorta_pars(aorta_results)
calc.ttest(aorta_results, 'parameters_ext.pkl')

# Kidney results
calc.ttest(kidney_results, 'parameters.pkl')

amber_dogs_report.generate('amber_dogs_report', os.path.join(datapath, 'Results', 'amber_dogs'))