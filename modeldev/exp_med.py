import os

import fit_aorta_1scan
import fit_liver_1scan
import fit_aorta_2scan
import fit_liver_2scan
#import fit_liver_1scan_vart

import plot
import calc
import exp_med_report

root = os.path.abspath(os.sep)
datapath = os.path.join(root, 'Users', 'steve', 'Dropbox')
sourcepath = os.path.join(datapath, 'Data', 'tristan_exp_med')
resultspath = os.path.join(datapath, 'Results', 'tristan_exp_med')

# Calculate
# fit_aorta_1scan.main(sourcepath, resultspath)
fit_liver_1scan.main(sourcepath, resultspath)
# fit_liver_1scan_vart.main()
# fit_aorta_2scan.main(sourcepath, resultspath)
# fit_liver_2scan.main(sourcepath, resultspath)

# Aorta 2-scan results.
src = os.path.join(resultspath, 'aorta_2scan')
calc.derive_aorta_pars(src)
calc.ttest(src, 'parameters_ext.pkl')

# Liver 2-scan results
src = os.path.join(resultspath, 'liver_2scan')
calc.derive_liver_pars(src)
calc.report_pars(src)
calc.desc_stats(src)
plot.effect_plot(src, ylim=[50,5])
plot.diurnal_k(src, ylim=[50,5])
calc.ttest(src, 'parameters_rep.pkl')

# Aorta 1-scan results
src = os.path.join(resultspath, 'aorta_1scan')
calc.derive_aorta_pars(src)
calc.ttest(src, 'parameters_ext.pkl')

# Liver 1-scan results
src = os.path.join(resultspath, 'liver_1scan')
calc.derive_liver_pars(src)
calc.desc_stats(src)
plot.effect_plot(src, ylim=[50,5])
calc.ttest(src, 'parameters_ext.pkl')

# # # results on varying the acquisition time
# # # plot.tmax_effect(os.path.join(filepath, 'results', 'liver_1scan_vart', 'parameters.csv'), pdf)

exp_med_report.generate('report_exp_med', resultspath)