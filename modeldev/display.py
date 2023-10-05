import os
import plot
from matplotlib.backends.backend_pdf import PdfPages

filepath = os.path.abspath("")
resultspath = os.path.join(filepath, 'results_liver')
output_file = os.path.join(resultspath, 'parameters.csv')
der_file = os.path.join(resultspath, 'pars_der.csv')
ylim = {
    'Kbh': [0,5],
    'Kbh_i': [0,5],
    'Kbh_f': [0,5],
    'Khe': [0,150],
    'k_he': [0,30],
    'k_he_i': [0,30],
    'k_he_m': [0,30],
    'k_he_f': [0,30],
    've': [0,100],
    'k_bh': [0,4], 
}

plot.pivot_table(output_file)
plot.derive_pars(output_file, der_file)

report_file = os.path.join(resultspath, '_tristan_exp_med_2023_01_10.pdf')

pdf = PdfPages(report_file)
plot.report_title(pdf)
plot.drug_effect_function(output_file, pdf)
plot.calc_effect_size(output_file, pdf)
plot.line_plot_effect(output_file, pdf)
plot.diurnal_k(output_file, pdf)
plot.report_heading(pdf, 'Maximum effect size')
plot.max_effect_size(output_file, pdf, ref='max')
plot.line_plot_max_effect(output_file, pdf)
#plot.max_effect_size(output_file, pdf, ref='min')
plot.report_heading(pdf, 'First scan analysis')

resultspath = os.path.join(filepath, 'results_liver_1scan')
output_file = os.path.join(resultspath, 'parameters.csv')
der_file = os.path.join(resultspath, 'pars_der.csv')

plot.drug_effect_function(output_file, pdf)
plot.line_plot_effect(output_file, pdf)

pdf.close()


# Secondary parameters
#plot.line_plot_extracellular(output_file)

# 
# plot.create_bar_chart(output_file, ylim=ylim)
# plot.create_box_plot(output_file, ylim=ylim)
# plot.drug_effect(output_file)
# 
