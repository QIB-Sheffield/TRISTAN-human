import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import data



sym = {
    'baseline': 'b-',
    'rifampicin': 'r-',
}
mark = {
    1: 'o',
    2: 'v',
    3: '^',
    4: '<',
    5: '>',
    6: 's',
    7: 'p',
    8: '*',
    9: 'x',
    10: 'd',
}

def report_title(pdf):
    firstPage = plt.figure(figsize=(8.27,11.69))
    firstPage.clf()
    txt = 'TRISTAN experimental medicine study: \n Results on the primary objective \n\n Internal report \n  TRISTAN work package 2 \n 01 oct 2023'
    firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=16, ha="center")
    pdf.savefig()
    plt.close()

def report_heading(pdf, txt):
    firstPage = plt.figure(figsize=(8.27,11.69))
    firstPage.clf()
    firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=16, ha="center")
    pdf.savefig()
    plt.close()

def derive_effect_size(output):
    subjects = output.subject.unique()
    # Create pivot table
    pivot = pd.pivot_table(output, values='value', index='parameter', columns=['subject', 'visit'])
    # Calculate effect size as new column for each subject
    for subj in subjects:
        base = pivot[subj]['baseline'].values
        try:
            rif = pivot[subj]['rifampicin'].values
            effect = 100*np.divide(rif-base, base)
        except:
            effect = np.full(len(base), np.nan)
        pivot[subj, 'effect'] = effect
        pivot = pivot.sort_index(axis=1)
    # Extract only effect size and drop the column label
    effect = pivot.xs('effect', axis=1, level=1, drop_level=False)
    effect.columns = effect.columns.droplevel(1)
    # Parameters in columns
    effect = effect.T
    return effect



def split_visits(output):
    # Create pivot table
    pivot = pd.pivot_table(output, values='value', index='parameter', columns=['subject', 'visit'])
    # Extract only effect size and drop the column label
    baseline = pivot.xs('baseline', axis=1, level=1, drop_level=False)
    baseline.columns = baseline.columns.droplevel(1)
    rifampicin = pivot.xs('rifampicin', axis=1, level=1, drop_level=False)
    rifampicin.columns = rifampicin.columns.droplevel(1)
    # Parameters in columns
    return baseline.T, rifampicin.T

def calc_effect_size(output_file, pdf):
    resultsfolder = os.path.dirname(output_file)
    output = pd.read_csv(output_file)
    effect = derive_effect_size(output)
    baseline, rifampicin = split_visits(output)
    rifampicin.to_csv(os.path.join(resultsfolder, '_table_rifampicin.csv'))
    baseline.to_csv(os.path.join(resultsfolder, '_table_baseline.csv'))
    effect.to_csv(os.path.join(resultsfolder, '_table_effect.csv'))
    # Calculate stats
    bstats = baseline.describe()
    bstats = bstats[['k_he', 'k_bh']].round(2)
    bstats = bstats.rename(columns={"k_he": "k_he baseline (mL/min/100mL)", "k_bh": "k_bh baseline (mL/min/100mL)"})
    rstats = rifampicin.describe()
    rstats = rstats[['k_he', 'k_bh']].round(2)
    rstats = rstats.rename(columns={"k_he": "k_he rifampicin (mL/min/100mL)", "k_bh": "k_bh rifampicin (mL/min/100mL)"})
    estats = effect.describe()
    estats = estats[['k_he', 'k_bh']].round(1)
    estats = estats.rename(columns={"k_he": "k_he effect size (%)", "k_bh": "k_bh effect size (%)"})
    stats = pd.concat([estats.T, bstats.T, rstats.T])
    stats=stats.reindex([
        "k_he effect size (%)",
        "k_he baseline (mL/min/100mL)",
        "k_he rifampicin (mL/min/100mL)",
        "k_bh effect size (%)",
        "k_bh baseline (mL/min/100mL)",
        "k_bh rifampicin (mL/min/100mL)",
        ])
    stats.to_csv(os.path.join(resultsfolder, '_table_k_stats.csv'))
    # Save table in pdf report
    fig, ax = plt.subplots(figsize=(8.27,11.69))
    #fig.subplots_adjust(left=0.0, right=1.0, bottom=0.1, top=0.9)
    ax.axis('tight')
    ax.axis('off')
    ax.set_title("Rifampicin effect size and absolute values \n of hepatocellular uptake (k_he) and biliary excretion (k_bh)\n of Gadoxetate")
    table = ax.table(cellText=stats.values,colLabels=stats.columns,rowLabels=stats.index.values,loc='center', cellLoc='center')
    table.scale(0.5, 1.5)
    pdf.savefig(fig, bbox_inches='tight')


def derive_pars(output_file, results_file=None):
    output = pd.read_csv(output_file)
    pivot = pd.pivot_table(output, values='value', columns='parameter', index=['visit','subject'])
    if not 'k_he_i' in pivot.columns:
        return pivot
    khe = pivot['k_he'].values
    khe_i = pivot['k_he_i'].values
    khe_f = pivot['k_he_f'].values
    i = np.isnan(khe_i)
    khe_i[i] = khe[i]
    f = np.isnan(khe_f)
    khe_f[i] = khe[f]
    khe_2pt = np.stack([khe_i, khe_f])
    khe_max = np.amax(khe_2pt, axis=0)
    pivot['k_he_max'] = khe_max
    khe_min = np.amin(khe_2pt, axis=0)
    pivot['k_he_min'] = khe_min

    kbh = pivot['k_bh'].values
    kbh_i = pivot['Kbh_i'].values * (1-pivot['ve'].values/100)
    kbh_f = pivot['Kbh_f'].values * (1-pivot['ve'].values/100)
    i = np.isnan(kbh_i)
    kbh_i[i] = kbh[i]
    f = np.isnan(kbh_f)
    kbh_f[i] = kbh[f]
    kbh_2pt = np.stack([kbh_i, kbh_f])
    kbh_max = np.amax(kbh_2pt, axis=0)
    pivot['k_bh_max'] = kbh_max
    kbh_min = np.amin(kbh_2pt, axis=0)
    pivot['k_bh_min'] = kbh_min
    pivot['k_bh_i'] = kbh_i
    pivot['k_bh_f'] = kbh_f
    if results_file is not None:
        pivot.to_csv(results_file)
    return pivot


def line_plot_max_effect(output_file, pdf):

    fontsize=12
    titlesize=16
    markersize=6
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8.27,11.69))
    #fig.tight_layout(pad=10.0)
    fig.subplots_adjust(
                    left=0.1,
                    bottom=0.3,
                    right=0.95,
                    top = 0.6, 
                    wspace=0.3,
                    #hspace=1,
                    )
    title = "Maximum effect size. \nIndividual values for hepatocellular uptake (k_he, left) and biliary excretion (k_bh, right) \n of Gadoxetate at baseline (maximum, left of plot) and after administration of rifampicin (minimum, right of plot)." 
    fig.suptitle(title, fontsize=12)
    ax1.set_title('Hepatocellular uptake rate', fontsize=fontsize, pad=10)
    ax1.set_ylabel('k_he (mL/min/100mL)', fontsize=fontsize)
    ax1.set_ylim(0, 40)
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax2.set_title('Biliary excretion rate', fontsize=fontsize, pad=10)
    ax2.set_ylabel('k_bh (mL/min/100mL)', fontsize=fontsize)
    ax2.set_ylim(0, 3.5)
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    
    resultsfolder = os.path.dirname(output_file)
    pivot = derive_pars(output_file)
    khe_ref = pivot.loc['baseline', 'k_he_max']
    khe_rif = pivot.loc['rifampicin', 'k_he_min']
    kbh_ref = pivot.loc['baseline', 'k_bh_max']
    kbh_rif = pivot.loc['rifampicin', 'k_bh_min']
    
    for s in khe_ref.index:
        if s in khe_rif.index:
            x = ['baseline', 'rifampicin']
            khe = [khe_ref[s],khe_rif[s]]
            kbh = [kbh_ref[s],kbh_rif[s]]
        else:
            x = ['baseline']
            khe = [khe_ref[s]]
            kbh = [kbh_ref[s]]            
        ax1.plot(x, khe, 'k-', label=str(s), marker=mark[s], markersize=markersize)
        ax2.plot(x, kbh, 'k-', label=str(s), marker=mark[s], markersize=markersize)
    #ax1.legend(loc='upper center', ncol=5, prop={'size': 14})
    #ax2.legend(loc='upper center', ncol=5, prop={'size': 14})
    plot_file = os.path.join(resultsfolder, '_lineplot_max_effect.png')
    #plt.show()
    plt.savefig(fname=plot_file)
    pdf.savefig()
    plt.close()


def line_plot_effect(output_file, pdf):

    fontsize=12
    titlesize=16
    markersize=6
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8.27,11.69))
    #fig.tight_layout(pad=10.0)
    fig.subplots_adjust(
                    left=0.1,
                    bottom=0.3,
                    right=0.95,
                    top = 0.6, 
                    wspace=0.3,
                    #hspace=1,
                    )
    title = "Average effect size. \nIndividual values for hepatocellular uptake (k_he, left) and biliary excretion (k_bh, right)\n of Gadoxetate at baseline (left of plot) and after administration of rifampicin (right of plot)." 
    fig.suptitle(title, fontsize=12)
    ax1.set_title('Hepatocellular uptake rate', fontsize=fontsize, pad=10)
    ax1.set_ylabel('k_he (mL/min/100mL)', fontsize=fontsize)
    ax1.set_ylim(0, 30)
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax2.set_title('Biliary excretion rate', fontsize=fontsize, pad=10)
    ax2.set_ylabel('k_bh (mL/min/100mL)', fontsize=fontsize)
    ax2.set_ylim(0, 3.5)
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    
    resultsfolder = os.path.dirname(output_file)
    pivot = derive_pars(output_file)
    khe_ref = pivot.loc['baseline', 'k_he']
    khe_rif = pivot.loc['rifampicin', 'k_he']
    kbh_ref = pivot.loc['baseline', 'k_bh']
    kbh_rif = pivot.loc['rifampicin', 'k_bh']
    
    for s in khe_ref.index:
        if s in khe_rif.index:
            x = ['baseline', 'rifampicin']
            khe = [khe_ref[s],khe_rif[s]]
            kbh = [kbh_ref[s],kbh_rif[s]]
        else:
            x = ['baseline']
            khe = [khe_ref[s]]
            kbh = [kbh_ref[s]]            
        ax1.plot(x, khe, 'k-', label=str(s), marker=mark[s], markersize=markersize)
        ax2.plot(x, kbh, 'k-', label=str(s), marker=mark[s], markersize=markersize)
    #ax1.legend(loc='upper center', ncol=5, prop={'size': 14})
    #ax2.legend(loc='upper center', ncol=5, prop={'size': 14})
    plot_file = os.path.join(resultsfolder, '_lineplot_effect.png')
    #plt.show()
    plt.savefig(fname=plot_file)
    pdf.savefig()
    plt.close()


def max_effect_size(output_file, pdf, ref='min'):
    resultsfolder = os.path.dirname(output_file)
    pivot = derive_pars(output_file)
    khe_rif = pivot.loc['rifampicin', 'k_he_min']
    khe_base = pivot.loc['baseline', 'k_he_max']
    khe_ref = pivot.loc['baseline', 'k_he_'+ref]
    khe_eff = 100*(khe_rif-khe_base)/khe_ref
    khe_eff = khe_eff[~khe_eff.isnull()]
    kbh_rif = pivot.loc['rifampicin', 'k_bh_min']
    kbh_base = pivot.loc['baseline', 'k_bh_max']
    kbh_ref = pivot.loc['baseline', 'k_bh_'+ref]
    kbh_eff = 100*(kbh_rif-kbh_base)/kbh_ref
    kbh_eff = kbh_eff[~kbh_eff.isnull()]
    all_data = [khe_eff.values.tolist(), kbh_eff.values.tolist()]
    parameters = ['k_he', 'k_bh']
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(8.27,11.69))
    linewidth = 1.5
    fontsize=14
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(linewidth)
    fig.subplots_adjust(
                    left=0.4,
                    right=0.6,
                    bottom=0.3,
                    top = 0.7,
                    # wspace=0.4,
                    # hspace=0.4,
                    )
    # box plot
    boxprops = dict(linestyle='-', linewidth=linewidth, color='black')
    medianprops = dict(linestyle='-', linewidth=linewidth, color='black')
    whiskerprops = dict(linestyle='-', linewidth=linewidth, color='black')
    capprops = dict(linestyle='-', linewidth=linewidth, color='black')
    flierprops = dict(marker='o', markerfacecolor='white', markersize=6,
                  markeredgecolor='black', markeredgewidth=linewidth)
    bplot = ax.boxplot(all_data,
                        whis = [2.5,97.5],
                        capprops=capprops,
                        flierprops=flierprops,
                        whiskerprops=whiskerprops,
                        medianprops=medianprops,
                        boxprops=boxprops,
                        widths=0.3,
                        vert=True,  # vertical box alignment
                        patch_artist=True,  # fill with color
                        labels=parameters)  # will be used to label x-ticks
    
    title = "Rifampicin maximum effect size (perc of "+ref+")\n on hepatocellular uptake (k_he, left) and biliary excretion (k_bh, right) of Gadoxetate. \n The boxplot shows median, interquartile range and 95 percent range."
    ax.set_title(title, fontsize=12, pad=60)
    ax.set_xticklabels(labels=parameters, fontsize=fontsize)
    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=fontsize)

    # fill with colors
    for patch in bplot['boxes']:
        patch.set_facecolor('white')

    # adding horizontal grid line
    ax.yaxis.grid(True)
    ax.set_ylabel('Rifampicin maximum effect size (perc of '+ ref+')', fontsize=fontsize)
    #ax.set_ylim(-100, 20)

    plot_file = os.path.join(resultsfolder, '_drug_max_effect_ref_'+ref+'.png')
    plt.savefig(fname=plot_file)
    pdf.savefig()
    plt.close()


def pivot_table(output_file):
    resultsfolder = os.path.dirname(output_file)
    output = pd.read_csv(output_file)
    results_file = os.path.join(resultsfolder, 'pivot.csv')
    pivot = pd.pivot_table(output, values='value', columns='parameter', index=['visit', 'subject'])
    pivot.to_csv(results_file)


def drug_effect_function(output_file, pdf):

    pars = ['k_he', 'k_bh']

    resultsfolder = os.path.dirname(output_file)
    output = pd.read_csv(output_file)

    # Create box plots for aorta and liver
    subjects = output['subject'].unique()
    structures = output['structure'].unique()
    for struct in structures:
        df_struct = output[output.structure==struct]
        all_data = []
        parameters = pars
        for par in parameters:
            df_par = df_struct[df_struct.parameter==par]
            data_par = []
            for s in subjects:
                df_subj = df_par[df_par.subject==s]
                df_baseline_subj = df_subj[df_subj.visit=='baseline']
                df_rifampicin_subj = df_subj[df_subj.visit=='rifampicin'] 
                if not df_baseline_subj.empty and not df_rifampicin_subj.empty:
                    v0 = df_baseline_subj['value'].values[0]
                    v1 = df_rifampicin_subj['value'].values[0]
                    if v0 != 0:
                        data_par.append(100*(v1-v0)/v0)
            all_data.append(data_par)

    fig, ax = plt.subplots(figsize=(8.27,11.69))
    linewidth = 1.5
    fontsize=14
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(linewidth)
    fig.subplots_adjust(
                    left=0.4,
                    right=0.6,
                    bottom=0.3,
                    top = 0.7,
                    # wspace=0.4,
                    # hspace=0.4,
                    )

    # box plot
    boxprops = dict(linestyle='-', linewidth=linewidth, color='black')
    medianprops = dict(linestyle='-', linewidth=linewidth, color='black')
    whiskerprops = dict(linestyle='-', linewidth=linewidth, color='black')
    capprops = dict(linestyle='-', linewidth=linewidth, color='black')
    flierprops = dict(marker='o', markerfacecolor='white', markersize=6,
                  markeredgecolor='black', markeredgewidth=linewidth)
    bplot = ax.boxplot(all_data,
                        whis = [2.5,97.5],
                        capprops=capprops,
                        flierprops=flierprops,
                        whiskerprops=whiskerprops,
                        medianprops=medianprops,
                        boxprops=boxprops,
                        widths=0.3,
                        vert=True,  # vertical box alignment
                        patch_artist=True,  # fill with color
                        labels=parameters)  # will be used to label x-ticks
    title = "Rifampicin effect size (%)\n on hepatocellular uptake (k_he, left) and biliary excretion (k_bh, right) of Gadoxetate. \n The boxplot shows median, interquartile range and 95 percent range."
    ax.set_title(title, fontsize=12, pad=60)
    ax.set_xticklabels(labels=parameters, fontsize=fontsize)
    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=fontsize)

    # fill with colors
    for patch in bplot['boxes']:
        patch.set_facecolor('white')

    # adding horizontal grid line
    ax.yaxis.grid(True)
    ax.set_ylabel('Rifampicin effect size (%)', fontsize=fontsize)
    #ax.set_ylim(-100, 20)

    #plt.show()

    plot_file = os.path.join(resultsfolder, '_drug_effect_function.png')
    plt.savefig(fname=plot_file)
    pdf.savefig()
    plt.close()




def diurnal_k(output_file, pdf):
    resultsfolder = os.path.dirname(output_file)
    output = pd.read_csv(output_file)
    fontsize=12
    titlesize=14
    markersize=6
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(8.27,11.69))
    fig.subplots_adjust(
                    left=0.1,
                    right=0.95,
                    bottom=0.2,
                    top = 0.80, 
                    wspace=0.3,
                    #hspace=1,
                    )
    title = "Intra-day changes \nin hepatocellular uptake (k_he, top row) and biliary excretion (k_bh, bottom row) of Gadoxetate \n at baseline (left column) and after administration of rifampicin (right column). \n Full lines connect values taken in the same subject at the same day." 
    fig.suptitle(title, fontsize=12)
    ax = {
        'baselinek_he': ax1,
        'rifampicink_he': ax2,
        'baselinek_bh': ax3,
        'rifampicink_bh': ax4,
    }
    ax1.set_title('Baseline', fontsize=titlesize)
    ax1.set_xlabel('Time of day (hrs)', fontsize=fontsize)
    ax1.set_ylabel('k_he (mL/min/100mL)', fontsize=fontsize)
    ax1.set_ylim(0, 50)
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax2.set_title('Rifampicin', fontsize=titlesize)
    ax2.set_xlabel('Time of day (hrs)', fontsize=fontsize)
    ax2.set_ylabel('k_he (mL/min/100mL)', fontsize=fontsize)
    ax2.set_ylim(0, 3)
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    #ax3.set_title('Baseline', fontsize=titlesize)
    ax3.set_xlabel('Time of day (hrs)', fontsize=fontsize)
    ax3.set_ylabel('k_bh (mL/min/100mL)', fontsize=fontsize)
    ax3.set_ylim(0, 4)
    ax3.tick_params(axis='x', labelsize=fontsize)
    ax3.tick_params(axis='y', labelsize=fontsize)
    #ax4.set_title('Rifampicin', fontsize=titlesize)
    ax4.set_xlabel('Time of day (hrs)', fontsize=fontsize)
    ax4.set_ylabel('k_bh (mL/min/100mL)', fontsize=fontsize)
    ax4.set_ylim(0, 4)
    ax4.tick_params(axis='x', labelsize=fontsize)
    ax4.tick_params(axis='y', labelsize=fontsize)
    
    # Create box plots for aorta and liver
    structures = output.structure.unique()
    for struct in structures:
        df_struct = output[output.structure==struct]
        visits = df_struct.visit.unique()
        for visit in visits:
            df_visit = df_struct[df_struct.visit==visit]
            subjects = df_visit.subject.unique()
            for s in subjects:
                df_subj = df_visit[df_visit.subject==s]
                for par in ['k_he', 'k_bh']:
                    data_subj = []
                    df_par = df_subj[df_subj.parameter=='ve']
                    if not df_par.empty:
                        ve = df_par.value.values[0]
                        vh = 1-ve/100
                    if par == 'k_bh':
                        par_t = ['Kbh_i', 'Kbh_f']
                    else:
                        par_t = [par+'_i', par+'_f']
                    for p in par_t:
                        df_par = df_subj[df_subj.parameter==p]
                        if not df_par.empty:
                            v = df_par.value.values[0]
                            if par == 'k_bh':
                                v *= vh
                            data_subj.append(v)
                    t = []
                    for p in ['t0', 't3']:
                        df_par = df_subj[df_subj.parameter==p]
                        if not df_par.empty:
                            v = df_par.value.values[0]
                            t.append(v)
                    if len(data_subj) == 2:
                        ax[visit+par].plot(t, data_subj, 'k-', label=str(s), marker=mark[s], markersize=markersize)

    plot_file = os.path.join(resultsfolder, '_diurnal_function.png')
    plt.savefig(fname=plot_file)
    pdf.savefig()
    plt.close()






def line_plot_extracellular(output_file):
    resultsfolder = os.path.dirname(output_file)
    output = pd.read_csv(output_file)
    fontsize=20
    titlesize=30
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,8))
    fig.tight_layout(pad=10.0)
    fig.suptitle('Extracellular drug effects', fontsize=titlesize)
    ax1.set_title('Volume fraction', fontsize=24)
    ax1.set_ylabel('ve (mL/100mL)', fontsize=fontsize)
    ax1.set_ylim(0, 60)
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=18)
    ax2.set_title('Transit time', fontsize=24)
    ax2.set_ylabel('Te (min)', fontsize=fontsize)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.tick_params(axis='y', labelsize=18)

    ax = {
        've': ax1,
        'Te': ax2,
    }
    subjects = output['subject'].unique()
    visits = ['baseline', 'rifampicin']
    structures = output['structure'].unique()
    for struct in structures:
        df_struct = output[output.structure==struct]
        for par in ['ve', 'Te']:
            df_par = df_struct[df_struct.parameter==par]
            for s in subjects:
                df_subj = df_par[df_par.subject==s]
                x = []
                y = []
                for visit in visits:
                    df_visit = df_subj[df_subj.visit==visit]
                    if not df_visit.empty:
                        v = df_visit['value'].values[0]
                        x.append(visit)
                        y.append(v)
                ax[par].plot(x, y, 'k-', label=str(s), marker=mark[s], markersize=12)
    #ax1.legend(loc='upper center', ncol=5, prop={'size': 14})
    #ax2.legend(loc='upper center', ncol=5, prop={'size': 14})
    plot_file = os.path.join(resultsfolder, '_lineplot_extracellular.png')
    #plt.show()
    plt.savefig(fname=plot_file)
    plt.close()




def drug_effect(output_file, pars=None, name='', ylim=[-100,100]):

    resultsfolder = os.path.dirname(output_file)
    output = pd.read_csv(output_file)

    # Create box plots for aorta and liver
    subjects = output['subject'].unique()
    structures = output['structure'].unique()
    for struct in structures:
        df_struct = output[output.structure==struct]
        all_data = []
        if pars is None:
            parameters = df_struct['parameter'].unique()
        else:
            parameters = pars
        for par in parameters:
            df_par = df_struct[df_struct.parameter==par]
            data_par = []
            for s in subjects:
                df_subj = df_par[df_par.subject==s]
                df_baseline_subj = df_subj[df_subj.visit=='baseline']
                df_rifampicin_subj = df_subj[df_subj.visit=='rifampicin'] 
                if not df_baseline_subj.empty and not df_rifampicin_subj.empty:
                    v0 = df_baseline_subj['value'].values[0]
                    v1 = df_rifampicin_subj['value'].values[0]
                    if v0 != 0:
                        data_par.append(100*(v1-v0)/v0)
            all_data.append(data_par)

        fig, ax = plt.subplots(layout='constrained')

        # notch shape box plot
        bplot = ax.boxplot(all_data,
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            labels=parameters)  # will be used to label x-ticks
        ax.set_title('Drug effect on ' + struct)

        # fill with colors
        for patch in bplot['boxes']:
            patch.set_facecolor('blue')

        # adding horizontal grid line
        ax.yaxis.grid(True)
        ax.set_xlabel('Parameter')
        ax.set_ylabel('Rifampicin effect (%)')
        ax.set_ylim(ylim[0], ylim[1])

        plot_file = os.path.join(resultsfolder, '_drug_effect_' +name+ '_' + struct + '.png')
        plt.savefig(fname=plot_file)
        plt.close()


def create_box_plot(output_file, ylim={}):

    resultsfolder = os.path.dirname(output_file)
    output = pd.read_csv(output_file)

    # Create box plots for each parameter
    subjects = output['subject'].unique()
    visits = ['baseline', 'rifampicin']
    structures = output['structure'].unique()
    for struct in structures:
        df_struct = output[output.structure==struct]
        for par in df_struct['parameter'].unique():
            df = df_struct[df_struct.parameter==par]
            all_data = []
            for visit in visits:
                df_visit = df[df.visit==visit]
                data_visit = []
                for s in subjects:
                    df_visit_subj = df_visit[df_visit.subject==s]
                    if not df_visit_subj.empty:
                        val = df_visit_subj['value'].values[0]
                    data_visit.append(val)
                all_data.append(data_visit)

            fig, ax = plt.subplots(layout='constrained')

            # notch shape box plot
            bplot = ax.boxplot(all_data,
                                #notch=True,  # notch shape
                                vert=True,  # vertical box alignment
                                patch_artist=True,  # fill with color
                                labels=visits)  # will be used to label x-ticks
            ax.set_title('Drug effect on ' + par)

            # fill with colors
            colors = ['slateblue', 'coral']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

            # adding horizontal grid line
            ax.yaxis.grid(True)
            ax.set_xlabel('Visit')
            units = df.unit.unique()[0]
            ax.set_ylabel(par + ' (' + str(units) + ')')
            #ax.legend(loc='upper left', ncols=2)
            if par in ylim:
                ax.set_ylim(ylim[par][0], ylim[par][1])

            plot_file = os.path.join(resultsfolder, '_boxplot_' + struct + '_' + par + '.png')
            plt.savefig(fname=plot_file)
            plt.close()


def create_bar_chart(output_file, ylim={}):

    resultsfolder = os.path.dirname(output_file)
    output = pd.read_csv(output_file)

    # Create bar charts for each parameter
    subjects = output['subject'].unique()
    visits = ['baseline', 'rifampicin']
    structures = output['structure'].unique()
    for struct in structures:
        df_struct = output[output.structure==struct]
        for par in df_struct['parameter'].unique():
            df = df_struct[df_struct.parameter==par]
            bar_chart = {}
            for visit in visits:
                df_visit = df[df.visit==visit]
                bar_chart[visit] = []
                for s in subjects:
                    df_visit_subj = df_visit[df_visit.subject==s]
                    if df_visit_subj.empty:
                        val = np.nan
                    else:
                        val = df_visit_subj['value'].values[0]
                    bar_chart[visit].append(val)
            x = np.arange(len(subjects))  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0

            fig, ax = plt.subplots(layout='constrained')
            colors = {'baseline':'slateblue', 'rifampicin':'coral'}
            for attribute, measurement in bar_chart.items():
                offset = width * multiplier
                if measurement != np.nan:
                    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[attribute])
                    ax.bar_label(rects, padding=3)
                multiplier += 1

            # Add some text for labels, title and custom x-axis tick labels, etc.
            units = df.unit.unique()[0]
            ax.set_ylabel(par + ' (' + str(units) + ')')
            ax.set_title('Drug effect on ' + par)
            ax.set_xticks(x + width, subjects)
            ax.legend(loc='upper left', ncols=2)
            if par in ylim:
                ax.set_ylim(ylim[par][0], ylim[par][1])

            plot_file = os.path.join(resultsfolder, '_plot_' + struct + '_' + par + '.png')
            plt.savefig(fname=plot_file)
            plt.close()