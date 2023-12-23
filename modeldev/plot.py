import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
clr = {
    1: 'tab:blue',
    2: 'tab:orange',
    3: 'tab:green',
    4: 'tab:red',
    5: 'tab:purple',
    6: 'tab:brown',
    7: 'tab:pink',
    8: 'tab:gray',
    9: 'tab:olive',
    10: 'tab:cyan',
}


def _line_plots(src, ax1, ax2, ylim=[50,5]):

    output = pd.read_pickle(os.path.join(src, 'parameters_ext.pkl'))
    visits = output.visit[output.visit!='change (%)'].unique()

    fontsize=10
    markersize=6
    ax1.set_title('Hepatocellular uptake rate', fontsize=fontsize, pad=10)
    ax1.set_ylabel('k_he (mL/min/100mL)', fontsize=fontsize)
    ax1.set_ylim(0, ylim[0])
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax2.set_title('Biliary excretion rate', fontsize=fontsize, pad=10)
    ax2.set_ylabel('k_bh (mL/min/100mL)', fontsize=fontsize)
    ax2.set_ylim(0, ylim[1])
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    
    pivot = pd.pivot_table(output[output.visit==visits[0]], values='value', columns='parameter', index='subject')
    khe_ref = pivot.loc[:, 'k_he']
    kbh_ref = pivot.loc[:, 'k_bh']
    pivot = pd.pivot_table(output[output.visit==visits[1]], values='value', columns='parameter', index='subject')
    khe_rif = pivot.loc[:, 'k_he']
    kbh_rif = pivot.loc[:, 'k_bh']
    
    for s in khe_ref.index:
        if s in khe_rif.index:
            x = [visits[0], visits[1]]
            khe = [khe_ref[s],khe_rif[s]]
            kbh = [kbh_ref[s],kbh_rif[s]]
        else:
            x = [visits[0]]
            khe = [khe_ref[s]]
            kbh = [kbh_ref[s]]            
        ax1.plot(x, khe, '-', label=str(s), marker=mark[s], markersize=markersize, color=clr[s])
        ax2.plot(x, kbh, '-', label=str(s), marker=mark[s], markersize=markersize, color=clr[s])


def _effect_box_plots(src, ax):

    pars = ['k_he', 'k_bh']
    df = pd.read_pickle(os.path.join(src, 'parameters_ext.pkl'))
    all_data = []
    for par in ['k_he effect size', 'k_bh effect size']:
        data = df[df.parameter==par].value.values.tolist()
        all_data.append(data)

    linewidth = 1.0
    fontsize=10
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(linewidth)

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
                        labels=pars)  # will be used to label x-ticks
    ax.set_xticklabels(labels=pars, fontsize=fontsize)
    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=fontsize)

    # fill with colors
    for patch in bplot['boxes']:
        patch.set_facecolor('lightsteelblue')

    # adding horizontal grid line
    ax.yaxis.grid(True)
    ax.set_ylabel('Effect size (%)', fontsize=fontsize)


def effect_plot(src, ylim=[50,5]):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, width_ratios=[2, 4, 4], figsize=(8,3))
    fig.subplots_adjust(wspace=0.5)
    _effect_box_plots(src, ax0)
    _line_plots(src, ax1, ax2, ylim=ylim)
    plt.savefig(fname=os.path.join(src, '_effect_plot.png'))
    plt.close()


def _max_line_plots(output_file, ax1, ax2):

    resultsfolder = os.path.dirname(output_file)
    output = pd.read_pickle(os.path.join(resultsfolder, 'parameters_ext.pkl'))
    visits = output.visit[output.visit!='change (%)'].unique()

    fontsize=12
    markersize=6
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

    pivot = pd.pivot_table(output[output.visit==visits[0]], values='value', columns='parameter', index='subject')
    khe_ref = pivot.loc[:, 'k_he_max']
    kbh_ref = pivot.loc[:, 'k_bh_max']
    pivot = pd.pivot_table(output[output.visit==visits[1]], values='value', columns='parameter', index='subject')
    khe_rif = pivot.loc[:, 'k_he_min']
    kbh_rif = pivot.loc[:, 'k_bh_min']
    
    for s in khe_ref.index:
        if s in khe_rif.index:
            x = visits
            khe = [khe_ref[s],khe_rif[s]]
            kbh = [kbh_ref[s],kbh_rif[s]]
        else:
            x = [visits[0]]
            khe = [khe_ref[s]]
            kbh = [kbh_ref[s]]            
        ax1.plot(x, khe, '-', label=str(s), marker=mark[s], markersize=markersize, color=clr[s])
        ax2.plot(x, kbh, '-', label=str(s), marker=mark[s], markersize=markersize, color=clr[s])
    #ax1.legend(loc='upper center', ncol=5, prop={'size': 14})
    #ax2.legend(loc='upper center', ncol=5, prop={'size': 14})


def _max_effect_box_plots(output_file, ax):

    resultsfolder = os.path.dirname(output_file)
    output = pd.read_pickle(os.path.join(resultsfolder, 'parameters_ext.pkl'))
    khe = output[output.parameter=='k_he max effect size 1'].value.values.tolist()
    kbh = output[output.parameter=='k_bh max effect size 1'].value.values.tolist()
    all_data = [khe, kbh]
    parameters = ['k_he', 'k_bh']
    
    # Create box plot
    linewidth = 1.0
    fontsize=10
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(linewidth)

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
    
    ax.set_xticklabels(labels=parameters, fontsize=fontsize)
    ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=fontsize)

    # fill with colors
    for patch in bplot['boxes']:
        patch.set_facecolor('white')

    # adding horizontal grid line
    ax.yaxis.grid(True)
    ax.set_ylabel('Maximum effect size (%)', fontsize=fontsize)
    #ax.set_ylim(-100, 20)


def max_effect_plot(output_file):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, width_ratios=[2, 4, 4], figsize=(8,3))
    fig.subplots_adjust(wspace=0.5)
    _max_effect_box_plots(output_file, ax0)
    _max_line_plots(output_file, ax1, ax2)
    resultsfolder = os.path.dirname(output_file)
    plt.savefig(fname=os.path.join(resultsfolder, '_max_effect_plot.png'))
    plt.close()


def tmax_effect(output_file, pdf):
    # Read parameter file
    output = pd.read_csv(output_file)
    output = output[output.structure=='liver']
    # Set up figure
    fontsize=12
    titlesize=14
    markersize=2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(8.27,11.69))
    fig.subplots_adjust(
                    left=0.1,
                    right=0.95,
                    bottom=0.2,
                    top = 0.80, 
                    wspace=0.3,
                    #hspace=1,
                    )
    title = "Effect of changing acquisition time \non hepatocellular uptake (k_he, top row) and biliary excretion (k_bh, bottom row) of Gadoxetate \n at baseline (left column) and after administration of rifampicin (right column). \n Full lines connect values taken in the same subject." 
    fig.suptitle(title, fontsize=12)
    ax = {
        'baselinek_he': ax1,
        'rifampicink_he': ax2,
        'baselinek_bh': ax3,
        'rifampicink_bh': ax4,
    }
    ax1.set_title('Baseline', fontsize=titlesize)
    ax1.set_xlabel('Scan time (mins)', fontsize=fontsize)
    ax1.set_ylabel('k_he (mL/min/100mL)', fontsize=fontsize)
    ax1.set_ylim(0, 50)
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax2.set_title('Rifampicin', fontsize=titlesize)
    ax2.set_xlabel('Scan time (mins)', fontsize=fontsize)
    ax2.set_ylabel('k_he (mL/min/100mL)', fontsize=fontsize)
    ax2.set_ylim(0, 3)
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    #ax3.set_title('Baseline', fontsize=titlesize)
    ax3.set_xlabel('Scan time (mins)', fontsize=fontsize)
    ax3.set_ylabel('k_bh (mL/min/100mL)', fontsize=fontsize)
    ax3.set_ylim(0, 4)
    ax3.tick_params(axis='x', labelsize=fontsize)
    ax3.tick_params(axis='y', labelsize=fontsize)
    #ax4.set_title('Rifampicin', fontsize=titlesize)
    ax4.set_xlabel('Scan time (mins)', fontsize=fontsize)
    ax4.set_ylabel('k_bh (mL/min/100mL)', fontsize=fontsize)
    ax4.set_ylim(0, 4)
    ax4.tick_params(axis='x', labelsize=fontsize)
    ax4.tick_params(axis='y', labelsize=fontsize)
    
    # Create plots
    for visit in output.visit.unique():
        for subject in output.subject.unique():
            df = output[output.visit==visit]
            df = df[df.subject==subject]
            if not df.empty:
                df = pd.pivot(df, values='value', columns='parameter', index='tmax') 
                for par in ['k_he', 'k_bh']:
                    ax[visit+par].plot(df.index.values/60, df[par].values, 'k-', label=str(subject), marker=mark[subject], markersize=markersize)

    # Export results
    resultsfolder = os.path.dirname(output_file)
    plot_file = os.path.join(resultsfolder, '_tmax.png')
    plt.savefig(fname=plot_file)
    pdf.savefig()
    #plt.show()
    plt.close()
                

def diurnal_k(src, ylim=[50,6]):
    output = pd.read_pickle(os.path.join(src, 'parameters_ext.pkl'))
    visits = output.visit[output.visit!='change (%)'].unique()

    fontsize=10
    titlesize=12
    markersize=6
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(8,8))
    fig.subplots_adjust(
                    left=0.1,
                    right=0.9,
                    bottom=0.1,
                    top = 0.9, 
                    wspace=0.3,
                    #hspace=1,
                    )
    ax = {
        visits[0]+'k_he': ax1,
        visits[1]+'k_he': ax2,
        visits[0]+'k_bh': ax3,
        visits[1]+'k_bh': ax4,
    }
    ax1.set_title(visits[0], fontsize=titlesize)
    ax1.set_xlabel('Time of day (hrs)', fontsize=fontsize)
    ax1.set_ylabel('k_he (mL/min/100mL)', fontsize=fontsize)
    ax1.set_ylim(0, ylim[0])
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax2.set_title(visits[1], fontsize=titlesize)
    ax2.set_xlabel('Time of day (hrs)', fontsize=fontsize)
    ax2.set_ylabel('k_he (mL/min/100mL)', fontsize=fontsize)
    ax2.set_ylim(0, ylim[0])
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    #ax3.set_title('Baseline', fontsize=titlesize)
    ax3.set_xlabel('Time of day (hrs)', fontsize=fontsize)
    ax3.set_ylabel('k_bh (mL/min/100mL)', fontsize=fontsize)
    ax3.set_ylim(0, ylim[1])
    ax3.tick_params(axis='x', labelsize=fontsize)
    ax3.tick_params(axis='y', labelsize=fontsize)
    #ax4.set_title('Rifampicin', fontsize=titlesize)
    ax4.set_xlabel('Time of day (hrs)', fontsize=fontsize)
    ax4.set_ylabel('k_bh (mL/min/100mL)', fontsize=fontsize)
    ax4.set_ylim(0, ylim[1])
    ax4.tick_params(axis='x', labelsize=fontsize)
    ax4.tick_params(axis='y', labelsize=fontsize)

    # Create box plots
    for visit in visits:
        df_visit = output[output.visit==visit]
        subjects = df_visit.subject.unique()
        for s in subjects:
            df_subj = df_visit[df_visit.subject==s]
            for par in ['k_he', 'k_bh']:
                data_subj = []
                for p in [par+'_i', par+'_f']:
                    df_par = df_subj[df_subj.parameter==p]
                    if not df_par.empty:
                        v = df_par.value.values[0]
                        data_subj.append(v)
                t = []
                for p in ['t0', 't3']:
                    df_par = df_subj[df_subj.parameter==p]
                    if not df_par.empty:
                        v = df_par.value.values[0]
                        t.append(v)
                if len(data_subj) == 2:
                    ax[visit+par].plot(t, data_subj, '-', label=str(s), marker=mark[s], markersize=markersize, color=clr[s])

    plot_file = os.path.join(src, '_diurnal_function.png')
    plt.savefig(fname=plot_file)
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
    visits = output.visit[output.visit!='change (%)'].unique()
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
    visits = output.visit[output.visit!='change (%)'].unique()
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
    visits = output.visit[output.visit!='change (%)'].unique()

    # Create bar charts for each parameter
    subjects = output['subject'].unique()
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
            colors = {visits[0]:'slateblue', visits[1]:'coral'}
            for attribute, measurement in bar_chart.items():
                offset = width * multiplier
                if measurement != np.nan:
                    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[attribute])
                    ax.bar_label(rects, padding=3)
                multiplier += 1

            # Add some text for labels, title and custom x-axis tick labels, etc.
            units = df.unit.unique()[0]
            ax.set_ylabel(par + ' (' + str(units) + ')')
            ax.set_title('Values per visit ' + par)
            ax.set_xticks(x + width, subjects)
            ax.legend(loc='upper left', ncols=2)
            if par in ylim:
                ax.set_ylim(ylim[par][0], ylim[par][1])
            
            plot_file = os.path.join(resultsfolder, '_plot_' + par + '_' + struct + '.png')
            plt.savefig(fname=plot_file)
            plt.close()


# def effect_size(resultsfolder):
#     stats = pd.read_pickle(os.path.join(resultsfolder, '_table_k_stats.pkl'))
#     # Save table in pdf report
#     fig, ax = plt.subplots(figsize=(11, 4))
#     fig.subplots_adjust(left=0.1, right=1.0, bottom=0.1, top=0.9)
#     ax.axis('tight')
#     ax.axis('off')
#     table = ax.table(cellText=stats.values,colLabels=stats.columns,rowLabels=stats.index.values,loc='center', cellLoc='center')
#     table.auto_set_font_size(False)
#     table.set_fontsize(12)
#     table.scale(0.65, 1.7)
#     #plt.show()
#     plt.savefig(fname=os.path.join(resultsfolder, 'effect_size.png' ))
#     plt.close()

if __name__ == "__main__":
    tmax_effect()