import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def diurnal_changes(output_file, parameters):

    resultsfolder = os.path.dirname(output_file)
    output = pd.read_csv(output_file)

    # Create box plots for aorta and liver
    all_data = []
    structures = output.structure.unique()
    for struct in structures:
        df_struct = output[output.structure==struct]
        visits = df_par.visit.unique()
        for visit in visits:
            df_visit = df_par[df_par.visit==visit]
            subjects = df_visit.subjects.unique()
            for s in subjects:
                df_subj = df_visit[df_visit.subject==s]
                if not df_subj.empty:
                    v = df_subj.loc['parameters','value'].values
                    all_data.append(v)

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
        ax.set_ylim(-100, 100)

        plot_file = os.path.join(resultsfolder, '_drug_effect_' + struct + '.png')
        plt.savefig(fname=plot_file)
        plt.close()

def drug_effect(output_file):

    resultsfolder = os.path.dirname(output_file)
    output = pd.read_csv(output_file)

    # Create box plots for aorta and liver
    subjects = output['subject'].unique()
    structures = output['structure'].unique()
    for struct in structures:
        df_struct = output[output.structure==struct]
        all_data = []
        parameters = df_struct['parameter'].unique()
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
        ax.set_ylim(-100, 100)

        plot_file = os.path.join(resultsfolder, '_drug_effect_' + struct + '.png')
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