import os
import pandas as pd
import numpy as np

import pylatex as pl
from pylatex.utils import NoEscape

import miblab_report.report as report

def generate(filename, results):

    print('Creating report..')

    report.setup(os.path.abspath(""), results)

    doc = pl.Document()
    doc.documentclass = pl.Command('documentclass',"epflreport")

    report.makecover(doc, 
            title = 'Gothenburg patient study',
            subtitle = 'Interim analysis',
            subject = 'D2.07 - Internal report')
    report.titlepage(doc, results)
    # TOC page
    doc.append(pl.NewPage())
    doc.append(NoEscape('\\tableofcontents'))
    doc.append(NoEscape('\\mainmatter'))

    doc.append(NoEscape('\\clearpage'))
    doc.append(pl.Command('chapter', 'Two-scan results'))
    with doc.create(pl.Section('Data summary')):

        with doc.create(pl.Figure(position='h!')) as pic:
            pic.append(pl.Command('centering'))
            im = os.path.join(results, 'liver_2scan', '_effect_plot.png')
            pic.add_image(im, width='7in')
            pic.add_caption("Effect size (%) on hepatocellular uptake (k_he, left) and biliary excretion (k_bh, right) of Gadoxetate. The boxplot shows median, interquartile range and 95 percent range. The line plots show individual values for hepatocellular uptake (k_he, middle) and biliary excretion (k_bh, right) of Gadoxetate at baseline (left of plot) and at visit 2 (right of plot).")

        df = pd.read_pickle(os.path.join(results, 'liver_2scan', '_table_k_stats.pkl'))
        with doc.create(pl.Table(position='h!')) as table:
            table.append(pl.Command('centering'))
            with table.create(pl.Tabular('l'+'c'*df.shape[1])) as tab:
                tab.add_hline()
                tab.add_row([df.index.name] + list(df.columns))
                tab.add_hline()
                for row in df.index:
                    tab.add_row([row] + list(df.loc[row,:]))
                tab.add_hline()
            table.add_caption("Effect size and absolute values of hepatocellular uptake (k_he) and biliary excretion (k_bh) of Gadoxetate")

        with doc.create(pl.Figure(position='h!')) as pic:
            im = os.path.join(results, 'liver_2scan',  '_diurnal_function.png')
            pic.add_image(im, width='6in')
            pic.add_caption("Intra-day changes in hepatocellular uptake (k_he, top row) and biliary excretion (k_bh, bottom row) of Gadoxetate at baseline (left column) and at visit 2 (right column). Full lines connect values taken in the same subject at the same day." )

    df = pd.read_pickle(os.path.join(results, 'liver_2scan', 'parameters_rep.pkl'))
    dfa = pd.read_pickle(os.path.join(results, 'aorta_2scan', 'parameters_ext.pkl'))
    for subject in df.subject.unique():
        doc.append(NoEscape('\\clearpage'))
        subj = str(subject).zfill(3)
        with doc.create(pl.Section('Subject ' + subj)):

            dfs = df[df.subject==subject]
            dfas = dfa[dfa.subject==subject]
            dfr = dfs[dfs.visit=='visit2']

            # Liver results
            with doc.create(pl.Figure(position='h!')) as pic:
                im = os.path.join(results, 'liver_2scan',  subj +'_visit1_Liver_all.png')
                pic.add_image(im, width='5.5in')
                pic.add_caption("Liver signal-time curves for subject "+subj+' at visit 1.')
            if not dfr.empty:
                with doc.create(pl.Figure(position='h!')) as pic:
                    im = os.path.join(results, 'liver_2scan',  subj +'_visit2_Liver_all.png')
                    pic.add_image(im, width='5.5in')
                    pic.add_caption("Liver signal-time curves for subject "+subj+' at visit 2.')  

            pivot = dfs.drop('parameter', axis=1)
            pivot = pd.pivot_table(pivot, values='value', columns='visit', index=['name','unit'])
            cols = pivot.columns.tolist()
            if len(cols)>1:
                pivot = pivot[['visit1','visit2','change (%)']]
            with doc.create(pl.Table(position='h!')) as table:
                table.append(pl.Command('centering'))
                with table.create(pl.Tabular('ll'+'c'*pivot.shape[1])) as tab:
                    tab.add_hline()
                    tab.add_row(['Biomarker', 'Units'] + list(pivot.columns))
                    tab.add_hline()
                    for row in pivot.index:
                        tab.add_row([row[0],row[1]] + list(np.around(pivot.loc[row,:].values,2)))
                    tab.add_hline()
                table.add_caption("Values for liver of subject "+subj)  

            # Aorta results
            doc.append(NoEscape('\\clearpage'))
            with doc.create(pl.Figure(position='h!')) as pic:
                im = os.path.join(results, 'aorta_2scan',  subj +'_visit1_all.png')
                pic.add_image(im, width='5.5in')
                pic.add_caption("Aorta signal-time curves for subject "+subj+' at visit 1.')
            if not dfr.empty:
                with doc.create(pl.Figure(position='h!')) as pic:
                    im = os.path.join(results, 'aorta_2scan',  subj +'_visit2_all.png')
                    pic.add_image(im, width='5.5in')
                    pic.add_caption("Aorta signal-time curves for subject "+subj+' at visit 2.')

            pivot = pd.pivot_table(dfas, values='value', columns='visit', index=['name','unit'])
            cols = pivot.columns.tolist()
            if len(cols)>1:
                pivot = pivot[['visit1','visit2','change (%)']]
            with doc.create(pl.Table(position='h!')) as table:
                table.append(pl.Command('centering'))
                with table.create(pl.Tabular('ll'+'c'*pivot.shape[1])) as tab:
                    tab.add_hline()
                    tab.add_row(['Biomarker', 'Units'] + list(pivot.columns))
                    tab.add_hline()
                    for row in pivot.index:
                        tab.add_row([row[0],row[1]] + list(np.around(pivot.loc[row,:].values,2)))
                    tab.add_hline()
                table.add_caption("Values for aorta of subject "+subj)

    doc.append(NoEscape('\\clearpage'))
    doc.append(pl.Command('chapter', 'One-scan results'))
    with doc.create(pl.Section('Data summary')):
        with doc.create(pl.Figure(position='h!')) as pic:
            pic.append(pl.Command('centering'))
            im = os.path.join(results, 'liver_1scan', '_effect_plot.png')
            pic.add_image(im, width='7in')
            pic.add_caption("Effect size (%) on hepatocellular uptake (k_he, left) and biliary excretion (k_bh, right) of Gadoxetate. The boxplot shows median, interquartile range and 95 percent range. The line plots show individual values for hepatocellular uptake (k_he, middle) and biliary excretion (k_bh, right) of Gadoxetate at baseline (left of plot) and at visit 2 (right of plot).")

        df = pd.read_pickle(os.path.join(results, 'liver_1scan', '_table_k_stats.pkl'))
        with doc.create(pl.Table(position='h!')) as table:
            table.append(pl.Command('centering'))
            with table.create(pl.Tabular('l'+'c'*df.shape[1])) as tab:
                tab.add_hline()
                tab.add_row([df.index.name] + list(df.columns))
                tab.add_hline()
                for row in df.index:
                    tab.add_row([row] + list(df.loc[row,:]))
                tab.add_hline()
            table.add_caption("Effect size and absolute values of hepatocellular uptake (k_he) and biliary excretion (k_bh) of Gadoxetate")

    df = pd.read_pickle(os.path.join(results, 'liver_1scan', 'parameters_ext.pkl'))
    dfa = pd.read_pickle(os.path.join(results, 'aorta_1scan', 'parameters_ext.pkl'))
    for subject in df.subject.unique():
        doc.append(NoEscape('\\clearpage'))
        subj = str(subject).zfill(3)
        with doc.create(pl.Section('Subject ' + subj)):

            dfs = df[df.subject==subject]
            dfas = dfa[dfa.subject==subject]
            dfr = dfs[dfs.visit=='visit2']

            # Liver results
            with doc.create(pl.Figure(position='h!')) as pic:
                im = os.path.join(results, 'liver_1scan',  subj +'_visit1_Liver_all.png')
                pic.add_image(im, width='5.5in')
                pic.add_caption("Liver signal-time curves for subject "+subj+' at baseline.')
            if not dfr.empty:
                with doc.create(pl.Figure(position='h!')) as pic:
                    im = os.path.join(results, 'liver_1scan',  subj +'_visit2_Liver_all.png')
                    pic.add_image(im, width='5.5in')
                    pic.add_caption("Liver signal-time curves for subject "+subj+' at visit 2.')

            pivot = pd.pivot_table(dfs, values='value', columns='visit', index=['name','unit'])
            cols = pivot.columns.tolist()
            if len(cols)>1:
                pivot = pivot[['visit1','visit2','change (%)']]
            with doc.create(pl.Table(position='h!')) as table:
                table.append(pl.Command('centering'))
                with table.create(pl.Tabular('ll'+'c'*pivot.shape[1])) as tab:
                    tab.add_hline()
                    tab.add_row(['Biomarker', 'Units'] + list(pivot.columns))
                    tab.add_hline()
                    for row in pivot.index:
                        tab.add_row([row[0],row[1]] + list(np.around(pivot.loc[row,:].values,2)))
                    tab.add_hline()
                table.add_caption("Values for liver of subject "+subj)

            # Aorta results
            doc.append(NoEscape('\\clearpage'))
            with doc.create(pl.Figure(position='h!')) as pic:
                im = os.path.join(results, 'aorta_1scan',  subj +'_visit1_all.png')
                pic.add_image(im, width='5.5in')
                pic.add_caption("Aorta signal-time curves for subject "+subj+' at baseline.')
            if not dfr.empty:
                with doc.create(pl.Figure(position='h!')) as pic:
                    im = os.path.join(results, 'aorta_1scan',  subj +'_visit2_all.png')
                    pic.add_image(im, width='5.5in')
                    pic.add_caption("Aorta signal-time curves for subject "+subj+' at visit 2.')

            pivot = pd.pivot_table(dfas, values='value', columns='visit', index=['name','unit'])
            cols = pivot.columns.tolist()
            if len(cols)>1:
                pivot = pivot[['visit1','visit2','change (%)']]
            with doc.create(pl.Table(position='h!')) as table:
                table.append(pl.Command('centering'))
                with table.create(pl.Tabular('ll'+'c'*pivot.shape[1])) as tab:
                    tab.add_hline()
                    tab.add_row(['Biomarker', 'Units'] + list(pivot.columns))
                    tab.add_hline()
                    for row in pivot.index:
                        tab.add_row([row[0],row[1]] + list(np.around(pivot.loc[row,:].values,2)))
                    tab.add_hline()
                table.add_caption("Values for aorta of subject "+subj)

    report.create(doc, os.path.abspath(""), filename, results)


if __name__ == "__main__":
    generate()
