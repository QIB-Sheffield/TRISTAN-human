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
            title = 'Ghent dog study',
            subtitle = 'Demo results',
            subject = 'Internal report')
    report.titlepage(doc, results)
    # TOC page
    doc.append(pl.NewPage())
    doc.append(NoEscape('\\tableofcontents'))
    doc.append(NoEscape('\\mainmatter'))


    doc.append(NoEscape('\\clearpage'))
    doc.append(pl.Command('chapter', 'Kidney biomarkers'))

    df = pd.read_pickle(os.path.join(results, 'kidneys', 'stats2.pkl'))
    with doc.create(pl.Table(position='h!')) as table:
        table.append(pl.Command('centering'))
        with table.create(pl.Tabular('ll'+'c'*(df.shape[1]-1))) as tab:
            tab.add_hline()
            tab.add_row([df.index.name] + list(df.columns))
            tab.add_hline()
            for row in df.index:
                tab.add_row([row] + list(df.loc[row,:]))
            tab.add_hline()
        table.add_caption("Results of a pair wise comparison testing for differences in kidney biomarkers between baseline visit and followup. The results are ranked by their p-value, with most significant differences at the top of the list.")

    df = pd.read_pickle(os.path.join(results, 'kidneys', 'stats1.pkl'))
    with doc.create(pl.Table(position='h!')) as table:
        table.append(pl.Command('centering'))
        with table.create(pl.Tabular('ll'+'c'*(df.shape[1]-1))) as tab:
            tab.add_hline()
            tab.add_row([df.index.name] + list(df.columns))
            tab.add_hline()
            for row in df.index:
                tab.add_row([row] + list(df.loc[row,:]))
            tab.add_hline()
        table.add_caption("Mean values along with their 95 percent confidence intervals for all kidney biomarkers at the baseline visit and at follow-up. The results are ranked by their p-value, with most significant differences at the top of the list.")


    doc.append(NoEscape('\\clearpage'))
    doc.append(pl.Command('chapter', 'Aorta biomarkers'))

    df = pd.read_pickle(os.path.join(results, 'aorta', 'stats2.pkl'))
    with doc.create(pl.Table(position='h!')) as table:
        table.append(pl.Command('centering'))
        with table.create(pl.Tabular('ll'+'c'*(df.shape[1]-1))) as tab:
            tab.add_hline()
            tab.add_row([df.index.name] + list(df.columns))
            tab.add_hline()
            for row in df.index:
                tab.add_row([row] + list(df.loc[row,:]))
            tab.add_hline()
        table.add_caption("Results of a pair wise comparison testing for differences in systemic biomarkers between baseline visit and follow-up. The results are ranked by their p-value, with most significant differences at the top of the list.")

    df = pd.read_pickle(os.path.join(results, 'aorta', 'stats1.pkl'))
    with doc.create(pl.Table(position='h!')) as table:
        table.append(pl.Command('centering'))
        with table.create(pl.Tabular('ll'+'c'*(df.shape[1]-1))) as tab:
            tab.add_hline()
            tab.add_row([df.index.name] + list(df.columns))
            tab.add_hline()
            for row in df.index:
                tab.add_row([row] + list(df.loc[row,:]))
            tab.add_hline()
        table.add_caption("Mean values along with their 95 percent confidence intervals for all systemic biomarkers at the baseline visit and after rifampicin. The results are ranked by their p-value, with most significant differences at the top of the list.")

    doc.append(NoEscape('\\clearpage'))
    doc.append(pl.Command('chapter', 'Case notes'))

    df = pd.read_pickle(os.path.join(results, 'kidneys', 'parameters.pkl'))
    dfa = pd.read_pickle(os.path.join(results, 'aorta', 'parameters_ext.pkl'))
    for i, subject in enumerate(df.subject.unique()):
        if i>0:
            doc.append(NoEscape('\\clearpage'))
        subj = str(subject).zfill(3)
        with doc.create(pl.Subsection('Subject ' + subj)):

            dfs = df[df.subject==subject]
            dfas = dfa[dfa.subject==subject]
            dfr = dfs[dfs.visit=='visit2']

            # Liver results
            with doc.create(pl.Figure(position='h!')) as pic:
                im = os.path.join(results, 'kidneys',  subj +'_baseline_Kidney_all.png')
                pic.add_image(im, width='5.5in')
                pic.add_caption("Kidney signal-time curves for subject "+subj+' at baseline.')
            if not dfr.empty:
                with doc.create(pl.Figure(position='h!')) as pic:
                    im = os.path.join(results, 'kidneys',  subj +'_visit2_Kidney_all.png')
                    pic.add_image(im, width='5.5in')
                    pic.add_caption("Kidney signal-time curves for subject "+subj+' at visit2.')

            pivot = pd.pivot_table(dfs, values='value', columns='visit', index=['name','unit'])
            cols = pivot.columns.tolist()
            if len(cols)>1:
                pivot = pivot[['visit1','visit2']]
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
                im = os.path.join(results, 'aorta',  subj +'_baseline_all.png')
                pic.add_image(im, width='5.5in')
                pic.add_caption("Aorta signal-time curves for subject "+subj+' at baseline.')
            if not dfr.empty:
                with doc.create(pl.Figure(position='h!')) as pic:
                    im = os.path.join(results, 'aorta',  subj +'_visit2_all.png')
                    pic.add_image(im, width='5.5in')
                    pic.add_caption("Aorta signal-time curves for subject "+subj+' aft visit2.')

            pivot = pd.pivot_table(dfas, values='value', columns='visit', index=['name','unit'])
            cols = pivot.columns.tolist()
            if len(cols)>1:
                pivot = pivot[['visit1','visit2']]
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
