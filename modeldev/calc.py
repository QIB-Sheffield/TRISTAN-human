import os
import math
import pandas as pd
import numpy as np
import pingouin as pg


def _derive_rates(output):
    # Calculate derived excretion rates
    visits = output.visit.unique()
    structure = 'liver'
    for subject in output.subject.unique():
        df_subj = output[output.subject==subject]
        for visit in visits:
            df = df_subj[df_subj.visit==visit]
            df = df[df.structure==structure]
            khe = df[df.parameter.isin(['k_he_i','k_he_f'])]
            khe = khe[khe.notna()]
            if len(khe)==2:
                khe_max = np.amax(khe.value.values)
                khe_min = np.amin(khe.value.values)
                output.loc[len(output)] = [subject, visit, structure, 'Hepatocellular uptake rate (max)', khe_max, 'mL/min/100mL', 'k_he_max']
                output.loc[len(output)] = [subject, visit, structure, 'Hepatocellular uptake rate (max)', khe_min, 'mL/min/100mL', 'k_he_min']
            Kbh = df[df.parameter.isin(['Kbh_i','Kbh_f'])]
            Kbh = Kbh[Kbh.notna()]
            if len(Kbh)==2:
                ve = df[df.parameter=='ve'].value.values[0]
                Kbh_i = Kbh[Kbh.parameter=='Kbh_i'].value.values[0]
                Kbh_f = Kbh[Kbh.parameter=='Kbh_f'].value.values[0]
                kbh_i = Kbh_i * (1-ve/100)
                kbh_f = Kbh_f * (1-ve/100)
                kbh_max = np.amax([kbh_i, kbh_f])
                kbh_min = np.amin([kbh_i, kbh_f])
                output.loc[len(output)] = [subject, visit, structure, 'Biliary excretion rate (initial)', kbh_i, 'mL/min/100mL', 'k_bh_i']
                output.loc[len(output)] = [subject, visit, structure, 'Biliary excretion rate (final)', kbh_f, 'mL/min/100mL', 'k_bh_f']
                output.loc[len(output)] = [subject, visit, structure, 'Biliary excretion rate (max)', kbh_max, 'mL/min/100mL', 'k_bh_max']
                output.loc[len(output)] = [subject, visit, structure, 'Biliary excretion rate (min)', kbh_min, 'mL/min/100mL', 'k_bh_min']
    return output


def _derive_effect_sizes(output):
    # Calculate effect sizes
    for subject in output.subject.unique():
        df_subj = output[output.subject==subject]
        for structure in df_subj.structure.unique():
            df_struct = df_subj[df_subj.structure==structure]
            for par in df_struct.parameter.unique():
                df_par = df_struct[df_struct.parameter==par]
                if len(df_par)==2:
                    visits = df_par.visit.unique()
                    v0 = df_par[df_par.visit==visits[0]].value.values[0]
                    v1 = df_par[df_par.visit==visits[1]].value.values[0]
                    effect = 100*np.divide(v1-v0, v0)
                    name = df_par[df_par.visit==visits[0]].name.values[0]
                    unit = df_par[df_par.visit==visits[0]].unit.values[0]
                    output.loc[len(output)] = [subject, 'change (%)', structure, name, effect, unit, par + ' effect size']
    return output


def _derive_max_effect(output):
    # Calculate maximum and minimum effect size for khe and kbh
    structure = 'liver'
    visits = output.visit[output.visit!='change (%)'].unique()
    for subject in output.subject.unique():
        df = output[output.subject==subject]
        khe_min1 = df[(df.visit==visits[1]) & (df.parameter=='k_he_min')].value.values
        kbh_min1 = df[(df.visit==visits[1]) & (df.parameter=='k_bh_min')].value.values
        khe_max0 = df[(df.visit==visits[0]) & (df.parameter=='k_he_max')].value.values
        kbh_max0 = df[(df.visit==visits[0]) & (df.parameter=='k_bh_max')].value.values
        if len(khe_max0)>0 and len(khe_min1>0):
            khe_eff = 100*(khe_min1[0]-khe_max0[0])/khe_max0[0]
            kbh_eff = 100*(kbh_min1[0]-kbh_max0[0])/kbh_max0[0]
            output.loc[len(output)] = [subject, 'change (%)', structure, 'Hepatocellular uptake rate (max effect)', khe_eff, 'mL/min/100mL', 'k_he max effect size']
            output.loc[len(output)] = [subject, 'change (%)', structure, 'Biliary excretion rate (max effect)', kbh_eff, 'mL/min/100mL', 'k_bh max effect size']
    return output


def derive_liver_pars(src):
    output_file = os.path.join(src, 'parameters.csv')
    output = pd.read_csv(output_file)
    # df = output[output.parameter.isin(['k_he_i','k_he_f','Kbh_i','Kbh_f'])]
    # output.drop(index=df.index, inplace=True)
    output = _derive_rates(output)
    output = _derive_effect_sizes(output)
    #output = _derive_max_effect(output)
    output.to_csv(os.path.join(src, 'parameters_ext.csv'))
    output.to_pickle(os.path.join(src, 'parameters_ext.pkl'))


def derive_aorta_pars(src):
    output_file = os.path.join(src, 'parameters.csv')
    output = pd.read_csv(output_file)
    output = _derive_effect_sizes(output)
    #output = _derive_max_effect(output)
    todrop = ['S02','BAT1','BAT2']
    todrop += [v + ' effect size' for v in todrop]
    df = output[output.parameter.isin(todrop)]
    output.drop(index=df.index, inplace=True)
    output.reset_index(drop=True, inplace=True)
    output.to_csv(os.path.join(src, 'parameters_ext.csv'))
    output.to_pickle(os.path.join(src, 'parameters_ext.pkl'))


def report_pars(src):
    output = pd.read_pickle(os.path.join(src, 'parameters_ext.pkl'))
    todrop = ['S02','t0','t1','t2','t3','dt1','dt2',
              'k_he_i','k_he_f','k_he_min','k_he_max',
              'k_bh_i','k_bh_f','k_bh_min','k_bh_max', 
              'Khe_i','Khe_f','Kbh_i','Kbh_f']
    todrop += [v + ' effect size' for v in todrop]
    df = output[output.parameter.isin(todrop)]
    output.drop(index=df.index, inplace=True)
    output.reset_index(drop=True, inplace=True)
    output.to_csv(os.path.join(src, 'parameters_rep.csv'))
    output.to_pickle(os.path.join(src, 'parameters_rep.pkl'))

def desc_stats(src):
    output_file = os.path.join(src, 'parameters_ext.pkl')
    output = pd.read_pickle(output_file)
    visits = output.visit[output.visit!='change (%)'].unique()
    v0 = output[output.visit==visits[0]]
    v0 = pd.pivot_table(v0, values='value', index='subject', columns='parameter')
    v1 = output[output.visit==visits[1]]
    v1 = pd.pivot_table(v1, values='value', index='subject', columns='parameter')
    ef = output[output.visit=='change (%)']
    ef = pd.pivot_table(ef, values='value', index='subject', columns='parameter')
    v0.to_csv(os.path.join(src, '_table_'+visits[0]+'.csv'))
    v1.to_csv(os.path.join(src, '_table_'+visits[1]+'.csv'))
    ef.to_csv(os.path.join(src, '_table_effect.csv'))
    # Calculate stats
    bstats = v0.describe()
    bstats = bstats[['k_he', 'k_bh']].round(2)
    bstats = bstats.rename(columns={"k_he": "k_he "+visits[0]+" (mL/min/100mL)", "k_bh": "k_bh "+visits[0]+" (mL/min/100mL)"})
    rstats = v1.describe()
    rstats = rstats[['k_he', 'k_bh']].round(2)
    rstats = rstats.rename(columns={"k_he": "k_he "+visits[1]+" (mL/min/100mL)", "k_bh": "k_bh "+visits[1]+" (mL/min/100mL)"})
    estats = ef.describe()
    estats = estats[['k_he effect size', 'k_bh effect size']].round(1)
    estats = estats.rename(columns={"k_he effect size": "k_he effect size (%)", "k_bh effect size": "k_bh effect size (%)"})
    stats = pd.concat([estats.T, bstats.T, rstats.T])
    stats=stats.reindex([
        "k_he effect size (%)",
        "k_he "+visits[0]+" (mL/min/100mL)",
        "k_he "+visits[1]+" (mL/min/100mL)",
        "k_bh effect size (%)",
        "k_bh "+visits[0]+" (mL/min/100mL)",
        "k_bh "+visits[1]+" (mL/min/100mL)",
        ])
    stats.to_csv(os.path.join(src, '_table_k_stats.csv'))
    stats.to_pickle(os.path.join(src, '_table_k_stats.pkl'))



def ttest(src, filename):
    
    df = pd.read_pickle(os.path.join(src, filename))
    df.unit.fillna(' ', inplace=True)
    visits = df.visit[df.visit!='change (%)'].unique()

    # Get mean, sdev and count of all variables
    avr = pd.pivot_table(df, values='value', columns='visit', index=['structure','name','unit'], aggfunc='mean')
    std = pd.pivot_table(df, values='value', columns='visit', index=['structure','name','unit'], aggfunc='std')
    cnt = pd.pivot_table(df, values='value', columns='visit', index=['structure','name','unit'], aggfunc='count')

    avr = avr.sort_values('name')
    std = std.sort_values('name')
    cnt = cnt.sort_values('name')

    # Calculate 95% CI intervals
    b_avr = around_sig(avr[visits[0]].values, 3)
    r_avr = around_sig(avr[visits[1]].values, 3)
    c_avr = around_sig(avr['change (%)'].values, 3)
    b_err = around_sig(1.96*std[visits[0]].values / np.sqrt(cnt[visits[0]].values), 2)
    r_err = around_sig(1.96*std[visits[1]].values / np.sqrt(cnt[visits[1]].values), 2)
    c_err = around_sig(1.96*std['change (%)'].values / np.sqrt(cnt['change (%)'].values), 2)

    # Perform t-tests
    output = None
    df = df[df.visit != 'change (%)']
    for par in df.name.unique():
        dfp = df[df.name==par]
        stats = pg.pairwise_tests(data=dfp, dv='value', within='visit', subject='subject', return_desc=False, effsize='odds-ratio')
        stats['name'] = [par]
        if output is None:
            output = stats
        else:
            output = pd.concat([output, stats])

    # Update output array
    output = output.sort_values('name')
    output['unit'] = [i[2] for i in avr.index.tolist()]
    output[visits[0]] = [str(b_avr[i]) + ' (' + str(b_err[i]) + ') ' for i in range(b_avr.size)]
    output[visits[1]] = [str(r_avr[i]) + ' (' + str(r_err[i]) + ') ' for i in range(r_avr.size)]
    output['change (%)'] = [str(c_avr[i]) + ' (' + str(c_err[i]) + ') ' for i in range(c_avr.size)]
    output.reset_index(drop=True, inplace=True)
    output.drop(columns=['Contrast', 'A', 'B', 'Paired', 'Parametric', 'T', 'dof', 'alternative'], inplace=True)
    output = output[['name', 'unit', visits[0], visits[1], 'change (%)', 'p-unc','BF10', 'odds-ratio']]
    output.rename(columns={"name": "Biomarker", "unit": "Units", "p-unc": "p-value", 'BF10': 'Bayes Factor', 'odds-ratio': 'Odds Ratio'}, inplace=True)
    output = output.sort_values('p-value', ascending=True)
    output.set_index('Biomarker', inplace=True)
    output1 = output[['Units', visits[0], visits[1], 'change (%)']]
    output2 = output[['Units', "p-value", 'Bayes Factor', 'Odds Ratio']]
    output2.loc[:,'p-value'] = np.around(output2['p-value'].values, 5)
    # #output2.loc[:,'Bayes Factor'] = np.around(output2['Bayes Factor'].values, 2)
    #output2.loc[:,'Odds Ratio'] = around_sig(output2['Odds Ratio'].values, 3)
    output1.to_csv(os.path.join(src, 'stats1.csv'))
    output1.to_pickle(os.path.join(src, 'stats1.pkl'))
    output2.to_csv(os.path.join(src, 'stats2.csv'))
    output2.to_pickle(os.path.join(src, 'stats2.pkl'))


def first_digit(x):
    return -int(math.floor(math.log10(abs(x))))
def round_sig(x, n):
    # Round to n significant digits
    if x==0:
        return x
    return round(x, first_digit(x) + (n-1))
def round_to_first_digit(x):
    # Round to first significant digit
    n = first_digit(x)
    y = round(x, n)
    if n<=0:
        return int(y)
    return y
def round_meas(x, xerr):
    # Round measurement with known error
    if xerr==0:
        return x, xerr
    n = first_digit(xerr)
    y = round(x, n)
    yerr = round(xerr, n)
    if n<=0:
        return int(y), int(yerr)
    else:
        return y, yerr
def around_sig(x, n):
    return np.array([round_sig(v,n) for v in x])
def around_meas(x, xerr):
    n = len(x)
    if len(xerr) != n:
        raise ValueError(
            'The array with error values must have the same length as the array with measurements.'
        )
    y = np.empty(n)
    yerr = np.empty(n)
    for i in range(n):
        y[i], yerr[i] = round_meas(x[i], xerr[i])
    return y, yerr

if __name__ == '__main__':
    assert round_to_first_digit(0.0163) == 0.02
    assert round_to_first_digit(0.163) == 0.2
    assert round_to_first_digit(1.63) == 2
    assert round_to_first_digit(16.3) == 20
    assert round_to_first_digit(163) == 200
    assert round_meas(123.654, 0.0678) == (123.65, 0.07)
    assert round_meas(123.654, 0.678) == (123.7, 0.7)
    assert round_meas(123.654, 6.78) == (124, 7)
    assert round_meas(123.654, 67.8) == (120, 70)
    assert round_meas(123.654, 678) == (100, 700)
    assert round_meas(123.654, 6780) == (0, 7000)




