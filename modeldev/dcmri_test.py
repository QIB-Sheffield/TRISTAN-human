import numpy as np
import matplotlib.pyplot as plt
import dcmri

tmax = 120 # sec
dt = 0.01 # sec
MTT = 20 # sec

weight = 70.0           # Patient weight in kg
conc = 0.25             # mmol/mL (https://www.bayer.com/sites/default/files/2020-11/primovist-pm-en.pdf)
dose = 0.025            # mL per kg bodyweight (quarter dose)
rate = 1                # Injection rate (mL/sec)
start = 20.0        # sec
dispersion = 90    # %


def test_res_comp():
    t = np.arange(0, tmax, dt)
    J = dcmri.injection(t, weight, conc, dose, rate, start)
    C = dcmri.res_comp(t, J, 0)
    assert np.amax(C) == 0
    C = dcmri.res_comp(t, J, np.inf)
    assert round(np.amax(C),1) == round(dose*weight*conc,1)
    plt.plot(t, C)
    plt.show()

def test_uconv():
    t = np.arange(0, tmax, dt)
    Ji = dcmri.injection(t, weight, conc, dose, rate, start)
    Jo = dcmri.expconv(MTT, t, Ji)
    Jo2 = dcmri.uconv(dt, Ji, np.exp(-t/MTT)/MTT)
    assert 100*np.linalg.norm(Jo-Jo2)/np.linalg.norm(Jo) < 1e-3
    plt.plot(t, Jo, 'r-')
    plt.plot(t, Jo2, 'b--')
    plt.show()

def test_chain_residue():
    t = np.arange(0, tmax, dt)
    #assert 100*np.linalg.norm(Jo-Jo2)/np.linalg.norm(Jo) < 1e-3
    plt.plot(t, dcmri.comp_residue(t, MTT), 'r-')
    plt.plot(t, dcmri.plug_residue(t, MTT), 'r-')
    #plt.plot(t, dcmri.chain_residue(t, MTT, 0), 'b--')
    #plt.plot(t, dcmri.chain_residue(t, MTT, 0.1), 'b--')
    plt.plot(t, dcmri.chain_residue(t, MTT, 1.0), 'b--')
    #plt.plot(t, dcmri.chain_residue(t, MTT, 1), 'b--')
    #plt.plot(t, dcmri.chain_residue(t, MTT, 10), 'b--')
    #plt.plot(t, dcmri.chain_residue(t, MTT, 100), 'b--')
    plt.show()

def test_ures_chain():
    t = np.arange(0, tmax, dt)
    Ji = dcmri.injection(t, weight, conc, dose, rate, start)
    Ccm = dcmri.res_comp(t, Ji, MTT)
    Cpl = dcmri.res_plug(t, Ji, MTT)
    Cch = dcmri.ures_chain(dt, Ji, MTT, 10)
    #assert 100*np.linalg.norm(Jo-Jo2)/np.linalg.norm(Jo) < 1e-3
    plt.plot(t, Ccm, 'r-')
    plt.plot(t, Cpl, 'r-')
    plt.plot(t, Cch, 'b--')
    plt.show()


def test_convolve():

    t = np.arange(0, tmax, dt)
    P = dcmri.compartment_propagator(t, MTT)
    J = dcmri.injection_gv(t, weight, conc, dose, rate, start, dispersion=dispersion)

    ref = dcmri.propagate_compartment(t, J, MTT)
    new = dcmri.convolve(t, t, J, t, P)

    plt.plot(t, ref, 'b-')
    plt.plot(t, new, 'rx')
    plt.show()

    dif = ref-new
    error = 100*np.sqrt(np.mean(dif**2))/ np.sqrt(np.mean(ref**2))
    print('Error (%)', error)

if __name__ == "__main__":
    #test_res_comp()
    #test_uconv()
    # test_chain_residue()
    test_ures_chain()
    #test_convolve()
    

