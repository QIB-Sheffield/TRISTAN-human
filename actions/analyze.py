import os
import numpy as np

import mdreg
from mdreg.models import constant
import dbdicom as db
import weasel





class MDRegDynamics(weasel.Action):

    def enable(self, app):

        if not hasattr(app, 'folder'):
            return False
        return True

    def run(self, app): 
        """
        Perform model-driven motion correction
        """
        series = app.get_selected(3)[0]
        array, dataset = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)

        mdr = mdreg.MDReg()
    #    mdr.signal_parameters = []
    #    mdr.signal_model = constant
        mdr.set_elastix(MaximumNumberOfIterations = 256)
    #    #mdr.precision = 1
        mdr.status = app.status

        for z in range(array.shape[2]):
            mdr.pinned_message = 'MDR for slice ' + str(z) + ' of ' + str(array.shape[2])
            # weasel.status.progress(z, array.shape[2], 'Fitting model..')
            mdr.pixel_spacing = dataset[z,0,0].PixelSpacing
            mdr.set_array(np.squeeze(array[:,:,z,:,0]))
            mdr.fit()   # Add status bar option like in dbdicom
            array[:,:,z,:,0] = mdr.coreg
            #mdr.fit_signal()
            #array[:,:,z,:,0] = mdr.model_fit

        fit = series.new_cousin(SeriesDescription = series.SeriesDescription + '_coreg')
        fit.set_array(array, dataset, pixels_first=True)

        #TO REPLACE BY:
        #xarray = db.array(series, sortby=['SliceLocation', 'AcquisitionTime'], pixels_first=True)
        #perform mdreg on dbarray.tonumpy()
        #fit = series.new_sibling(SeriesDescription = series.SeriesDescription + '_array')
        #fit.write_array(xarray, pixels_first=True)

        app.refresh() 






