import os
import numpy as np


import dbdicom as db
import weasel




class Region(weasel.Action):

    def enable(self, app):
        
        if app.__class__.__name__ != 'Windows':
            return False
        return app.nr_selected(3) != 0

    def run(self, app):

        for series in app.get_selected(3):

            viewer = weasel.widgets.SeriesViewerROI(series, dimensions=attributes + ['StudyDate'])
            viewer.dataWritten.connect(app.treeView.setFolder)
            app.addAsSubWindow(viewer, title=series.label())


class FourDimArrayDisplay(weasel.Action):

    def enable(self, app):
        return app.nr_selected(3) != 0

    def run(self, app):
        series = app.get_selected(3)[0]
        array = series.load_npy()
        no_array = array is None
        if no_array:
            array, _ = series.array(['SliceLocation', 'AcquisitionTime'], pixels_first=True)
            array = np.squeeze(array[...,0])
        if array.ndim != 4:
            app.dialog.information("Please select a 4D array for this viewer")
            return
        viewer = weasel.widgets.FourDimViewer(weasel.status, array)
        app.addAsSubWindow(viewer, title=series.label())
        app.status.message('Saving array for rapid access..')
        if no_array:
            series.save_npy(array=array)
        app.status.message('')





