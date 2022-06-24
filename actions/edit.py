import os
import numpy as np


import dbdicom as db
import weasel


class MergeDynamics(weasel.Action):

    def enable(self, app):

        if not hasattr(app, 'folder'):
            return False
        return True

    def run(self, app): 
        """
        Merge the dynamics with FA 15 of all studies in the database.

        TODO: include merge of VFA data for T1-mapping
        """

        # Find all series with the correct SeriesDescriptions
        studies = app.folder.studies()
        desc = ['fl3d_fast_fb_fa15_W', 'fl3d_fast_fb_fa15_dynamic_W']
        dyn_series = db.find_series(studies, SeriesDescription=desc)
        sorted_series = []
        for i, series in enumerate(dyn_series):
            app.status.progress(i+1, len(dyn_series), 'Sorting dynamics..')
            sorted = series.dataset(['SliceLocation', 'AcquisitionTime'], status=False)
            sorted_series.append(sorted)
        
        # Create a new study & series to merge them in
        merged = dyn_series[0].new_cousin(
            StudyDescripton = 'list of merged series',
            SeriesDescription = 'fl3d_fast_fb_fa15_W_merged')

        # Merge with overwriting slice locations
        z_indices = range(sorted_series[0].shape[0])
        cnt = 1
        cntmax = len(sorted_series) * len(z_indices)
        for i, series in enumerate(sorted_series):
            for z in z_indices:
                app.status.progress(cnt, cntmax, 'Merging dynamics..')
                dynamics = np.squeeze(series[z,:,0]).tolist()
                db.set_value(dynamics, 
                    SliceLocation = z, 
                    PatientID = merged.UID[0],
                    StudyInstanceUID = merged.UID[1],
                    SeriesInstanceUID = merged.UID[2])
                cnt += 1
        app.refresh()


class RenameSeries(weasel.Action):

    def enable(self, app):
        if not hasattr(app, 'folder'):
            return False
        return app.nr_selected(3) != 0

    def run(self, app): 
        series_list = app.get_selected(3)
        for s in series_list:
            desc = s.SeriesDescription
            app.status.message('Renaming series ' + desc)
            db.set_value(s.instances(), SeriesDescription=desc+'_new_name')
        app.status.hide()
        app.refresh()

