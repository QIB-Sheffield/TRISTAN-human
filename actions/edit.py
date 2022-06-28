import os
import numpy as np

import dbdicom as db
import weasel

class ExtractSeries(weasel.Action):

    def run(self, app):

        series = app.get_selected(3)[0]
        ds = series.dataset(['SliceLocation', 'AcquisitionTime'], status=True)
        nz, nt = ds.shape[0], ds.shape[1]
        x0, x1, t0, t1 = 0, nz, 0, nt
        invalid = True
        while invalid:
            cancel, f = app.dialog.input(
                {"type":"integer", "label":"Slice location from index..", "value":x0, "minimum": 0, "maximum": nz},
                {"type":"integer", "label":"Slice location to index..", "value":x1, "minimum": 0, "maximum": nz},
                {"type":"integer", "label":"Acquisition time from index..", "value":t0, "minimum": 0, "maximum": nt},
                {"type":"integer", "label":"Acquisition time to index..", "value":t1, "minimum": 0, "maximum": nt},
                title='Select parameter ranges')
            if cancel: return
            x0, x1, t0, t1 = f[0]['value'], f[1]['value'], f[2]['value'], f[3]['value']
            invalid = (x0 >= x1) or (t0 >= t1)
            if invalid:
                app.dialog.information("Invalid selection - first index must be lower than second")
        name = ' [' + str(x0) + ':' + str(x1) 
        name += ', ' + str(t0) + ':' + str(t1) + ']'
        new = series.new_cousin(
            StudyDescription = 'extracted',
            SeriesDescription = series.SeriesDescription + name, 
            )
        db.copy(ds[x0:x1,t0:t1,0], new, status=app.status)
        app.refresh()


class RenameSeries(weasel.Action):

    def enable(self, app):
        if not hasattr(app, 'folder'):
            return False
        return app.nr_selected(3) != 0

    def run(self, app): 
        series_list = app.get_selected(3)
        for s in series_list:
            cancel, f = app.dialog.input(
                {"type":"string", "label":"New series name:", "value": s.SeriesDescription},
                title = 'Enter new series name')
            if cancel:
                return
            db.set_value(s.instances(), SeriesDescription=f[0]['value'])
        app.status.hide()
        app.refresh()


class MergeDynamics(weasel.Action):

    def enable(self, app):

        if not hasattr(app, 'folder'):
            return False
        return True

    def run(self, app): 
        """
        Merge the dynamics with FA 15 of all studies in the database.
        Since slices of two separate scans have different slice locations
        this overwrites the slice location with its index.

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
            StudyDescription = 'list of merged series',
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



