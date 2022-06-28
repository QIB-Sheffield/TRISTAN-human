import os
import weasel


class Open(weasel.Action):
    """Generalizes the current open action in weasel
    to allow setting of general attributes"""

    def enable(self, app):

        if not hasattr(app, 'folder'):
            return False
        return True

    def run(self, app):
        """
        Open a DICOM folder and update display.
        """

        app.status.message("Opening DICOM folder..")
        path = app.dialog.directory("Select a DICOM folder")
        if path == '':
            app.status.message('') 
            return
        app.status.cursorToHourglass()
        app.close()
        app.open(path, attributes=self.options)
        app.status.cursorToNormal()


class OpenSubFolders(weasel.Action):

    def enable(self, app):

        if not hasattr(app, 'folder'):
            return False
        return True

    def run(self, app):
        """
        Open a DICOM folder and update display.
        """
        app.status.message("Opening DICOM folder..")
        path = app.dialog.directory("Select the top folder..")
        if path == '':
            app.status.message('') 
            return
        subfolders = next(os.walk(path))[1]
        subfolders = [os.path.join(path, f) for f in subfolders]
        app.close()
        app.status.cursorToHourglass()
        for i, path in enumerate(subfolders):
            msg = 'Reading folder ' + str(i+1) + ' of ' + str(len(subfolders))
            app.open(path, attributes=self.options, message=msg)
            app.folder.save()
        app.status.cursorToNormal()
        app.display(app.folder)

class ExportSeries(weasel.Action):
    """Export selected series"""

    def enable(self, app):

        if not hasattr(app, 'folder'):
            return False
        return True

    def run(self, app):

        series = app.get_selected(3)
        if series == []:
            app.dialog.information("Please select at least one series")
            return
        path = app.dialog.directory("Where do you want to export the data?")
        for i, s in enumerate(series):
            app.status.progress(i, len(series), 'Exporting data..')
            s.export(path)
        app.status.hide()

