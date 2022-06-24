import os
import numpy as np


import dbdicom as db
import weasel



# These attributes are used frequently in iBEAt.
# They are loaded up front to avoid rereading 
# the folder every time they are needed.
attributes = [
    'SliceLocation', 'AcquisitionTime', 
    'FlipAngle', 'EchoTime', 'InversionTime',                           
]


class Open(weasel.Action):

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
        app.folder.set_attributes(attributes, scan=False)
        app.open(path)
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
        app.folder.set_attributes(attributes, scan=False)
        for i, path in enumerate(subfolders):
            msg = 'Reading folder ' + str(i+1) + ' of ' + str(len(subfolders))
            app.folder.open(path, message=msg)
            app.folder.save()
        app.status.cursorToNormal()
        app.display(app.folder)

