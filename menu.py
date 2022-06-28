
import weasel

import actions.file as file
import actions.edit as edit
import actions.view as view
import actions.analyze as analyze

# These attributes are used frequently in TRISTAN.
# They are loaded up front to avoid rereading 
# the folder every time they are needed.
attr = [
    'SliceLocation', 'AcquisitionTime', 
    'FlipAngle', 'EchoTime', 'InversionTime',                           
]


def dev(parent): 

    menu = parent.menu('File')
    menu.action(file.Open, text='Open (TRISTAN)', options=attr)
    menu.action(file.OpenSubFolders, text='Open subfolders (TRISTAN)', options=attr)
    menu.action(file.ExportSeries, text='Export series (TRISTAN)', options=attr)
    menu.separator()
    weasel.actions.folder.menu(menu)

    menu = parent.menu('Edit')
    menu.action(edit.ExtractSeries, text='Extract subseries')
    menu.action(edit.RenameSeries, text='Rename series..')
    menu.separator()
    weasel.actions.edit.menu(menu)
    
    menu = parent.menu('View')
    menu.action(view.Region, text='Draw ROI (TRISTAN)', options=attr+['StudyDate'])
    menu.action(view.FourDimArrayDisplay, text='View 4D Array (TRISTAN)')
    menu.separator()
    weasel.actions.view.menu(menu)

    menu = parent.menu('Dev (TRISTAN)')
    menu.action(edit.MergeDynamics, text='Merge all FA 15 dynamics')
    menu.action(analyze.MDRegDynamics, text='Motion-correct dynamics')
    
    menu = parent.menu('About')
    weasel.actions.about.menu(menu)

