import os
import numpy as np


import weasel
#import actions.file as file
import actions


def menu(parent): 

    menu = parent.menu('File')
    menu.action(actions.file.Open, shortcut='Ctrl+O')
    menu.action(weasel.actions.folder.Read)
    menu.action(weasel.actions.folder.Save, shortcut='Ctrl+S')
    menu.action(weasel.actions.folder.Restore, shortcut='Ctrl+R')
    menu.action(weasel.actions.folder.Close, shortcut='Ctrl+C')

    weasel.actions.edit.menu(parent.menu('Edit'))

    menu = parent.menu('View')
    menu.action(weasel.actions.view.Series)
    menu.action(actions.view.Region, text='Draw ROI')
    menu.action(actions.view.FourDimArrayDisplay, text='4D Array')
    menu.separator()
    menu.action(weasel.actions.view.CloseWindows, text='Close windows')
    menu.action(weasel.actions.view.TileWindows, text='Tile windows')

    menu = parent.menu('dev')
    #menu.action(actions.RenameSeries, text='Rename series..')
    menu.action(actions.file.OpenSubFolders, text='Read subfolders')
    menu.separator()
    menu.action(actions.edit.MergeDynamics, text='Merge all FA 15 dynamics')
    menu.action(actions.analyse.MDRegDynamics, text='Motion-correct dynamics')
    
    weasel.actions.about.menu(parent.menu('About'))
