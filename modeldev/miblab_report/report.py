import os
import shutil
import pylatex as pl
from pylatex.utils import NoEscape

#path = os.path.abspath("")

def force_move(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    os.rename(src, dst)

def force_copy(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy(src, dst)

def force_move_dir(src, dst):
    force_copy_dir(src, dst)
    shutil.rmtree(src)

def force_copy_dir(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
        
def setup(dst, results):
    src = os.path.dirname(__file__)
    outputpath = os.path.join(results, 'report')
    force_copy(os.path.join(src, 'cover.jpg'), os.path.join(dst, 'cover.jpg'))
    force_copy(os.path.join(src, 'epflreport.cls'), os.path.join(dst, 'epflreport.cls'))
    force_copy_dir(os.path.join(src, 'layout'), os.path.join(outputpath, 'layout'))


def makecover(doc, title='Title', subtitle='Interim analysis', subject='Subject'):
    # Cover page
    doc.append(NoEscape('\\frontmatter'))
    doc.append(pl.Command('title', title))
    doc.append(pl.Command('subtitle', subtitle))
    doc.append(pl.Command('author', 'TRISTAN work package 2'))
    doc.append(pl.Command('subject', subject))
    doc.append(pl.Command('affiliation', 'https://www.imi-tristan.eu/liver'))
    doc.append(pl.Command('coverimage', 'cover.jpg')) 
    doc.append(pl.Command('definecolor', arguments = ['title','HTML','FF0000'])) # Color for cover title
    doc.append(NoEscape('\\makecover'))


def titlepage(doc, results):
    # Title page
    doc.append(pl.Command('begin', 'titlepage'))
    doc.append(pl.Command('begin', 'center'))
    # Title
    doc.append(NoEscape('\\makeatletter'))
    doc.append(NoEscape('\\largetitlestyle\\fontsize{45}{45}\\selectfont\\@title'))
    doc.append(NoEscape('\\makeatother'))
    # Subtitle
    doc.append(NoEscape('\\linebreak'))
    doc.append(NoEscape('\\makeatletter'))
    doc.append(pl.Command('ifdefvoid', arguments=[NoEscape('\\@subtitle'),'',NoEscape('\\bigskip\\titlestyle\\fontsize{20}{20}\\selectfont\\@subtitle')]))
    # Author
    doc.append(NoEscape('\\makeatother'))
    doc.append(NoEscape('\\linebreak'))
    doc.append(NoEscape('\\bigskip'))
    doc.append(NoEscape('\\bigskip'))
    doc.append('by')
    doc.append(NoEscape('\\linebreak'))
    doc.append(NoEscape('\\bigskip'))
    doc.append(NoEscape('\\bigskip'))
    doc.append(NoEscape('\\makeatletter'))
    doc.append(NoEscape('\\largetitlestyle\\fontsize{25}{25}\\selectfont\\@author'))
    doc.append(NoEscape('\\makeatother'))
    doc.append(NoEscape('\\vfill'))
    # Table with information
    doc.append(NoEscape('\\large'))
    with doc.create(pl.Tabular('ll')) as tab:
        tab.add_hline()
        tab.add_row(['Report compiled by: ', 'Steven Sourbron'])
        tab.add_row(['Institute: ', 'University of Sheffield'])
        tab.add_row(['Department: ', 'Section of Medical Imaging and Technologies'])
        tab.add_row(['Email: ', 's.sourbron@sheffield.ac.uk'])
        tab.add_row(['Date: ', NoEscape('\\today')])
        tab.add_hline()
    # TRISTAN logo
    with doc.create(pl.Figure(position='b!')) as pic:
        pic.append(pl.Command('centering'))
        im = os.path.join(results, 'report', 'layout', 'tristan-logo.jpg')
        pic.add_image(im, width='2in')

    doc.append(pl.Command('end', 'center'))
    doc.append(pl.Command('end', 'titlepage'))


def create(doc, path, filename, results):

    # Create report
    outputpath = os.path.join(results, 'report')
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    doc.generate_pdf(filename, clean=False, clean_tex=False, compiler='pdfLaTeX', compiler_args=['-output-directory', outputpath])

    # Move all files to output folder and clean up
    force_move(os.path.join(path, 'cover.jpg'), os.path.join(outputpath, 'cover.jpg'))
    force_move(os.path.join(path, 'epflreport.cls'), os.path.join(outputpath, 'epflreport.cls'))
    force_move(os.path.join(path, filename+'.tex'), os.path.join(outputpath, filename+'.tex'))


