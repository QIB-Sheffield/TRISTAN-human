import time
import traceback
from pipelines import log, export, segment, csv, motioncorrect, plotting, mapping
from utilities import helper

## SETUP LOGGER

def setup_logger(subject_details):
    try:
        logger = log.create_logger(subject_details)
        logger.info('Logger setup successfully')
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        logger = None

    return logger

## DETAILS SETUP

def setup_detail_dict(subject_details, study_name, logger=None):
    try:
        info = helper.details_dict(subject_details, study_name, logger)
        if logger:
            logger.info('Details dictionary setup successfully')
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        if logger:
            logger.error('Error setting up details dictionary')
        info = None

    return info


## DATA EXTRACTION

def extract_arrays(info):
    start_time = time.time()
    try:
        export.arrays_from_dicom(info)
        total_time = time.time() - start_time
        if info['logger']:
            info['logger'].info('Data extraction completed in {} seconds'.format(total_time))
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        if info['logger']:
            info['logger'].error('Error extracting arrays from dicom')
        pass

    return

## SEGMENTATION

def segment_ROI(info, seg_type):
    start_time = time.time()
    try:
        segment.setup_segmentation(info, seg_type)
        total_time = time.time() - start_time
        if info['logger']:
            info['logger'].info('ROI segmentation completed in {} seconds'.format(total_time))
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        if info['logger']:
            info['logger'].error('Error segmenting ROI')
        pass

    return

## MOTION CORRECTION

def correct_motion(info):
    start_time = time.time()
    try:
        motioncorrect.setup_mdreg(info)
        total_time = time.time() - start_time
        if info['logger']:
            info['logger'].info('Completed Motion Correction via mdreg in {} seconds'.format(total_time))
    except Exception as e:
        print("An error occured:")
        traceback.print_exc()
        if info['logger']:
            info['logger'].error('Error completing motion correction')
        pass
    return
    
def t1mapping_molli(info):
    start_time = time.time()
    try:
        mapping.map_molli(info, 'pre_contrast')
        total_time = time.time() - start_time
        if info['logger']:
            info['logger'].info('Completed Motion Correction via mdreg in {} seconds'.format(total_time))
    except Exception as e:
        print("An error occured:")
        traceback.print_exc()
        if info['logger']:
            info['logger'].error('Error completing motion correction')
    
## REFORMAT TO CSV

def format_to_csv(info):
    start_time = time.time()
    try:
        csv.create_csv(info)
        total_time = time.time() - start_time
        if info['logger']:
            info['logger'].info('Completed csv file creation via pds in {} seconds'.format(total_time))
    except Exception as e:
        print("An error occured:")
        traceback.print_exc()
        if info['logger']:
            info['logger'].error('Error completing data reformatting to CSV')
        pass
    return

## SHOW ROIS

def show_ROIS(info, checkpoint):
    start_time = time.time()
    try:
        plotting.overlay_masks(info, checkpoint)
        total_time = time.time() - start_time
        if info['logger']:
            info['logger'].info('Completed showing ROIS in {} seconds'.format(total_time))
    except Exception as e:
        print("An error occured:")
        traceback.print_exc()
        if info['logger']:
            info['logger'].error('Error showing ROIS')
        pass
    return













