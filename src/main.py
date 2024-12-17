from scripts import steps

def main(subject_details, study_name):

    # setup logger
    logger = steps.setup_logger(subject_details)

    # setup details dictionary
    info = steps.setup_detail_dict(subject_details, study_name, logger)

    # # extract arrays from dicom
    steps.extract_arrays(info)

    # # segment ROI pre coregistration
    # steps.segment_ROI(info, 'pre_coreg')

    # # setup mdreg
    # steps.correct_motion(info)

    # segment ROI post coregistration
    #steps.segment_ROI(info, 'post_coreg')

    # show ROIS
    #steps.show_ROIS(info, 'postcoreg')

    # Map T1
    steps.t1mapping_molli(info)

    return


if __name__ == '__main__':

    subjects = [5]
    visits = [1]
    scans = [1, 2]

    study_name = 'tristan_twocomp'

    for subject in subjects:
        for visit_number in visits:
            for scan_number in scans:

                subject_details = (subject, visit_number, scan_number)
                main(subject_details, study_name)