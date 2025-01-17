from scripts import steps

def main(subject_details, study_name):

    scan = subject_details[2]

    # setup logger
    logger = steps.setup_logger(subject_details)

    # setup details dictionary
    info = steps.setup_detail_dict(subject_details, study_name, logger)

    # extract arrays from dicom
    steps.extract_arrays(info)

    # segment ROI pre coregistration
    steps.segment_ROI(info, 'pre_coreg')

    # show ROIS
    steps.show_ROIS(info, 'pre_coreg')

    # setup mdreg
    steps.correct_motion(info, [20,90])

    # segment ROI post coregistration
    steps.segment_ROI(info, 'post_coreg')

    # show ROIS
    steps.show_ROIS(info, 'post_coreg')

    # Map T1
    steps.t1mapping_molli(info, 'pre_coreg')

    if scan == 1:
        steps.t1mapping_molli(info, 'post_coreg')

    if scan == 2:
        # export CSV
        # involves both scans 1 and 2
        steps.format_to_csv(info)
        steps.run_dcmri(info)


    return


if __name__ == '__main__':

    subjects = [5]
    visits = [1]
    scans = [1,2]

    study_name = 'tristan_twocomp'

    for subject in subjects:
        for visit_number in visits:
            for scan_number in scans:

                subject_details = (subject, visit_number, scan_number)
                main(subject_details, study_name)