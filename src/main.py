
import os
import pipelines


DATAPATH = "C:\\Users\\steve\\Dropbox\\Data\\TRISTAN_2C"
RESULTSPATH = "C:\\Users\\steve\\Dropbox\\Results\\TRISTAN_2C"


if __name__ == '__main__':

    pipelines.onescandev(os.path.join(DATAPATH, 'MEDCIC_02', 'Visit1', 'Scan1'), RESULTSPATH)
    #pipelines.onescan(DATAPATH, RESULTSPATH, 2,1,1)