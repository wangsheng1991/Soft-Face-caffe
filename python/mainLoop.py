
import numpy as np
import os
import cv2
import sys
import shutil
from pickle import load, dump
from zipfile import ZipFile
from urllib import urlretrieve

#Import helper functions
from DataRow import DataRow, ErrorAcum, Predictor, getGitRepFolder, createDataRowsFromCSV, getValidWithBBox, cropFace

###########################    PATHS TO SET   ####################

# Either define CAFFE_ROOT in your enviroment variables or set it here
CAFFE_ROOT = os.environ.get('CAFFE_ROOT','~/caffe/distribute')
sys.path.append(CAFFE_ROOT+'/python')  # Add caffe python path so we can import it
import caffe

# Make sure dlib python path exists on PYTHONPATH else "pip install dlib" if needed.
import dlib
detector=dlib.get_frontal_face_detector() # Load dlib's face detector

ROOT = getGitRepFolder()  # ROOT is the git root folder .
sys.path.append(os.path.join(ROOT, 'python'))  # Assume git root directory
DATA_PATH = os.path.join(ROOT, 'data')
AFLW_CSV_TEST  = os.path.join(ROOT, 'data', 'testImageList.txt')
AFLW_CSV_TRAIN = os.path.join(ROOT, 'data', 'trainImageList.txt')

AFW_DATA_PATH = os.path.join(ROOT, 'data', 'testimages')
PATH_TO_WEIGHTS  = os.path.join(ROOT, 'model', 'vanillaCNN.caffemodel')
PATH_TO_DEPLOY_TXT = os.path.join(ROOT, 'model', 'vanilla_deploy.prototxt')


###########################    STEPS TO RUN       ####################
#STEPS =['trainSetHD5', 'calcTrainSetMean'] # AFLW_STEPS+AFW_STEPS # Run AFLW and AFW steps
#STEPS = ['downloadAFLW', 'makeAFLWTestSet', 'makeAFLWTrainSet']
STEPS = ['downloadAFLW', 'makeAFLWTrainSet', 'makeAFLWTestSet']


##########################################    SCRIPT STEPS       ##################################################

# DOWNLOAD AFLW DATA
if 'downloadAFLW' in STEPS:
    theurl='http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip'
    filename = ROOT+'/data/AFLW.zip'
    if os.path.isfile(filename):
        print "AFLW.zip training data already downloaded"
    else:
        print "Downloading "+theurl + " ....."
        name, hdrs = urlretrieve(theurl, filename)
        print "Finished downloading AFLW....."

    print 'Extracting zip data...'
    folderPATH = ROOT+'/data'
    with ZipFile(filename) as theOpenedFile:
        theOpenedFile.extractall(folderPATH)
        theOpenedFile.close()
    print "Done extracting AFW zip folder"



# MAKE AFLW TESTSET(VALIDATION PROCESS IS NEEDED)
if 'makeAFLWTestSet' in STEPS :
    print "Creating AFLW test set....."
    dataRowsTest_CSV  = createDataRowsFromCSV(AFLW_CSV_TEST , DataRow.DataRowFromNameBoxInterlaved, DATA_PATH)
    print "Finished reading %d images from testset. Validation session is ongoing....." % len(dataRowsTest_CSV)

    dataRowsTestValid,R = getValidWithBBox(dataRowsTest_CSV)
    print "Valid Images:", len(dataRowsTestValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch
    print "Valid images are copied to data/testAFLW..... "
 
    # dataRowsTestValid - valid testset(face exist), landmark points(left eye, right eye, nose, left mouth, right mouth)	
    destpath = DATA_PATH + '/testAFLW'
    destpathlist = destpath + '/testImageList.txt'   
    
    cropFace(dataRowsTestValid, destpath, destpathlist)	
    print "Done .."    



# MAKE AFLW TRAINSET(VALIDATION PROCESS IS ALSO NEEDED)
if 'makeAFLWTrainSet' in STEPS :
    print "Creating AFLW train set....."
    dataRowsTest_CSV  = createDataRowsFromCSV(AFLW_CSV_TRAIN , DataRow.DataRowFromNameBoxInterlaved, DATA_PATH)
    print "Finished reading %d images from trainset. Validation session is ongoing....." % len(dataRowsTest_CSV)

    dataRowsTestValid,R = getValidWithBBox(dataRowsTest_CSV)
    print "Valid Images:", len(dataRowsTestValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch
    print "Valid images are copied to data/trainAFLW..... "
    

    destpath = DATA_PATH + '/trainAFLW'
    destpathlist = destpath + '/trainImageList.txt'   
    
    cropFace(dataRowsTestValid, destpath, destpathlist)	
    print "Done .."    

''' 
DEBUG = True
if 'testErrorMini' in STEPS:
    with open('testSetMini.pickle','r') as f:
        dataRowsTrainValid = load(f)
        
    testErrorMini=ErrorAcum()
    predictor = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT, weightsPath=PATH_TO_WEIGHTS)
    for i, dataRow in enumerate(dataRowsTrainValid):
        dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox).copyMirrored()
        image, lm_0_5 = predictor.preprocess(dataRow40.image, dataRow40.landmarks())
        prediction = predictor.predict(image)
        testErrorMini.add(lm_0_5, prediction)
        dataRow40.prediction = (prediction+0.5)*40.  # Scale -0.5..+0.5 to 0..40
        
        if DEBUG:
            dataRow.prediction = dataRow40.inverseScaleAndOffset(dataRow40.prediction) # Scale up to the original image scale
            dataRow.show()
            if i>40:
                print "Debug breaked after %d rows:" % i
            

    print "Test Error mini:", testErrorMini
'''
   

# Run the same caffe test set using python
DEBUG = False  # Set this to true if you wish to plot the images
if 'testError' in STEPS:
    with open('testSet.pickle','r') as f:
        dataRowsTrainValid = load(f)
        
    testError=ErrorAcum()
    predictor = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT, weightsPath=PATH_TO_WEIGHTS)
    for i, dataRow in enumerate(dataRowsTrainValid):
        dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
        image, lm40 = predictor.preprocess(dataRow40.image, dataRow40.landmarks())
        prediction = predictor.predict(image)
        testError.add(lm40, prediction)
        dataRow40.prediction = (prediction+0.5)*40.
        
        if DEBUG and i%40 ==0:
            dataRow.prediction = dataRow40.inverseScaleAndOffset(dataRow40.prediction)
            dataRow.show()
            break
        
            
    print "Test Error:", testError

  
    
# AFW test - Make the pickle data set
if 'createAFW_TestSet' in STEPS:
    print "Parsing AFW anno-v7.mat ....."
    from scipy.io import loadmat
    annotaions = loadmat(AFW_MAT_PATH)['anno']
    dataRowsAFW = []
        
    for anno in annotaions:
        dataRow = DataRow.DataRowFromAFW(anno, AFW_DATA_PATH)
        if dataRow is not None:
            dataRowsAFW.append(dataRow)
    print "Finished parsing anno-v7.mat with total rows:", len(dataRowsAFW)
    annotaions = None  # remove from memory
    
    dataRowsAFW_Valid, R=getValidWithBBox(dataRowsAFW)
    print "Original AFW:",len(dataRowsAFW), "Valid Rows:", len(dataRowsAFW_Valid), " No faces at all", R.noFacesAtAll, " illegal landmarks:", R.outsideLandmarks, " Could not match:", R.couldNotMatch
    dataRowsAFW = None  # remove from Memory
    with open('afwTestSet.pickle','w') as f:
        dump(dataRowsAFW_Valid, f)
        print "Data saved to afwTestSet.pickle"
    
    
DEBUG = False
if 'testAFW_TestSet' in STEPS:
    with open('afwTestSet.pickle','r') as f:
        dataRowsAFW_Valid = load(f)

    testErrorAFW=ErrorAcum()
    predictor = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT, weightsPath=PATH_TO_WEIGHTS)
    for i, dataRow in enumerate(dataRowsAFW_Valid):
        dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
        image, lm_0_5 = predictor.preprocess(dataRow40.image, dataRow40.landmarks())
        prediction = predictor.predict(image)
        testErrorAFW.add(lm_0_5, prediction)
        dataRow40.prediction = (prediction+0.5)*40.  # Scale -0.5..+0.5 to 0..40
        
        if DEBUG:
            dataRow.prediction = dataRow40.inverseScaleAndOffset(dataRow40.prediction) # Scale up to the original image scale
            dataRow.show()
            

    print "Test Error AFW:", testErrorAFW

    

# Create the MTFL benchmark
if 'createAFLW_TestSet' in STEPS:  
    MTFL_LINK = 'http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip'
    MTFL_ZIP = ROOT+"/MTFL.zip"
    if os.path.isfile(MTFL_ZIP):
        print "MTFL.zip already downloaded"
    else:
        print "Downloading:"+MTFL_ZIP+" from url:"+MTFL_LINK+"....."
        urlretrieve(MTFL_LINK, MTFL_ZIP)
        print "Finished download. Extracting file....."
        with ZipFile(MTFL_ZIP) as f:
            f.extractall(ROOT+'/data')
            print "Done extracting MTFL"
            f.close()
            
    AFLW_PATH = os.path.join(ROOT,'data')
    CSV_MTFL = os.path.join(AFLW_PATH,'testing.txt')
    dataRowsMTFL_CSV  = createDataRowsFromCSV(CSV_MTFL , DataRow.DataRowFromMTFL, AFLW_PATH)
    print "Finished reading %d rows from test" % len(dataRowsMTFL_CSV)
    dataRowsMTFLValid,R = getValidWithBBox(dataRowsMTFL_CSV)
    print "Original test:",len(dataRowsMTFL_CSV), "Valid Rows:", len(dataRowsMTFLValid), " No faces at all", R.noFacesAtAll, " Illegal landmarks:", R.outsideLandmarks, " Could not match:", R.couldNotMatch
    with open('testSetMTFL.pickle','w') as f:
        dump(dataRowsMTFLValid,f)
    print "Finished dumping to testSetMTFL.pickle"        


# Run AFLW benchmark
DEBUG = False
if 'testAFLW_TestSet' in STEPS:
    print "Running AFLW benchmark........."
    with open('testSetMTFL.pickle','r') as f:
        dataRowsAFW_Valid = load(f)
    print "%d rows in AFLW benchmark ....." % len(dataRowsAFW_Valid)
    testErrorAFLW=ErrorAcum()
    predictor = Predictor(protoTXTPath=PATH_TO_DEPLOY_TXT, weightsPath=PATH_TO_WEIGHTS)
    for i, dataRow in enumerate(dataRowsAFW_Valid):
        dataRow40 = dataRow.copyCroppedByBBox(dataRow.fbbox)
        image, lm_0_5 = predictor.preprocess(dataRow40.image, dataRow40.landmarks())
        prediction = predictor.predict(image)
        testErrorAFLW.add(lm_0_5, prediction)
        dataRow40.prediction = (prediction+0.5)*40.  # Scale -0.5..+0.5 to 0..40
        
        if DEBUG and i%40 == 0:
            dataRow.prediction = dataRow40.inverseScaleAndOffset(dataRow40.prediction) # Scale up to the original image scale
            dataRow.show()


    print "Test Error AFLW:", testErrorAFLW
