#!/usr/bin/env python3.6

from os import path

from sklearn import preprocessing
import pandas as pd
import numpy as np
import json
import argparse
try:
   import _pickle as pickle
except:
   import pickle

import os
import librosa
import collections
from bbn_primitives.time_series import *

from d3m_metadata.container import ndarray
from d3m_metadata.container import List
from d3m_metadata import hyperparams, metadata as metadata_module, params, container, utils
from primitive_interfaces.transformer import TransformerPrimitiveBase
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitives.sklearn_wrap import *

import sklearn.metrics

# Example for the documentation of the TA1 pipeline submission process
#
# It executes a TA1 pipeline using a ta1-pipeline-config.json file that follows this structure:
#  {
#    "train_data":"path/to/train/data/folder/",
#    "test_data":"path/to/test/data/folder/",
#    "output_folder":"path/to/output/folder/"
#  }

supportedResType = 'timeseries'
supportedTaskType = 'classification'
supportedTaskSubType = 'multiClass'


#def parse_dataset(datasetSchema):
#    filename, start, end = None, None, None
#
#    num_attribute = 0
#    for colDesc in datasetSchema['dataResources'][1]['columns']:
#        if 'attribute' in colDesc['role']:
#            filename = colDesc['colName']
#            num_attribute += 1
#        if 'boundaryIndicator' in colDesc['role'] and colDesc['colName'] == 'start':
#            start = colDesc['colName']
#        if 'boundaryIndicator' in colDesc['role'] and colDesc['colName'] == 'end':
#            end = colDesc['colName']
#
#    if num_attribute != 1:
#        raise Exception('Datasets with one column with attribute role supported (assumed to be filename).')
#
#    return AudioDataset(filename = filename, start = start, end = end)

def extract_feats(inputs, inputsBoundaries, dir_name, fext_pipeline = None,
                  resampling_rate = None):
    features = List()
    i = 0
    for idx, row in inputs.iterrows():
        if row[0] == '':
            features.append(np.array([]))
            continue
        filename = os.path.join(dir_name, row[0])
        print(filename)
        file_csvdata=pd.read_csv(filename,index_col=0)
        csvdata = List[ndarray]([file_csvdata], {
                    'schema': metadata_module.CONTAINER_SCHEMA_VERSION,
                    'structural_type': List[ndarray],
                    'dimension': {
                        'length': 1
                    }
                    })
        last_output = csvdata
        for fext_step in fext_pipeline:
            product = fext_step.produce(inputs = last_output)
            last_output = product.value

        features.append(last_output[0])
        i+=1
    return features

def pipeline(inputs, inputsBoundaries, dataset_path, dataset_schema,
             fext_pipeline = None, proc_pipeline = None,
             resampling_rate = None, train = False, train_targets = None,
             fext_cacheFN = None):
    #audio_dataset = parse_dataset(dataset_schema)
    # generate or load cached features - curve fittings
    audio_path = path.join(dataset_path, dataset_schema['dataResources'][0]['resPath'])
    if fext_cacheFN is None or not os.path.isfile(fext_cacheFN):
        segm_fittings = extract_feats(inputs, inputsBoundaries, audio_path,
                        fext_pipeline = fext_pipeline,
                        resampling_rate = resampling_rate)
        if fext_cacheFN is not None:
            with open(fext_cacheFN, 'wb') as fp:
                pickle.dump(List([ List(x) for x in segm_fittings ]), fp)

    if fext_cacheFN is not None:
        with open(fext_cacheFN, 'rb') as fp:
            segm_fittings = pickle.load(fp)

    # process features - curve fittings
    last_output = segm_fittings
    for proc_step in proc_pipeline:
        if train and not isinstance(proc_step, TransformerPrimitiveBase):
            if isinstance(proc_step, UnsupervisedLearnerPrimitiveBase):
                proc_step.set_training_data(inputs = last_output)
            else:
                proc_step.set_training_data(inputs = last_output,
                                            outputs = train_targets)
            proc_step.fit()
        product = proc_step.produce(inputs = last_output)
        last_output = product.value

    return last_output

###############################################################################
###############        MAIN             #######################################
###############################################################################

parser = argparse.ArgumentParser(description='TA1 pipeline',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument('--cachedir', type=str, default=None,
                   help="directory to cache the extracted features")
parser.add_argument('--fext_wlen', type=float, default=200,
                   help="")
parser.add_argument('--fext_rel_wshift', type=float, default=1,
                   help="")
parser.add_argument('--fext_mfcc_ceps', type=int, default=3,
                   help="")
parser.add_argument('--fext_poly_deg', type=int, default=2,
                   help="")
parser.add_argument('--n_clusters', type=int, default=32,
                   help="")
parser.add_argument('--ngram', type=int, default=1,
                   help="")
parser.add_argument('--tfidf_norm', type=str, default='l2',
                   help="")
parser.add_argument('--tfidf_use_idf', type=int, default=1,
                   help="")
parser.add_argument('--tfidf_smooth_idf', type=int, default=1,
                   help="")
parser.add_argument('--tfidf_sublinear_tf', type=int, default=1,
                   help="")
parser.add_argument('--svc_penalty', type=str, default='l2',
                   help="")
parser.add_argument('--svc_loss', type=str, default='squared_hinge',
                   help="")
parser.add_argument('--svc_C', type=float, default=1,
                   help="")

args = parser.parse_args()

# Load the json configuration file
with open("ta1-pipeline-config.json", 'r') as inputFile:
    jsonCall = json.load(inputFile)
    inputFile.close()

# Load the problem description schema
with open( path.join(jsonCall['train_data'], 'problem_TRAIN', 'problemDoc.json' ) , 'r') as inputFile:
    problemSchema = json.load(inputFile)
    inputFile.close()

# Load the json dataset description file
trainDatasetPath = path.join(jsonCall['train_data'], 'dataset_TRAIN')
with open( path.join(trainDatasetPath, 'datasetDoc.json' ) , 'r') as inputFile:
    datasetSchema = json.load(inputFile)
    inputFile.close()

taskType = problemSchema['about']['taskType']
if taskType != supportedTaskType:
    raise Exception('supported tasktype is %s, provided problem is of type %s' % (supportedTaskType, taskType))

taskSubType = problemSchema['about']['taskSubType']
if taskSubType != supportedTaskSubType:
    raise Exception('supported tasktype is %s, provided problem is of type %s' % (supportedTaskSubType, taskSubType))

# Load the json dataset description file
with open( path.join(jsonCall['train_data'], 'dataset_TRAIN', 'datasetDoc.json' ) , 'r') as inputFile:
    datasetSchema = json.load(inputFile)
    inputFile.close()

if datasetSchema['dataResources'][0]['resType'] != supportedResType:
    raise Exception('Supported resType is only %s' % (supportedResType))

# Get the target and attribute column ids from the dataset schema for training data
trainAttributesColumnIds = [ item['colIndex'] for item in datasetSchema['dataResources'][1]['columns'] if 'attribute' in item['role'] ]
boundariesColumnIds = [ item['colIndex'] for item in datasetSchema['dataResources'][1]['columns'] if 'boundaryIndicator' in item['role'] ]
trainTargetsColumnIds = [ item['colIndex'] for item in problemSchema['inputs']['data'][0]['targets'] ]

# Exit if more than one target
if len(trainAttributesColumnIds) != 1:
    raise Exception('Only one attribute column expected, %d found in the problem. Exiting.' % (len(trainAttributesColumnIds)))

if len(trainTargetsColumnIds) != 1:
    raise Exception('Only one target column expected, %d found in the problem. Exiting.' % (len(trainTargetsColumnIds)))

# Get the attribute column ids from the problem schema for test data (in this example, they are the same)
testAttributesColumnIds = trainAttributesColumnIds

# Load the tabular data file for training, replace missing values, and split it in train data and targets
trainDataResourcesPath = path.join(jsonCall['train_data'], 'dataset_TRAIN', datasetSchema['dataResources'][1]['resPath'])
#trainData = pd.read_csv( trainDataResourcesPath, header=0, usecols=trainAttributesColumnIds).fillna('0').replace('', '0')
#trainTargets = pd.read_csv( trainDataResourcesPath, header=0, usecols=trainTargetsColumnIds).fillna('0').replace('', '0')
trainData = pd.read_csv( trainDataResourcesPath, header=0, usecols=trainAttributesColumnIds).fillna('')
trainBoundaries = pd.read_csv( trainDataResourcesPath, header=0, usecols=boundariesColumnIds).fillna('')
trainTargets = pd.read_csv( trainDataResourcesPath, header=0, usecols=trainTargetsColumnIds).fillna('')

# Load the tabular data file for training, replace missing values, and split it in train data and targets
testDatasetPath = path.join(jsonCall['test_data'], 'dataset_TEST')
testDataResourcesPath = path.join(testDatasetPath, datasetSchema['dataResources'][1]['resPath'])
testData = pd.read_csv( testDataResourcesPath, header=0, usecols=testAttributesColumnIds).fillna('')
testBoundaries = pd.read_csv( testDataResourcesPath, header=0, usecols=boundariesColumnIds).fillna('')

# Get the d3mIndex of the testData
d3mIndex = pd.read_csv( testDataResourcesPath, header=0, usecols=['d3mIndex'])

# Encode the categorical data in training data
trainDataCatLabels = []
trainDataLabelEncoders = dict()
# Encode the categorical data in the test targets, uses the last target of the dataset as a target
trainTargetsCatLabel = ''
trainTargetsLabelEncoder = preprocessing.LabelEncoder()

for colDesc in datasetSchema['dataResources'][1]['columns']:
    if colDesc['colType']=='categorical' and 'attribute' in colDesc['role']:
        trainDataCatLabels.append(colDesc['colName'])
        trainDataLabelEncoders[colDesc['colName']] = preprocessing.LabelEncoder().fit(trainData[colDesc['colName']])
        trainData[colDesc['colName']] = trainDataLabelEncoders[colDesc['colName']].transform(trainData[colDesc['colName']])
    elif colDesc['colType']=='categorical' and 'suggestedTarget' in colDesc['role']:
        trainTargetsCatLabel = colDesc['colName']
        trainTargetsLabelEncoder = trainTargetsLabelEncoder.fit(trainTargets[colDesc['colName']])
        trainTargets = trainTargetsLabelEncoder.transform(trainTargets[colDesc['colName']])

# Train the model

# Build the feature extraction pipeline
resampling_rate = 1
channel_mixer = ChannelAverager(
                hyperparams = ChannelAverager.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].defaults()
            )
dither = SignalDither(
                hyperparams = SignalDither.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].defaults()
            )
framer_hyperparams = SignalFramer.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
framer_custom_hyperparams = dict()
framer_custom_hyperparams['sampling_rate'] = resampling_rate
if args.fext_wlen is not None:
  framer_custom_hyperparams['frame_length_s'] = args.fext_wlen

if args.fext_rel_wshift is not None:
  framer_custom_hyperparams['frame_shift_s'] = args.fext_rel_wshift*args.fext_wlen

framer = SignalFramer(
                hyperparams = framer_hyperparams(
                    framer_hyperparams.defaults(), **framer_custom_hyperparams
                )
            )
mfcc_hyperparams = SignalMFCC.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
mfcc_custom_hyperparams = dict()
mfcc_custom_hyperparams['sampling_rate'] = resampling_rate
if args.fext_mfcc_ceps is not None:
  mfcc_custom_hyperparams['num_ceps'] = args.fext_mfcc_ceps
mfcc = SignalMFCC(
                hyperparams = mfcc_hyperparams(
                    mfcc_hyperparams.defaults(), **mfcc_custom_hyperparams
                )
            )
segm = UniformSegmentation(
            hyperparams = UniformSegmentation.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].defaults()
        )

segm_fitter_hyperparams = SegmentCurveFitter.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
segm_fitter_custom_hyperparams = dict()
if args.fext_poly_deg is not None:
  segm_fitter_custom_hyperparams['deg'] = args.fext_poly_deg
segm_fitter = SegmentCurveFitter(
                hyperparams = segm_fitter_hyperparams(
                    segm_fitter_hyperparams.defaults(), **segm_fitter_custom_hyperparams
                )
            )
fext_pipeline = [ channel_mixer, dither, framer, mfcc, segm, segm_fitter ]



print('Feature extraction pipeline:')
for fext_step in fext_pipeline:
    print(fext_step.hyperparams)

# Build the classification pipeline
#clusterer = ClusterCurveFittingKMeans(hyperparams = ClusterCurveFittingKMeans.Hyperparams())
clusterer_hyperparams = ClusterCurveFittingKMeans.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
clusterer_custom_hyperparams = dict()
if args.n_clusters is not None:
  clusterer_custom_hyperparams['n_clusters'] = args.n_clusters
clusterer = ClusterCurveFittingKMeans(
                hyperparams = clusterer_hyperparams(
                    clusterer_hyperparams.defaults(), **clusterer_custom_hyperparams
                )
            )

fittings_framer_hyperparams = SignalFramer.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
fittings_framer_custom_hyperparams = {
    'sampling_rate': 1, 'frame_shift_s': 1, 'flatten_output': False,
}
if args.ngram is not None:
  fittings_framer_custom_hyperparams['frame_length_s'] = args.ngram
fittings_framer = SignalFramer(
                    hyperparams = fittings_framer_hyperparams(
                        fittings_framer_hyperparams.defaults(), **fittings_framer_custom_hyperparams
                    )
                  )

fittings_to_bot_hyperparams = SequenceToBagOfTokens.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
fittings_to_bot = SequenceToBagOfTokens(
                hyperparams = fittings_to_bot_hyperparams(
                    fittings_to_bot_hyperparams.defaults()
                )
            )

tfidf_hyperparams = BBNTfidfTransformer.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
tfidf_custom_hyperparams = dict()
if args.tfidf_norm is not None:
  tfidf_custom_hyperparams['norm'] = args.tfidf_norm
if args.tfidf_use_idf is not None:
  tfidf_custom_hyperparams['use_idf'] = bool(args.tfidf_use_idf)
if args.tfidf_smooth_idf is not None:
  tfidf_custom_hyperparams['smooth_idf'] = bool(args.tfidf_smooth_idf)
if args.tfidf_sublinear_tf is not None:
  tfidf_custom_hyperparams['sublinear_tf'] = bool(args.tfidf_sublinear_tf)
tfidf = BBNTfidfTransformer(
                    hyperparams = tfidf_hyperparams(
                        tfidf_hyperparams.defaults(), **tfidf_custom_hyperparams
                    )
                  )

seq_modeler_hyperparams = SKLinearSVC.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
seq_modeler_custom_hyperparams = { 'dual': True }
if args.svc_penalty is not None:
  seq_modeler_custom_hyperparams['penalty'] = args.svc_penalty
if args.svc_loss is not None:
  seq_modeler_custom_hyperparams['loss'] = args.svc_loss
if args.svc_C is not None:
  seq_modeler_custom_hyperparams['C'] = args.svc_C
seq_modeler = SKLinearSVC(
                hyperparams = seq_modeler_hyperparams(
                    seq_modeler_hyperparams.defaults(), **seq_modeler_custom_hyperparams
                )
            )
proc_pipeline = [ clusterer, fittings_framer, fittings_to_bot, tfidf, seq_modeler ]

print('Classification pipeline:')
for proc_step in proc_pipeline:
    print(proc_step.hyperparams)

#trainPredict = pipeline(trainData, clusterer, seq_modeler, train = True, train_targets = trainTargets)
#trainN = 3
trainN = len(trainData)
trainData = trainData[:trainN]
trainTargets = trainTargets[:trainN]
trainCacheFN = None #if args.cachedir is None else os.path.join(args.cachedir, 'fext_train.pkl')
trainPredict = pipeline(trainData, trainBoundaries, trainDatasetPath, datasetSchema,
                        fext_pipeline = fext_pipeline, proc_pipeline = proc_pipeline,
                        train = True, train_targets = trainTargets, resampling_rate = resampling_rate,
                        fext_cacheFN = trainCacheFN)

acc = np.mean(trainTargets == trainPredict)
print('Training accuracy: %f\n' % (acc))
confmat = sklearn.metrics.confusion_matrix(trainTargets, trainPredict)
print('Training confusion matrix: \n%s\n\n' % (confmat))

# Encode the testData using the previous label encoders
for colLabel in trainDataCatLabels:
    testData[colLabel] = trainDataLabelEncoders[colLabel].transform(testData[colLabel])

# Predicts targets from the test data
#testN = 3
testN = len(testData)
testData = testData[:testN]
testCacheFN = None #if args.cachedir is None else os.path.join(args.cachedir, 'fext_test.pkl')
predictedTargets = pipeline(testData, testBoundaries, testDatasetPath, datasetSchema,
                            fext_pipeline = fext_pipeline, proc_pipeline = proc_pipeline,
                            train = False, resampling_rate = resampling_rate,
                            fext_cacheFN = testCacheFN)

# Reverse the label encoding for predicted targets
predictedTargets = trainTargetsLabelEncoder.inverse_transform(predictedTargets)

# Append the d3mindex column to the predicted targets
predictIndex = d3mIndex['d3mIndex'][:testN]
predictedTargets = pd.DataFrame({'d3mIndex':predictIndex, trainTargetsCatLabel:predictedTargets})
#predictedTargets = pd.DataFrame({'d3mIndex':d3mIndex['d3mIndex'], trainTargetsCatLabel:predictedTargets})

# Get the file path of the expected outputs
outputFilePath = path.join(jsonCall['output_folder'], problemSchema['expectedOutputs']['predictionsFile'])

# Outputs the predicted targets in the location specified in the JSON configuration file
with open(outputFilePath, 'w') as outputFile:
    output = predictedTargets.to_csv(outputFile, index=False, columns=['d3mIndex', trainTargetsCatLabel])


