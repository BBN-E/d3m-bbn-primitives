#!/usr/bin/env python3.6

from os import path
import json
import pandas as pd
import numpy as np
import sklearn.metrics
from pandas_ml import ConfusionMatrix

# Load the json configuration file
with open("ta1-pipeline-config.json", 'r') as inputFile:
    jsonCall = json.load(inputFile)
    inputFile.close()

# Load the problem description schema
with open( path.join(jsonCall['train_data'], 'problem_TRAIN', 'problemDoc.json' ) , 'r') as inputFile:
    problemSchema = json.load(inputFile)
    inputFile.close()

#Get the target class or label name refrence used
#print(problemSchema['inputs']['data'][0]['targets'][0]['colName'])
score_label_name=problemSchema['inputs']['data'][0]['targets'][0]['colName']

# Get the file path of the expected outputs
predictionsFN = path.join(jsonCall['output_folder'], problemSchema['expectedOutputs']['predictionsFile'])
evalFN = predictionsFN + '.eval'


# Outputs the predicted targets in the location specified in the JSON configuration file
outputData = pd.read_csv(predictionsFN, header=0)

targetsFilePath = path.join(jsonCall['test_data'], '..', 'SCORE', 'targets.csv')
targetsData = pd.read_csv(targetsFilePath, header=0).fillna('')

if len(outputData) != len(targetsData):
    raise Exception('Number of outputs does not match number of targets, %d vs %d' % (outputData, targetsData))

outputC = [ outputData[score_label_name][i] for i in range(len(targetsData)) if targetsData[score_label_name][i] != '' ]
targetsC = [ x for x in targetsData[score_label_name] if x != '' ]
lbls = sorted(set(outputC+targetsC))

#acc = np.mean(targetsC == outputC)
acc = len([ i for i in range(len(outputC)) if outputC[i] == targetsC[i]]) / len(outputC)
confmat = sklearn.metrics.confusion_matrix(targetsC, outputC, lbls)
confmat_norm = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.nan)
pd_confmat = ConfusionMatrix(targetsC, outputC)
with open(evalFN, 'w') as evalF:
    evalF.write('Accuracy: %f\n' % (acc))
    #evalF.write('Classes: \n%s\n\n' % (" ".join(lbls)))
    evalF.write('Confusion matrix normalized:\n%s\n\n' % (confmat_norm))
    evalF.write('Confusion matrix: \n%s\n\n' % (pd_confmat))
    evalF.write('%s' % (pd_confmat._str_stats()))

with open(evalFN, 'r') as evalF:
    for line in evalF:
        print(line)

