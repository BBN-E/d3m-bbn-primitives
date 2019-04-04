import sys
import json

jsonFN = sys.argv[1]
jsonPath = sys.argv[2]

with open(jsonFN, 'r') as inputFile:
    jsonFile = json.load(inputFile)
    inputFile.close()

print(jsonFile[jsonPath])

