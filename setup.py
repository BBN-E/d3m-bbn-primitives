import os
import re
import sys
from setuptools import setup, find_packages
#from urllib.parse import urlparse

PACKAGE_NAME = 'bbn_primitives'
MINIMUM_PYTHON_VERSION = 3, 6

REQUIREMENTS_MAP = {}
def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert False, "'{0}' not found in '{1}'".format(key, module_path)


def package_from_requirement(requirement):
    """Convert pip requirement string to a package name."""
    return re.sub(r'-',
                  r'_',
                  re.sub(r'\[.*?\]|.*/([^@/]*?)(\.git)?.*',
                         r'\1',
                         requirement))


def read_requirements():
    """Read the requirements."""
    with open('requirements.txt') as requirements:
        return [package_from_requirement(requirement.strip())
                for requirement in requirements]


check_python_version()

setup(
    name=PACKAGE_NAME,
    version=read_package_variable('__version__'),
    description='BBN D3M primitives',
    author='BBN',
    #packages=['d3m-bbn-primitives', 'd3m-bbn-primitives.bbn_primitives.time_series'],
    #packages=find_packages('time_series'),
    packages=find_packages(exclude=['examples', 'notebooks', 'pipelines', 'annotations', 'tools']),
    install_requires=read_requirements(),
    #include_package_data=True,
    url='https://gitlab.datadrivendiscovery.org/BBN/d3m-bbn-primitives/tree/d3m.api2018.6.5.a',
    #download_url='https://gitlab.datadrivendiscovery.org/BBN/d3m-bbn-primitives/repository/archive.tar.gz?ref=fall2017.a',
    #classifiers=[
    #    "Programming Language :: Python :: 3"
    #],
    entry_points = {
        'd3m.primitives': [
            'data_preprocessing.AudioReader.BBN = bbn_primitives.time_series.audio_reader:AudioReader',
            'data_preprocessing.TargetsReader.BBN = bbn_primitives.time_series.targets_reader:TargetsReader',
             #'bbn.time_series.CSVReader = bbn_primitives.time_series.csv_reader:CSVReader',
            'data_preprocessing.ChannelAverager.BBN = bbn_primitives.time_series.channel_averaging:ChannelAverager',
            'time_series_segmentation.SignalFramer.BBN = bbn_primitives.time_series.signal_framing:SignalFramer',
            'data_preprocessing.SignalDither.BBN = bbn_primitives.time_series.signal_dither:SignalDither',
            'feature_extraction.SignalMFCC.BBN = bbn_primitives.time_series.signal_mfcc:SignalMFCC',
            'time_series_segmentation.UniformSegmentation.BBN = bbn_primitives.time_series.segmentation_uniform:UniformSegmentation',
            'data_transformation.SegmentCurveFitter.BBN = bbn_primitives.time_series.segment_curve_fitter:SegmentCurveFitter',
            'clustering.ClusterCurveFittingKMeans.BBN  = bbn_primitives.time_series.cluster_curve_fitting_kmeans:ClusterCurveFittingKMeans',
            'data_transformation.SequenceToBagOfTokens.BBN = bbn_primitives.time_series.sequence_to_bot:SequenceToBagOfTokens',
            'feature_extraction.BBNTfidfTransformer.BBN = bbn_primitives.time_series.tfidf_transformer:BBNTfidfTransformer',
            'data_transformation.IVectorExtractor.BBN = bbn_primitives.time_series.ivector_extraction:IVectorExtractor',
            'classification.BBNMLPClassifier.BBN = bbn_primitives.sklearn_wrap.mlp_classifier:BBNMLPClassifier',
        ],
    },
)
