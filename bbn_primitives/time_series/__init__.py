"""
   bbn_primitives.time_series sub-package
   __init__.py
"""

__version__ = '0.1.4'
__author__ = 'BBN'


## sub-packages
from .audio_reader import AudioReader
#from .csv_reader import CSVReader
from .channel_averaging import ChannelAverager
from .signal_dither import SignalDither
from .signal_framing import SignalFramer
#from .signal_spectrogram import SignalSpectrogram
from .signal_mfcc import SignalMFCC
#from .segmentation_discont import DiscontinuitySegmentation
from .segmentation_uniform import UniformSegmentation
from .segment_curve_fitter import SegmentCurveFitter
#from .cluster_curve_fitting import ClusterCurveFitting
from .cluster_curve_fitting_kmeans import ClusterCurveFittingKMeans
from .sequence_to_bot import SequenceToBagOfTokens
from .tfidf_transformer import BBNTfidfTransformer
from .ivector_extraction import IVectorExtractor
from .targets_reader import TargetsReader
#from .svc import BBNSVC
#from .sequence_bow_classifier import SequenceBagOfWordsClassifier
#from .sequence_bow_svm_classifier import SequenceBagOfWordsSVMClassifier
