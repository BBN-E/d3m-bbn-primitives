import typing

import numpy as np
import scipy.stats
import stopit
import sys, os
import logging

from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse.csr import csr_matrix

from .time_series_common import *
from .segmentation_common import *
from .signal_framing import SignalFramer

from d3m.container import ndarray as d3m_ndarray, List
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.metadata import base as metadata_module
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase


from . import __author__, __version__

Inputs = d3m_dataframe
Outputs = d3m_dataframe

_logger = logging.getLogger('d3m.primitives.bbn.time_series.BBNTfidfTransformer')

class Params(params.Params):
    _idf_diag: typing.Optional[csr_matrix]

class Hyperparams(hyperparams.Hyperparams):
    norm = hyperparams.Enumeration[str](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
          #'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        ],
        values=['l1', 'l2'],
        default='l2',
        description='Norm used to normalize term vectors. None for no normalization.'
    )
    use_idf = hyperparams.Hyperparameter[bool](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
          #'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        ],
        default=True,
        description='Enable inverse-document-frequency reweighting.',
    )
    smooth_idf = hyperparams.Hyperparameter[bool](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
          #'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        ],
        default=True,
        description='Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.',
    )
    sublinear_tf = hyperparams.Hyperparameter[bool](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
          #'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        ],
        default=False,
        description='Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).',
    )

class BBNTfidfTransformer(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    BBN D3M TfidfTransformer wraps sklearn.feature_extraction.text.TfidfTransformer. It transforms a count (bag of tokens) matrix to a normalized tf or tf-idf representation.
    Input: Array of bag of tokens of shape [ n_samples, num_tokens ]
    Output: Array of tf or tf-idf representation of bag of tokens of shape [ n_samples, num_tokens ]
    Applications include: audio, time-series, text 
    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': 'fefcb78f-c2f5-4557-a23b-0910b2127769',
        'version': __version__,
        'name': "Tfidf Transformer",
        'description': """BBN D3M TFIDF transformer wraps sklearn.feature_extraction.text.TfidfTransformer\n
                       It transforms a count (bag of tokens) matrix to a normalized tf or tf-idf representation.\n
											 Input: Array of bag of tokens of shape [ n_samples, num_tokens ]\n
											 Output: Array of tf or tf-idf representation of bag of tokens of shape [ n_samples, num_tokens ]\n
                       Applications include: audio, time-series classification""",
        'keywords': [],
        'source': {
            'name': __author__,
            'contact':'mailto:prasannakumar.muthukumar@raytheon.com',
            'uris': [
                'https://github.com/BBN-E/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/tfidf_transformer.py'.format(
                    git_commit=__git_commit__
                ),
                'https://github.com/BBN-E/d3m-bbn-primitives.git',
            ],
        },
        'installation': [{
            'type': 'PIP',
            'package_uri': 'git+https://github.com/BBN-E/d3m-bbn-primitives.git@{git_commit}#egg={egg}'.format(
                git_commit=__git_commit__, egg='bbn_primitives'
            ),
        }],
        'python_path': 'd3m.primitives.feature_extraction.BBNTfidfTransformer.BBN', #'d3m.primitives.bbn.time_series.BBNTfidfTransformer', #'d3m.primitives.feature_extraction.tfidf_transformer.BBN',
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.FEATURE_SCALING], 
        'primitive_family': metadata_module.PrimitiveFamily.FEATURE_EXTRACTION, 
    })

    def __init__(
        self, *, hyperparams: Hyperparams, random_seed: int = 0,
        docker_containers: typing.Dict[str, DockerContainer] = None
    ) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._training_inputs = None
        self._tfidf = TfidfTransformer(
                norm = self.hyperparams['norm'],
                use_idf = self.hyperparams['use_idf'],
                smooth_idf  = self.hyperparams['smooth_idf'],
                sublinear_tf = self.hyperparams['sublinear_tf'],
            )
        self._fitted: bool = False

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None:
            raise Exception('Missing training data')

        with stopit.ThreadingTimeout(timeout) as timer:
            self._tfidf.fit(self._training_inputs)
            self._fitted = True

        if timer.state == timer.EXECUTED:
            return CallResult(None)
        else:
            raise TimeoutError('BBNTfidfTransformer exceeded time limit')

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        with stopit.ThreadingTimeout(timeout) as timer:
            x = self._tfidf.transform(inputs).toarray()
            outputs = d3m_dataframe(x, generate_metadata=False)

            metadata = inputs.metadata.clear({
                'schema': metadata_module.CONTAINER_SCHEMA_VERSION,
                'structural_type': type(outputs),
                'semantic_types': [ 'https://metadata.datadrivendiscovery.org/types/Table' ],
                'dimension': {
                    'length': outputs.shape[0],
                    'name': 'rows',
                    'semantic_types': [ 'https://metadata.datadrivendiscovery.org/types/TabularRow' ]
                }
            }, for_value=outputs).update(
                ((metadata_base.ALL_ELEMENTS,)), {
                'dimension': {
                    'length': outputs.shape[1],
                    'name': 'columns',
                    'semantic_types': [ 'https://metadata.datadrivendiscovery.org/types/TabularColumn' ]
                }
                }
            ).update(
                ((metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS)), {
                #'structural_type': self._v.dtype,
                'semantic_types': [ 'https://metadata.datadrivendiscovery.org/types/Attribute' ],
                }
            )

            # Set metadata attribute.
            outputs.metadata = metadata

        if timer.state == timer.EXECUTED:
            return CallResult(outputs)
        else:
            raise TimeoutError('BBNTfidfTransformer exceeded time limit')


    def get_params(self) -> Params:
        if not self._fitted:
            raise ValueError("Fit not performed.")
        return Params(
                _idf_diag = getattr(self._tfidf, '_idf_diag', None)
            )


    def set_params(self, *, params: Params) -> None:
        self._tfidf._idf_diag = params['_idf_diag']
        self._fitted = True
