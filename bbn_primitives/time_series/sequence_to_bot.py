import typing

import numpy as np
import scipy.stats
import stopit
import sys, os
import logging

import sklearn.svm
import sklearn.naive_bayes
from sklearn.feature_extraction.text import TfidfTransformer

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

Inputs = List #List[List[d3m_ndarray]]
Outputs = d3m_dataframe

_logger = logging.getLogger('d3m.primitives.data_transformation.sequence_to_bag_of_tokens.BBN')

class Params(params.Params):
    vocab: dict

class Hyperparams(hyperparams.Hyperparams):
		pass

class SequenceToBagOfTokens(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
		# TODO: FeaturizationPrimitiveBase provides fit(Inputs, Outputs)
		#	while FeaturizationTransformerPrimitiveBase doesn't provide fit() at all
		# Here we would need a Transformer with unsupervised learning which is missing

    """
    BBN D3M Sequence to Bag-of-Tokens transformer takes list of sequences of tokens (each token is assumed to be vector/n-gram of discrete symbols) as input,identifies unique tokens and represents sequences by counts of individual tokens (bag of tokens).
    Input: List of arrays with tokens on rows [ num_tokens, symbol_ngram ]
    Output: Array of bag of tokens of shape [ num_inputs, num_unique_tokens ]
    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': 'bdcceba0-b6bd-4611-92de-225e3353f07d',
        'version': __version__,
        'name': "Sequence to Bag-of-Tokens transformer",
        'description': """BBN D3M Sequence to Bag-of-Tokens identifies takes list of sequences of tokens (each token is assumed to be vector/n-gram of discrete symbols) as input,
											  identifies unique tokens and represents sequences by counts of individual tokens (bag of tokens).
												Input: List of arrays with tokens on rows [ num_tokens, symbol_ngram ]\n
												Output: Array of bag of tokens of shape [ num_inputs, num_unique_tokens ]\n
                        Applications include: audio, time-series classification""",
        'keywords': [],
        'source': {
            'name': __author__,
            'contact':'mailto:prasannakumar.muthukumar@raytheon.com',
            'uris': [
                'https://github.com/BBN-E/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/sequence_to_bot.py'.format(
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
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.data_transformation.sequence_to_bag_of_tokens.BBN', #'d3m.primitives.bbn.time_series.SequenceToBagOfTokens', #'d3m.primitives.data_transformation.sequence_to_bag_of_tokens.BBN',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.DATA_PROFILING], # TODO: review before submission
        'primitive_family': metadata_module.PrimitiveFamily.DATA_TRANSFORMATION, # TODO: review before submission
    })

    def __init__(
        self, *, hyperparams: Hyperparams, random_seed: int = 0,
        docker_containers: typing.Dict[str, DockerContainer] = None
    ) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._training_inputs = None
        self._vocab: dict = None
        self._fitted: bool = False

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Arguments
            - inputs: List( # Data
                         List( # Segments
                            [ context0, ..., contextN ], ... # for N-gram
                         )
                       ),
        """
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None:
            raise Exception('Missing training data')

        with stopit.ThreadingTimeout(timeout) as timer:
            self._vocab = seq_vocab(self._training_inputs)
            #train_x = seq_to_tokenfreq_csr(self._training_inputs, self._vocab)
            self._fitted = True

        if timer.state == timer.EXECUTED:
            return CallResult(None)
        else:
            raise TimeoutError('SequenceToBagOfTokens exceeded time limit')

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Arguments
            - inputs: List( # Data
                         List( # Segments
                            [ context0, ..., contextN ], ... # for N-gram
                         )
                       ),

        Returns:
            - List(d3m_ndarray)
        """
        with stopit.ThreadingTimeout(timeout) as timer:
            x = seq_to_tokenfreq_csr(inputs, self._vocab).toarray()
            #outputs = List([ x[i] for i in range(x.shape[0]) ])
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
            raise TimeoutError('SequenceToBagOfTokens exceeded time limit')


    def set_params(self, *, params: Params) -> None:
        self._vocab = params['vocab']
        self._fitted = True

    def get_params(self) -> Params:
        return Params(
                  vocab = self._vocab,
                )

