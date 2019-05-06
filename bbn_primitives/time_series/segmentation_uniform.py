import typing

import numpy as np
import stopit
import sys, os
import logging

from d3m.container import ndarray as d3m_ndarray, Dataset
from d3m.container.list import List
from d3m.metadata import hyperparams, params
from d3m import utils
from d3m.metadata import base as metadata_module
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase

from . import __author__, __version__
from .time_series_common import *

Inputs = List #List[d3m_ndarray]
Outputs = List #List[List[d3m_ndarray]]

_logger = logging.getLogger('d3m.primitives.bbn.time_series.UniformSegmentation')

class Hyperparams(hyperparams.Hyperparams):
    seg_len = hyperparams.Hyperparameter[int](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/TuningParameter',
		#'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = 10,
        description = 'Segment length'
    )
    seg_shift = hyperparams.Hyperparameter[typing.Union[int, type(None)]](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/TuningParameter',
		#'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = None,
        description = 'Segment shift'
    )

class UniformSegmentation(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    """
    BBN D3M Uniform Sequence Segmentation divides the input into different segments.
    Input: List of arrays with feature vectors [ num_frames, num_features ]
    Output: List of lists of segmented sequence of feature vectors, i.e. List( [ seg_length_1, num_features ], [ seg_length_2, num_features ], ...)
    Applications include: audio, time-series classification
    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': '2a6f820a-3943-4bb7-a580-689a91c338f3',
        'version': __version__,
        'name': "Uniform Segmentation",
        'description': """BBN D3M Uniform Segmentation Primitive divides the input into different segments.\n
                       Input: List of arrays with feature vectors [ num_frames, num_features ]\n
                       Output: List of lists of segmented sequence of feature vectors, i.e. List( [ seg_length_1, num_features ], [ seg_length_2, num_features ], ...)\n
                       Applications include: audio, time-series classification""",
        'keywords': [],
        'source': {
            'name': __author__,
            'contact':'mailto:prasannakumar.muthukumar@raytheon.com',
            'uris': [
                'https://github.com/BBN-E/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/segmentation_uniform.py'.format(
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
        'python_path': 'd3m.primitives.time_series_segmentation.UniformSegmentation.BBN',#'d3m.primitives.bbn.time_series.UniformSegmentation', #'d3m.primitives.time_series_segmentation.uniform_segmentation.BBN',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.UNIFORM_TIME_SERIES_SEGMENTATION],
        'primitive_family': metadata_module.PrimitiveFamily.TIME_SERIES_SEGMENTATION,
    })

    def __init__(
        self, *, hyperparams: Hyperparams, random_seed: int = 0,
        docker_containers: typing.Dict[str, DockerContainer] = None
    ) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        return

    def _seg_length(self, seg_len: int, sampling_rate: float) -> int:
        return int(seg_len * sampling_rate)

    def _seg_shift(self, seg_shift: int, sampling_rate: float) -> int:
        return max(int(seg_shift * sampling_rate), 1)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Arguments:
            - inputs: [ num_frames, num_feats ]

        Returns:
            - List([ num_frames, num_feats ], [ num_frames, num_feats], ...)
        """
        with stopit.ThreadingTimeout(timeout) as timer:
            seg_len = self.hyperparams['seg_len']
            seg_shift = self.hyperparams['seg_shift'] if self.hyperparams['seg_shift'] is not None \
                        else seg_len

            outputs = Outputs()

            metadata = inputs.metadata.clear({
                'schema': metadata_module.CONTAINER_SCHEMA_VERSION,
                'structural_type': Outputs,
                'dimension': {
                    'length': len(outputs)
                }
            }, for_value=outputs).update((metadata_module.ALL_ELEMENTS,), {
                'structural_type': List,
            })

            for input_id in range(len(inputs)):
                cinput = inputs[input_id]
                if cinput.size == 0:
                    outputs.append(d3m_ndarray(np.array([]), generate_metadata=False))
                    continue

                sampling_rate = inputs.metadata.query((input_id,))['sampling_rate'] if 'sampling_rate' in inputs.metadata.query((input_id,)) else 1
                frame_length = self._seg_length(seg_len, sampling_rate)
                frame_shift = self._seg_shift(seg_shift, sampling_rate)

                if cinput.shape[0] <= frame_length:
                    if len(cinput.shape) <= 2:
                        cinput = np.concatenate((cinput,
                                   #np.matlib.repmat(cinput[-1], frame_length-cinput.shape[0], 1)
                                   np.zeros((frame_length-cinput.shape[0],)+cinput.shape[1:], dtype=cinput.dtype)
                                 ))
                shape = ((cinput.shape[0] - frame_length) // frame_shift + 1,
                        frame_length) + cinput.shape[1:]
                strides = (cinput.strides[0]*frame_shift,cinput.strides[0]) + cinput.strides[1:]
                coutput = np.lib.stride_tricks.as_strided(cinput, shape=shape, strides=strides)

                segments = List()
                for i in range(coutput.shape[0]):
                    segments.append(d3m_ndarray(coutput[i], generate_metadata=False))
                outputs.append(segments)

            # Set metadata attribute.
            outputs.metadata = metadata

        if timer.state == timer.EXECUTED:
            return CallResult(outputs)
        else:
            raise TimeoutError('UniformSegmentation exceeded time limit')

