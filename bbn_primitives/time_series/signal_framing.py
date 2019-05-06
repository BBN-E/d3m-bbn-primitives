import typing

import numpy as np
import stopit
import sys, os
import logging

from d3m.container import ndarray as d3m_ndarray, Dataset, List
from d3m.metadata import hyperparams, params
from d3m import utils
from d3m.metadata import base as metadata_module
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase

from . import __author__, __version__

Inputs = List #List[d3m_ndarray]
Outputs = List #List[d3m_ndarray]

_logger = logging.getLogger('d3m.primitives.bbn.time_series.SignalFramer')

class Hyperparams(hyperparams.Hyperparams):
    frame_length_s = hyperparams.Bounded[float](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/TuningParameter',
		#'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = 0.025,
        lower = 0.0, upper = None,
        description = 'Frame length [s]'
    )
    frame_shift_s = hyperparams.Bounded[float](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/TuningParameter',
		#'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = 0.010,
        lower = 0.0, upper = None,
        description = 'Frame shift [s]'
    )
    flatten_output = hyperparams.Hyperparameter[bool](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = False,
        description = 'Flatten output'
    )

class SignalFramer(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    BBN D3M Signal Framing Primitive divides the audio signal into number of frames.
    Input: List of arrays with samples of shape [ num_samples ]
    Output: List of arrays with frames of shape [ num_frames, frame_length ]
    Applications include: audio, time-series classification
    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': '4d7160ef-ca70-4150-b513-36b90817ba45',
        'version': __version__,
        'name': "Signal Framing",
        'description': """BBN D3M Signal Framing Primitive divides the audio signal into number of frames.\n
			Input: List of arrays with samples of shape [ num_samples ]\n
			Output: List of arrays with frames of shape [ num_frames, frame_length ]\n
			Applications include: audio, time-series classification""",
        'keywords': [],
        'source': {
            'name': __author__,
            'contact':'mailto:prasannakumar.muthukumar@raytheon.com',
            'uris': [
                'https://github.com/BBN-E/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/signal_framing.py'.format(
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
        'python_path': 'd3m.primitives.time_series_segmentation.SignalFramer.BBN',#'d3m.primitives.bbn.time_series.SignalFramer', #'d3m.primitives.time_series_segmentation.signal_framer.BBN',
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

    def _frame_length(self, sampling_rate: float) -> int:
        return int(self.hyperparams['frame_length_s'] * sampling_rate)

    def _frame_shift(self, sampling_rate: float) -> int:
        return max(int(self.hyperparams['frame_shift_s'] * sampling_rate), 1)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Arguments:
            - inputs: [ num_samples ]

        Returns:
            - [ num_windows, window_len ]
        """
        with stopit.ThreadingTimeout(timeout) as timer:
            outputs = Outputs()
            metadata = inputs.metadata.clear({
                'schema': metadata_module.CONTAINER_SCHEMA_VERSION,
                'structural_type': Outputs,
                'dimension': {
                    'length': len(outputs)
                }
            }, for_value=outputs).update((metadata_module.ALL_ELEMENTS,), {
                'structural_type': d3m_ndarray,
            })

            for input_id in range(len(inputs)):
                cinput = inputs[input_id]
                # TODO: review the following because it's hacky
                # It was done in the way to enable handling both audio (high sampling_rate) and frames
                sampling_rate = inputs.metadata.query((input_id,))['sampling_rate'] if 'sampling_rate' in inputs.metadata.query((input_id,)) else 1
                frame_length = self._frame_length(sampling_rate)
                frame_shift = self._frame_shift(sampling_rate)

                if cinput.size == 0:
                    outputs.append(d3m_ndarray(np.array([]), generate_metadata=False))
                    continue

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
                outputs.append(d3m_ndarray(
			coutput.flatten() if self.hyperparams['flatten_output'] else coutput,
			generate_metadata=False))

                if 'sampling_rate' in inputs.metadata.query((input_id,)):
                    metadata = metadata.update((input_id,), { 'sampling_rate': inputs.metadata.query((input_id,))['sampling_rate'] })

            #metadata = metadata.update((), { 'dimension': { 'length': len(outputs) } })
            # Set metadata attribute.
            outputs.metadata = metadata

        if timer.state == timer.EXECUTED:
            return CallResult(outputs)
        else:
            raise TimeoutError('SignalFramer exceeded time limit')


