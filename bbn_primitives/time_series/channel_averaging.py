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

_logger = logging.getLogger('d3m.primitives.bbn.time_series.ChannelAverager')

class Hyperparams(hyperparams.Hyperparams):
    pass

class ChannelAverager(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    """
    BBN D3M Channel Averaging Primitive obtain a single channel by averaging the channel to stay within range and avoid clipping.
		

    Arguments:
    	Input: List of arrays with samples of shape [ num_samples, num_channels ]
	Output: List of arrays with samples of shape [ num_samples ]
        Applications include: audio, time-series classification
    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': '35afd4db-e11d-4e2d-a780-9e123b752bd7',
        'version': __version__,
        'name': "Channel Averager",
        'description': """BBN D3M Channel Averaging Primitive obtain a single channel by averaging the channel to stay within range and avoid clipping.\n
			Input: List of arrays with samples of shape [ num_samples, num_channels ]\n
			Output: List of arrays with samples of shape [ num_samples ]\n
			Applications include: audio, time-series classification""",
        'keywords': [],
        'source': {
            'name': __author__,
            'contact':'mailto:prasannakumar.muthukumar@raytheon.com',
            'uris': [
                'https://github.com/BBN-E/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/channel_averaging.py'.format(
                    git_commit=__git_commit__
                ),
                'https://github.com/BBN-E/d3m-bbn-primitives.git',
            ],
        },
        'installation': [
            {
                'type': 'PIP',
                'package_uri': 'git+https://github.com/BBN-E/d3m-bbn-primitives.git@{git_commit}#egg={egg}'.format(
                    git_commit=__git_commit__, egg='bbn_primitives'
            ),
        }],
        'python_path': 'd3m.primitives.bbn.time_series.ChannelAverager', #'d3m.primitives.data_preprocessing.channel_averager.BBN',
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.AUDIO_MIXING],
        'primitive_family': metadata_module.PrimitiveFamily.DATA_PREPROCESSING,
    })

    def __init__(
        self, *, hyperparams: Hyperparams, random_seed: int = 0,
        docker_containers: typing.Dict[str, DockerContainer] = None
    ) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        return


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Arguments:
            - inputs: [ num_samples, num_channels ]

        Returns:
            - [ num_samples ]
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

            for idx in range(len(inputs)):
                cinput = inputs[idx]
                if cinput.ndim > 2:
                    raise ValueError(
                        'Incompatible shape ' + str(cinput.shape)  + ' of cinput.'
                    )
                elif cinput.ndim == 1:
                    coutput = cinput.copy()
                else:
                    coutput = cinput.mean(axis = 1)

                outputs.append(d3m_ndarray(coutput))

                if 'sampling_rate' in inputs.metadata.query((idx,)):
                    metadata = metadata.update((idx,), { 'sampling_rate': inputs.metadata.query((idx,))['sampling_rate'] })

            metadata = metadata.update((), { 'dimension': { 'length': len(outputs) } })
            # Set metadata attribute.
            outputs.metadata = metadata

        if timer.state == timer.EXECUTED:
            return CallResult(outputs)
        else:
            raise TimeoutError('ChannelAverager exceeded time limit')


