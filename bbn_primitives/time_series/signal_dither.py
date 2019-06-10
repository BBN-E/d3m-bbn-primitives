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
#from d3m_metadata.container import ndarray as d3m_ndarray
#from d3m_metadata.container import Dataset
#from d3m_metadata.container import List
#from d3m_metadata import hyperparams, metadata as metadata_module, params, container, utils
#from primitive_interfaces import base
#from primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase

from . import __author__, __version__

Inputs = List #List[d3m_ndarray]
Outputs = List #List[d3m_ndarray]

_logger = logging.getLogger('d3m.primitives.data_preprocessing.signal_dither.BBN')

class Hyperparams(hyperparams.Hyperparams):
    level = hyperparams.Bounded[float](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = 1e-4,
        lower = 0.0, upper = 1.0,
        description = 'Dithering level'
    )

    reseed = hyperparams.Hyperparameter[bool](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = True,
        description = 'Reseed for each input instance'
    )

class SignalDither(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    """
    BBN D3M Signal Dithering Primitive intentionally applies noise to randamize quatization error.
    Input : List of arrays with samples of shape [ num_samples, num_channels ]
    Output: List of arrays with samples of shape [ num_samples, num_channels ]
    Applications include: audio, time-series classification
    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': '1ea935ec-e767-4a18-bbd5-b5f66855f4f3',
        'version': __version__,
        'name': "Signal Dithering",
        'description': """BBN D3M Signal Dithering Primitive intentionally applies noice to randamize quatization error.\n
                        Input : List of arrays with samples of shape [ num_samples, num_channels ]\n
			Output: List of arrays with samples of shape [ num_samples, num_channels ]\n
			Applications include: audio, time-series classification""",
        'keywords': [],
        'source': {
            'name': __author__,
            'contact':'mailto:prasannakumar.muthukumar@raytheon.com',
            'uris': [
                'https://github.com/BBN-E/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/signal_dither.py'.format(
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
        'python_path': 'd3m.primitives.data_preprocessing.signal_dither.BBN', #'d3m.primitives.bbn.time_series.SignalDither', #'d3m.primitives.data_preprocessing.signal_dither.BBN',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.SIGNAL_DITHERING],
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
            - inputs: List[d3m_ndarray]

        Returns:
            - List[d3m_ndarray]
        """
        with stopit.ThreadingTimeout(timeout) as timer:
            outputs = Outputs()
            metadata = inputs.metadata

            for cinput in inputs:
                if self.hyperparams['reseed']:
                    np.random.seed(self.random_seed)
                coutput = cinput + self.hyperparams['level'] * (np.random.rand(*cinput.shape)*2-1)
                outputs.append(d3m_ndarray(coutput, generate_metadata=False))


            # Set metadata attribute.
            outputs.metadata = metadata

        if timer.state == timer.EXECUTED:
            return CallResult(outputs)
        else:
            raise TimeoutError('SignalDither exceeded time limit')


