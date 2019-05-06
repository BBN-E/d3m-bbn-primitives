import typing

import numpy as np
import stopit
import sys, os
import scipy.fftpack
import logging

from d3m.container import ndarray as d3m_ndarray
from d3m.container.list import List
from d3m.metadata import hyperparams, params
from d3m import utils
from d3m.metadata import base as metadata_module
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase



from .time_series_common import *

from . import __author__, __version__

Inputs = List #List[d3m_ndarray]
Outputs = List #List[d3m_ndarray]

_logger = logging.getLogger('d3m.primitives.bbn.time_series.SignalMFCC')

class Hyperparams(hyperparams.Hyperparams):
    ## TODO: sampling_rate should become metadata accompanying the input data - sample specific
    #sampling_rate = hyperparams.Bounded(
    #    default = 16000.0,
    #    lower = 0.0, upper = None,
    #    description = 'Sampling rate'
    #)
    frame_mean_norm = hyperparams.Hyperparameter[bool](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/TuningParameter',
		#'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = False,
        description = 'Frame mean normalization'
    )
    preemcoef = hyperparams.Hyperparameter[typing.Union[float, type(None)]](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/ControlParameter',
		#'https://metadata.datadrivendiscovery.org/types/TuningParameter',
	],
        default = None,
        description = 'preemphasis'
    )
    nfft = hyperparams.Hyperparameter[typing.Union[int, type(None)]](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/TuningParameter',
		#'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = None,
        description = 'FFT size'
    )
    num_chans = hyperparams.Hyperparameter[int](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/TuningParameter',
		#'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = 20,
        description = 'Number of filter banks'
    )
    num_ceps = hyperparams.Hyperparameter[int](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/TuningParameter',
		#'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = 12,
        description = 'Number of cepstral coefficients (output dimensionality)'
    )
    use_power = hyperparams.Hyperparameter[bool](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/TuningParameter',
		#'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = False,
        description = 'Use power'
    )
    cep_lifter = hyperparams.Hyperparameter[float](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/TuningParameter',
		#'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = 22.0,
        description = 'Cepstrum lifter weight'
    )

class SignalMFCC(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    BBN D3M Signal Mel-Frequency Cepstral Coefficient (MFCC) Extraction Primitive extracts feature vectors of Mel-frequency Cepstral Coefficients (MFCCs) for frames of audio.
    Input: List of arrays with frames of shape [ num_frames, frame_length ]
    Output: List of arrays with feature vectors extracted for frames [ num_frames, num_features ]
    Applications include: audio, time-series classification
    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': 'a184a1d1-3187-4d1f-99c6-e1d5665c2c99',
        'version': __version__,
        'name': "MFCC Feature Extraction",
        'description': """BBN D3M MFCC Feature Extraction Primitive extracts feature vectors of Mel-frequency Cepstral Coefficients (MFCCs) for frames of audio.\n
                        Input: List of arrays with frames of shape [ num_frames, frame_length ]\n
                        Output: List of arrays with feature vectors extracted for frames [ num_frames, num_features ]\n
                        Applications include: audio, time-series classification""",
        'keywords': [],
        'source': {
            'name': __author__,
            'contact':'mailto:prasannakumar.muthukumar@raytheon.com',
            'uris': [
                'https://github.com/BBN-E/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/signal_mfcc.py'.format(
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
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.feature_extraction.SignalMFCC.BBN', #'d3m.primitives.bbn.time_series.SignalMFCC', #'d3m.primitives.feature_extraction.signal_mfcc.BBN',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.MFCC_FEATURE_EXTRACTION],
        'primitive_family': metadata_module.PrimitiveFamily.FEATURE_EXTRACTION,
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
            - inputs: [ num_windows, window_len ]

        Returns:
            - [ num_windows, nfft ]
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
                if cinput.size == 0:
                    outputs.append(d3m_ndarray(np.array([]), generate_metadata=False))
                    continue

                frame_length = cinput.shape[1]
                if self.hyperparams['nfft'] is None:
                    nfft = power_bit_length(frame_length)
                elif self.hyperparams['nfft'] < frame_length:
                    raise ValueError(
                        'SignalMFCC: nfft of ' + str(nfft) + ' is not equal ' +\
                        'or larger than frame length of ' + str(frame_length)
                    )
                else:
                    nfft = self.hyperparams['nfft']

                cinput *= 2**15 # TODO: this should be rather optional - add hyperparam
                window = np.hamming(frame_length)
                if self.hyperparams['frame_mean_norm']:
                    cinput = cinput - cinput.mean(axis=1)[:,np.newaxis]
                if self.hyperparams['preemcoef'] is not None:
                    cinput = preemphasis(cinput, self.hyperparams['preemcoef'])
                use_power = self.hyperparams['use_power'] + 1

                sampling_rate = inputs.metadata.query((input_id,))['sampling_rate']
                mel_filters = mel_fbank_filters(nfft, sampling_rate, self.hyperparams['num_chans'])
                idct_mat = scipy.fftpack.idct(
                                np.eye(self.hyperparams['num_ceps']+1, self.hyperparams['num_chans']), norm='ortho').T
                idct_mat[:,0] = np.sqrt(2.0/self.hyperparams['num_chans'])
                x = np.abs(scipy.fftpack.fft(cinput*window, nfft))
                x = x[:,:x.shape[1]//2+1]
                coutput = (x**use_power).dot(mel_filters)
                coutput = np.log(np.maximum(1.0, (x**use_power).dot(mel_filters))).dot(idct_mat)
                if self.hyperparams['cep_lifter'] is not None and self.hyperparams['cep_lifter'] > 0:
                    coutput *= 1.0 + 0.5 * self.hyperparams['cep_lifter'] * \
                            np.sin(np.pi * np.arange(self.hyperparams['num_ceps']+1) / self.hyperparams['cep_lifter'])
                outputs.append(d3m_ndarray(coutput, generate_metadata=False))


            # Set metadata attribute.
            outputs.metadata = metadata

        if timer.state == timer.EXECUTED:
            return CallResult(outputs)
        else:
            raise TimeoutError('SignalMFCC exceeded time limit')

