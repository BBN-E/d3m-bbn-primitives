import numpy as np
import scipy
import stopit
import sys

from .time_series_common import *

from d3m_types.sequence import ndarray as d3m_ndarray
from d3m_types.sequence import List
from primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase

Inputs = List[d3m_ndarray]
Outputs = List[d3m_ndarray]

class SignalSpectrogram(FeaturizationTransformerPrimitiveBase[Inputs, Outputs]):

    """
    BBN D3M Signal Spectrogram Primitive

    Arguments:
        - frame_mean_norm:
        - preemcoef:
        - nfft:
    """

    __author__ = 'BBN'
    __metadata__ = {
            "common_name": "Signal Spectrogram",
            "algorithm_type": ["Instance Based"],
            "original_name": "bbn_primitives.time_series.SignalSpectrogram",
            "task_type": ["Feature extraction"],
            "compute_resources": {
                "sample_size": [],
                "sample_unit": [],
                "disk_per_node": [],
                "expected_running_time": [],
                "gpus_per_node": [],
                "cores_per_node": [],
                "mem_per_gpu": [],
                "mem_per_node": [],
                "num_nodes": [],
                },
            "handles_regression": False,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
          }

    def __init__(
        self,
        frame_mean_norm=False,
        preemcoef=None,
        nfft=None,
    ):

        self.frame_mean_norm = frame_mean_norm
        self.preemcoef = preemcoef
        self.nfft = nfft

    def produce(self, inputs: Inputs, timeout=None, iterations=None) -> Outputs:
        """
        Arguments:
            - inputs: [ num_windows, window_len ]

        Returns:
            - [ num_windows, nfft ]
        """
        with stopit.ThreadingTimeout(timeout) as timer:
            outputs = Outputs()
            for cinput in inputs:
                if cinput.size == 0:
                    outputs.append(np.array([]))
                    continue

                frame_length = cinput.shape[1]
                if self.nfft is None:
                    nfft = power_bit_length(frame_length)
                elif self.nfft < frame_length:
                    raise ValueError(
                        'SignalSpectrogram: nfft of ' + str(nfft) + ' is not equal ' +\
                        'or larger than frame length of ' + str(frame_length)
                    )
                else:
                    nfft = self.nfft

                window = np.hamming(frame_length)
                if self.frame_mean_norm:
                    cinput = cinput - cinput.mean(axis=1)[:,np.newaxis]
                if self.preemcoef is not None:
                    cinput = preemphasis(cinput, self.preemcoef)
                coutput = scipy.fftpack.fft(cinput*window, nfft)
                coutput = coutput[:,:coutput.shape[1]//2+1]
                outputs.append(coutput)

        if timer.state == timer.EXECUTED:
            return outputs
        else:
            raise TimeoutError('SignalSpectrogram exceeded time limit')
