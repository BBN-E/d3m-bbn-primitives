import numpy as np
import stopit
import sys

from .time_series_common import *

from d3m_types.sequence import ndarray as d3m_ndarray
from d3m_types.sequence import List
from primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase

Inputs = List[d3m_ndarray]
Outputs = List[VarLenSegmentation]

class DiscontinuitySegmentation(FeaturizationTransformerPrimitiveBase[Inputs, Outputs]):

    """
    BBN D3M Sequence Segmentation based on Measuring Signal Discontinuity

    Arguments:
        - short_dist:
        - long_dist:
    """

    __author__ = 'BBN'
    __metadata__ = {
            "common_name": "Discontinuity-based segmentation",
            "algorithm_type": ["Instance Based"],
            "original_name": "bbn_primitives.time_series.DiscontinuitySegmentation",
            "task_type": ["Data preprocessing"],
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
        short_dist=2,
        long_dist=20,
    ):

        self.short_dist = short_dist
        self.long_dist = long_dist

    def produce(self, inputs: Inputs, timeout=None, iterations=None) -> Outputs:
        """
        Arguments:
            - inputs: [ num_frames, num_feats ]

        Returns:
            - List([ num_frames, num_feats ], [ num_frames, num_feats], ...)
        """
        with stopit.ThreadingTimeout(timeout) as timer:

            outputs = Outputs()
            for cinput in inputs:
                if cinput.size == 0:
                    outputs.append(np.array([]))
                    continue

                # dither
                cinput2 = cinput + np.random.randn(*cinput.shape)*1e-9

                norm = np.sqrt(np.power(cinput2, 2).sum(axis=1))
                short_dists = np.sum(cinput2[:-self.short_dist]*cinput2[self.short_dist:],
                                        axis=1)/(norm[:-self.short_dist]*norm[self.short_dist:])
                long_dists = np.sum(cinput2[:-self.long_dist]*cinput2[self.long_dist:],
                                        axis=1)/(norm[:-self.long_dist]*norm[self.long_dist:])

        if timer.state == timer.EXECUTED:
            return short_dists, long_dists
        else:
            raise TimeoutError('DiscontinuitySegmentation exceeded time limit')


#def 
