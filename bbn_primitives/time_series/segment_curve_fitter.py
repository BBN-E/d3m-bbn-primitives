import typing

import numpy as np
import stopit
import sys, os
import logging

from .time_series_common import *
from .segmentation_common import *

from d3m.container import ndarray as d3m_ndarray, Dataset, List
from d3m.metadata import hyperparams, params
from d3m import utils
from d3m.metadata import base as metadata_module
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase


from . import __author__, __version__

Inputs = List #List[List[d3m_ndarray]]
Outputs = List #List[List[d3m_ndarray]]

_logger = logging.getLogger('d3m.primitives.bbn.time_series.SegmentCurveFitter')

class Hyperparams(hyperparams.Hyperparams):
    deg = hyperparams.Hyperparameter[int](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/TuningParameter',
	],
        default = 2,
        description = 'Polynomial degree'
    )

class SegmentCurveFitter(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    """
    BBN D3M Segment Curve Fitter takes segmented sequence of feature vectors as input and for each segment and feature dimension separately replaces the series of values by coefficients of its polynomial approximation of specified degree
    Input: List of lists of segmented sequence of feature vectors, i.e. List( [ seg_length_1, num_features ], [ seg_length_2, num_features ], ...)
    Output: List of lists of segmented sequence of polynomial coefficients, i.e. List( [ poly_deg, num_features ], [ poly_deg, num_features ], ...)
    Applications include: audio, time-series classification

    For details, refer to Gish, H. and Ng, K., 1996, October. Parametric trajectory models for speech recognition. In Spoken Language, 1996. ICSLP 96. Proceedings., Fourth International Conference on (Vol. 1, pp. 466-469). IEEE.
    """
    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': '7c1d88a3-2388-4ba8-97c6-aa0aa2673024',
        'version': __version__,
        'name': "Segment Curve Fitter",
        'description': """BBN D3M Segment Curve Fitter takes segmented sequence of feature vectors as input and for each segment and feature dimension separately replaces the series of values by coefficients of its polynomial approximation of specified degree\n
                       Input: List of lists of segmented sequence of feature vectors, i.e. List( [ seg_length_1, num_features ], [ seg_length_2, num_features ], ...)\n
                       Output: List of lists of segmented sequence of polynomial coefficients, i.e. List( [ poly_deg, num_features ], [ poly_deg, num_features ], ...)\n
                       Applications include: audio, time-series classification""",
        'keywords': [],
        'source': {
            'name': __author__,
            'contact':'mailto:prasannakumar.muthukumar@raytheon.com',
            'uris': [
                'https://gitlab.datadrivendiscovery.org/BBN/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/segment_curve_fitter.py'.format(
                    git_commit=__git_commit__
                ),
                'https://gitlab.datadrivendiscovery.org/BBN/d3m-bbn-primitives.git',
            ],
        },
        'installation': [{
            'type': 'PIP',
            'package_uri': 'git+https://gitlab.datadrivendiscovery.org/BBN/d3m-bbn-primitives.git@{git_commit}#egg={egg}'.format(
                git_commit=__git_commit__, egg='bbn_primitives'
            ),
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.bbn.time_series.SegmentCurveFitter', #'d3m.primitives.data_transformation.segment_curve_fitter.BBN',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.PARAMETRIC_TRAJECTORY_MODELING],
        'primitive_family': metadata_module.PrimitiveFamily.DATA_TRANSFORMATION,
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
            - inputs: List(
                        List([ num_frames, num_feats ], [ num_frames, num_feats], ...)
                      )


        Returns:
            - List( # Data
                List( # Segments
                  [ deg, num_feats ], ...
                )
              )
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
                'structural_type': List,
            })

            for cinput in inputs:
                coutput = List()
                for segment in cinput:
                    if segment.ndim != 2 or segment.shape[0] < self.hyperparams['deg']:
                        raise ValueError(
                            'Incompatible shape ' + str(segment.shape)  + ' of cinput.'
                        )
                    n = segment.shape[0]

                    x = np.linspace(0., 1., n)
                    p = np.polyfit(x, segment, deg = self.hyperparams['deg'])
                    E = segment - applyFitting(n, p)
#                    for d in range(segment.shape[1]):
#                        pfcn = np.poly1d(p[:, d])
#                        E[:, d] = segment[:, d]-pfcn(x)
                    Sigma = np.dot(E.T, E)/n
                    #segment_output = CurveFitting(deg = self.deg,
                    #        beta = p, sigma = Sigma, N = n)
                    coutput.append(d3m_ndarray(p, generate_metadata=False))
                outputs.append(coutput)

            # Set metadata attribute.
            outputs.metadata = metadata

        if timer.state == timer.EXECUTED:
            return CallResult(outputs)
        else:
            raise TimeoutError('SegmentCurveFitter exceeded time limit')

