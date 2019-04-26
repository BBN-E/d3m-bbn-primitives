import typing

import numpy as np
import stopit
import sys, os
import logging

from .time_series_common import *
from .segmentation_common import *
from .signal_framing import SignalFramer

import scipy.cluster
from sklearn.cluster import MiniBatchKMeans
from numpy import ndarray

from d3m.container import ndarray as d3m_ndarray, List
from d3m.metadata import hyperparams, params
from d3m import utils
from d3m.metadata import base as metadata_module
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

from . import __author__, __version__

Inputs = List #List[List[d3m_ndarray]]
Outputs = List #List[d3m_ndarray]

_logger = logging.getLogger('d3m.primitives.bbn.time_series.ClusterCurveFittingKMeans')

class Params(params.Params):
    cluster_centers_: ndarray

class Hyperparams(hyperparams.Hyperparams):
    n_init = hyperparams.Hyperparameter[int](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/ControlParameter',
		'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
	],
        default = 10,
        description = 'Number of initializations with different centroid seeds'
    )
    n_clusters = hyperparams.Hyperparameter[int](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/TuningParameter',
		'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
		#'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = 32,
        description = 'Number of clusters'
    )
    max_iter = hyperparams.Hyperparameter[int](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/ControlParameter',
		'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
	],
        default = 300,
        description = 'Maximum number of iterations'
    )

class ClusterCurveFittingKMeans(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):

    """
    BBN D3M Curve Fitting Clusterer performs clustering of segments representations based on the k-means algorithm.The segments are expected to have constant duration (this can be achieved by using the polynomial approximation of variable-length segments, see SegmentCurveFitter.
    Input: List of lists of segmented sequence of polynomial coefficients, i.e. List( [ seg_length, num_features ], [ seg_length, num_features ], ...)
    Output: List of arrays of cluster ids assigned to segments of shape [ num_segments ]
    Applications include: audio, time-series classification

    For details, refer to Gish, H. and Ng, K., 1996, October. Parametric trajectory models for speech recognition. In Spoken Language, 1996. ICSLP 96. Proceedings., Fourth International Conference on (Vol. 1, pp. 466-469). IEEE.
    or Yeung, S.K.A., Li, C.F. and Siu, M.H., 2005, March. Sub-phonetic polynomial segment model for large vocabulary continuous speech recognition. In Acoustics, Speech, and Signal Processing, 2005. Proceedings.(ICASSP'05). IEEE International Conference on (Vol. 1, pp. I-193). IEEE.
    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': 'b358a1fd-8bf8-4991-935f-2f1806dae54d',
        'version': __version__,
        'name': "Clustering for Curve Fitting",
        'description': """BBN D3M Curve Fitting Clusterer performs clustering of segments representations based on the k-means algorithm.\n
	               The segments are expected to have constant duration (this can be achieved by using the polynomial approximation of variable-length segments, see SegmentCurveFitter.\n
                       Input: List of lists of segmented sequence of polynomial coefficients, i.e. List( [ seg_length, num_features ], [ seg_length, num_features ], ...)\n
                       Output: List of arrays of cluster ids assigned to segments of shape [ num_segments ]
                       Applications include: audio, time-series classification""",
        'keywords': [],
        'source': {
            'name': __author__,
            'contact':'mailto:prasannakumar.muthukumar@raytheon.com',
            'uris': [
                'https://github.com/BBN-E/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/cluster_curve_fitting_kmeans.py'.format(
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
        'python_path': 'd3m.primitives.bbn.time_series.ClusterCurveFittingKMeans', #'d3m.primitives.clustering.cluster_curve_fitting_kmeans.BBN',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.K_MEANS_CLUSTERING],
        'primitive_family': metadata_module.PrimitiveFamily.CLUSTERING,
    })


    def __init__(
        self, *, hyperparams: Hyperparams, random_seed: int = 0,
        docker_containers: typing.Dict[str, DockerContainer] = None
    ) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        #self._cluster_centers: d3m_ndarray = None
        self._training_inputs: Inputs = None
        #self._model: KMeans = KMeans(n_clusters=self.hyperparams['n_clusters'],
        self._model = MiniBatchKMeans(n_clusters=self.hyperparams['n_clusters'],
                                      init='k-means++',
                                      n_init=self.hyperparams['n_init'],
                                      max_iter=self.hyperparams['max_iter'],
                                      compute_labels=False,
                                      random_state=self.random_seed,
                      )
        self._fitted: bool = False

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Arguments
            - inputs: List( # Data
                         List( # Segments
                            [ deg, num_feats ], ...
                         )
                       ),
        """
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None:
            raise Exception('Missing training data')

        with stopit.ThreadingTimeout(timeout) as timer:
            inputs_curve_fitting = self._training_inputs
            num_data = sum([ len(x) for x in inputs_curve_fitting ]) # number of segments, each segment if formed by multiple data samples
            deg, num_feats = inputs_curve_fitting[0][0].shape
            betas = np.vstack([
                        np.array([
                            segment.flatten() for segment in cinput
                        ]) for cinput in inputs_curve_fitting if len(cinput) > 0
                    ])

            self._model.fit(betas)
            self._fitted = True

        if timer.state == timer.EXECUTED:
            return CallResult(None)
        else:
            raise TimeoutError('ClusterCurveFittingKMeans exceeded time limit')

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Arguments:
            - inputs: List( # Data
                        List( # Segments
                          [ deg, num_feats ], ...
                        )
                      )


        Returns:
            - List(d3m_ndarray)
        """
        if self._fitted is False:
            raise ValueError("Calling produce before fitting.")

        with stopit.ThreadingTimeout(timeout) as timer:
            outputs = List()

            metadata = inputs.metadata.clear({
                'schema': metadata_module.CONTAINER_SCHEMA_VERSION,
                'structural_type': Outputs,
                'dimension': {
                    'length': len(outputs)
                }
            }, for_value=outputs).update((metadata_module.ALL_ELEMENTS,), {
                'structural_type': d3m_ndarray,
            })
            #}, for_value=outputs, source=self)

            for cinput in inputs:
                if len(cinput) == 0:
                    outputs.append(d3m_ndarray(np.array([]), generate_metadata=False))
                    continue

                betas = np.array([
                            segment.flatten() for segment in cinput
                        ])
                outputs.append(d3m_ndarray(self._model.predict(betas), generate_metadata=False))
                #outputs.append(self._model.predict(betas))

            # Set metadata attribute.
            outputs.metadata = metadata

        if timer.state == timer.EXECUTED:
            return CallResult(outputs)
        else:
            raise TimeoutError('ClusterCurveFittingKMeans exceeded time limit')

    def get_params(self) -> Params:
        return Params(cluster_centers_=self._model.cluster_centers_)

    def set_params(self, *, params: Params) -> None:
        self._model.cluster_centers_ = params['cluster_centers_']
        self._fitted = True


