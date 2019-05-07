import typing

import numpy as np
import scipy.stats
import stopit
import sys, os
import logging

from sklearn.mixture import GaussianMixture
import sklearn.mixture.gaussian_mixture
from sklearn import preprocessing
from .ivector_tools import *

from d3m.container import ndarray as d3m_ndarray, List
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.metadata import base as metadata_module
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from numpy import ndarray


from . import __author__, __version__

Inputs = List #List[d3m_ndarray]
Outputs = d3m_dataframe

_logger = logging.getLogger('d3m.primitives.data_transformation.i_vector_extractor.IVectorExtractor')

class Params(params.Params):
    weights: ndarray
    means: ndarray
    covs: ndarray
    cov_type: str
    v: ndarray

class Hyperparams(hyperparams.Hyperparams):
    num_gauss = hyperparams.Bounded[int](
				semantic_types = [
					'https://metadata.datadrivendiscovery.org/types/TuningParameter',
					'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
				],
        lower=1, upper=None,
        default=32,
        description='Number of Gaussian components'
    )
    gmm_covariance_type = hyperparams.Enumeration[str](
				semantic_types = [
					'https://metadata.datadrivendiscovery.org/types/TuningParameter',
					'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
				],
        values=['full', 'tied', 'diag', 'spherical'],
        default='diag',
        description='Type of covariance matrix of Gaussian components'
    )
    max_gmm_iter = hyperparams.Bounded[int](
				semantic_types = [
					'https://metadata.datadrivendiscovery.org/types/ControlParameter',
					'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
				],
        lower=1,upper=None,
        default=20,
        description='Number of GMM estimation iterations'
    )
    # TODO: consider subsampling data for GMM estimation

    num_ivec_iter = hyperparams.Bounded[int](
				semantic_types = [
					'https://metadata.datadrivendiscovery.org/types/ControlParameter',
					'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
				],
        lower=1,upper=None,
        default=7,
        description='Number of i-vector estimation iterations'
    )
    ivec_dim = hyperparams.Bounded[int](
				semantic_types = [
					'https://metadata.datadrivendiscovery.org/types/TuningParameter',
					'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
				],
        lower=1,upper=None,
        default=50,
        description='Dimensionality of i-vectors'
    )
    ivec_normalize = hyperparams.Hyperparameter[bool](
        semantic_types = [
                'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        ],
        default = False,
        description = 'Read audio as mono'
    )


class IVectorExtractor(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    BBN D3M I-vector extractor extracts i-vectors for variable-length input sequences of feature vectors.
    Input: List of arrays with feature vectors extracted for frames [ num_frames, num_features ]
    Output: Array of i-vectors of shape [ num_inputs, ivec_dim ]
    Applications include: audio, time-series classification
    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': '1c5080bd-7b2f-4dbb-ac5f-0a65b59526a7',
        'version': __version__,
        'name': "I-vector extractor",
        
        'description': """BBN D3M I-vector extractor extracts i-vectors for variable-length input sequences of feature vectors.\n
                        Input: List of arrays with feature vectors extracted for frames [ num_frames, num_features ]\n
												Output: Array of i-vectors of shape [ num_inputs, ivec_dim ]\n
                        Applications include: audio, time-series classification""",
        'keywords': [],
        'source': {
            'name': __author__,
            'contact':'mailto:prasannakumar.muthukumar@raytheon.com',
            'uris': [
                'https://github.com/BBN-E/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/ivector_extraction.py'.format(
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
        'python_path': 'd3m.primitives.data_transformation.i_vector_extractor.IVectorExtractor', #'d3m.primitives.bbn.time_series.IVectorExtractor', #'d3m.primitives.data_transformation.ivector_extractor.BBN',
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.DATA_CONVERSION],
        'primitive_family': metadata_module.PrimitiveFamily.DATA_TRANSFORMATION,
    })

    def __init__(
        self, *, hyperparams: Hyperparams, random_seed: int = 0,
        docker_containers: typing.Dict[str, DockerContainer] = None
    ) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._training_inputs = None
        self._gmm = GaussianMixture(
                n_components = self.hyperparams['num_gauss'],
                covariance_type = self.hyperparams['gmm_covariance_type'],
                max_iter = self.hyperparams['max_gmm_iter']
            )
        self._v = None
        self._fitted: bool = False

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None:
            raise Exception('Missing training data')

        with stopit.ThreadingTimeout(timeout) as timer:
            # Train GMM
            _logger.info('Training GMM')
            num_data = len(self._training_inputs)
            #for idx in range(num_data):
            #    X = self._training_inputs[idx]
            #    print(X.shape)
            self._gmm.fit(np.vstack(
                    [ x for x in self._training_inputs if len(x.shape) == 2 ]
                  ))

            # Train i-vector extractor
            self._v = np.random.randn(self._gmm.n_components*self._gmm.means_.shape[1],
                                        self.hyperparams['ivec_dim'])
            _logger.info('Training i-vector extractor')
            N = np.zeros((num_data, self._gmm.n_components))
            F = np.zeros((num_data, self._gmm.n_components*self._gmm.means_.shape[1]))
            # TODO: Do the E-step in mini-batches to prevent memory overflow
            for idx in range(num_data):
                X = self._training_inputs[idx]
                if len(X.shape) != 2:
                    continue
                gamma = self._gmm.predict_proba(X)
                N0 = gamma.T.sum(axis = 1)
                F0 = gamma.T.dot(X)
                N0, F0 = normalize_stats(N0, F0,
                      self._gmm.means_, self._gmm.precisions_cholesky_)
                N[idx, :] = N0
                F[idx, :] = F0.flatten()

            for ivec_iter in range(self.hyperparams['num_ivec_iter']):
                _logger.info('Training i-vector extractor - iteration %d' % ivec_iter)
                num_data = len(self._training_inputs)
                A, C, Amd, Cmd, Nmd = None, None, None, None, None
                VtV, I = None, None

                A, C, Amd, Cmd, Nmd = E_step_with_MD(N, F, self._v,
                        VtV, I, A, C, Amd, Cmd, Nmd)
                em_v = M_step(A, C)
                md_v = M_step_MD(Amd, Cmd, Nmd, em_v)
                self._v = md_v.reshape((self._gmm.n_components*self._gmm.means_.shape[1],
                                        self.hyperparams['ivec_dim']))

            self._fitted = True

        if timer.state == timer.EXECUTED:
            return CallResult(None)
        else:
            raise TimeoutError('IVectorExtractor exceeded time limit')

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        with stopit.ThreadingTimeout(timeout) as timer:
            num_data = len(inputs)
            outputs = np.empty((num_data, self.hyperparams['ivec_dim']), dtype=self._v.dtype)
            VtV = compute_VtV(self._v, self._gmm.n_components)
            I = np.eye(self.hyperparams['ivec_dim'], dtype=self._v.dtype)
            for idx in range(num_data):
                X = inputs[idx]
                if len(X.shape) != 2:
                    outputs[idx] = np.zeros((self.hyperparams['ivec_dim']))
                    continue
                gamma = self._gmm.predict_proba(X)
                N0 = gamma.T.sum(axis = 1)
                F0 = gamma.T.dot(X)
                N0, F0 = normalize_stats(N0, F0,
                            self._gmm.means_, self._gmm.precisions_cholesky_)
                ivec = estimate_i(row(N0.astype(self._v.dtype)),
                                                row(F0.astype(self._v.dtype)),
                                                self._v, VtV, I)
                outputs[idx] = ivec.flatten()
            #adding normalization
            if(self.hyperparams['ivec_normalize']):
                outputs = preprocessing.normalize(outputs, norm='l2')
            outputs = d3m_dataframe(outputs, generate_metadata=False)

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
            raise TimeoutError('IVectorExtractor exceeded time limit')


    def get_params(self) -> Params:
        return Params(
                 weights = self._gmm.weights_,
                 means = self._gmm.means_,
                 covs = self._gmm.covariances_,
                 cov_type = self._gmm.covariance_type,
                 v = self._v
               )

    def set_params(self, *, params: Params) -> None:
        assert self._gmm.covariance_type == params['cov_type']
        # Consider adding additional assertations regarding dims

        self._gmm.weights_ = params['weights']
        self._gmm.means_ = params['means']
        self._gmm.covariances_ = params['covs']
        self._gmm.precisions_cholesky_ = sklearn.mixture.gaussian_mixture._compute_precision_cholesky(params['covs'], params['cov_type'])
        self._v = params['v']
