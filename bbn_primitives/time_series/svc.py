import typing
from typing import Any, List, Dict, Union
from collections import OrderedDict

import os, sklearn

from sklearn.svm.classes import SVC

from d3m_metadata.container.numpy import ndarray as d3m_ndarray
from d3m_metadata import hyperparams, params, metadata as metadata_module, utils
from primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from primitive_interfaces.base import CallResult

from . import __author__, __version__

Inputs = d3m_ndarray
Outputs = d3m_ndarray

class Params(params.Params):
    support: str
    support_vectors: str
    n_support: str
    dual_coef: d3m_ndarray
    coef: d3m_ndarray
    intercept: d3m_ndarray

class Hyperparams(hyperparams.Hyperparams):
    C = hyperparams.LogUniform(
        default=1,
        lower=0.03125,
        upper=32768,
        semantic_types=[],
        description='Penalty parameter C of the error term. '
    )
    kernel = hyperparams.Enumeration[str](
        values=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        default='rbf',
        description='Specifies the kernel type to be used in the algorithm. It must be one of \'linear\', \'poly\', \'rbf\', \'sigmoid\', \'precomputed\' or a callable. If none is given, \'rbf\' will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape ``(n_samples, n_samples)``. '
    )
    degree = hyperparams.UniformInt(
        default=3,
        lower=1,
        upper=5,
        description='Degree of the polynomial kernel function (\'poly\'). Ignored by all other kernels. '
    )
    gamma = hyperparams.LogUniform(
        default=0.1,
        lower=3.0517578125e-05,
        upper=8,
        description='Kernel coefficient for \'rbf\', \'poly\' and \'sigmoid\'. If gamma is \'auto\' then 1/n_features will be used instead.  coef0 : float, optional (default=0.0) Independent term in kernel function. It is only significant in \'poly\' and \'sigmoid\'. '
    )
    coef0 = hyperparams.Hyperparameter[float](
        default=0,
    )
    probability = hyperparams.Hyperparameter[bool](
        default=True,
        description='Whether to enable probability estimates. This must be enabled prior to calling `fit`, and will slow down that method. '
    )
    shrinking = hyperparams.Hyperparameter[bool](
        default=True,
        description='Whether to use the shrinking heuristic. '
    )
    tol = hyperparams.LogUniform(
        default=0.001,
        lower=0.0001,
        upper=0.1,
        description='Tolerance for stopping criterion. '
    )
    class_weight = hyperparams.Hyperparameter[Union[str, dict]](
        default='balanced',
        description='Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))`` '
    )
    max_iter = hyperparams.Hyperparameter[int](
        default=-1,
        description='Hard limit on iterations within solver, or -1 for no limit. '
    )
    decision_function_shape = hyperparams.Enumeration[str](
        values=['ovr', 'ovo'],
        default='ovr',
        description='Whether to return a one-vs-rest (\'ovr\') decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (\'ovo\') decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). The default of None will currently behave as \'ovo\' for backward compatibility and raise a deprecation warning, but will change \'ovr\' in 0.19.  .. versionadded:: 0.17 *decision_function_shape=\'ovr\'* is recommended.  .. versionchanged:: 0.17 Deprecated *decision_function_shape=\'ovo\' and None*. '
    )


class BBNSVC(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive wrapping for sklearn.ensemble.AdaBoostClassifier
    """

    __author__ = "JPL MARVIN"
    metadata = metadata_module.PrimitiveMetadata({ 
         "algorithm_types": ['ADABOOST'],
         "name": "sklearn.svm.classes.SVC",
         "primitive_family": "CLASSIFICATION",
         "python_path": "d3m.primitives.bbn.time_series.BBNSVC",
         "source": {'name': 'JPL'},
         "version": "0.1.0",
         "id": "a2ee7b2b-99c6-4326-b2e7-e081cd292d78",
         'installation': [{'type': metadata_module.PrimitiveInstallationType.PIP,
                           'package_uri': 'git+https://gitlab.datadrivendiscovery.org/jpl/d3m_sklearn_wrap.git@{git_commit}'.format(
                               git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                            ),
                         }]
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, str] = None,
                 _verbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._clf = SVC(
            C=self.hyperparams['C'],
            kernel=self.hyperparams['kernel'],
            degree=self.hyperparams['degree'],
            gamma=self.hyperparams['gamma'],
            coef0=self.hyperparams['coef0'],
            probability=self.hyperparams['probability'],
            shrinking=self.hyperparams['shrinking'],
            tol=self.hyperparams['tol'],
            class_weight=self.hyperparams['class_weight'],
            max_iter=self.hyperparams['max_iter'],
            decision_function_shape=self.hyperparams['decision_function_shape'],
            verbose=_verbose,
            random_state=self.random_seed,
        )
        self._training_inputs = None
        self._training_outputs = None
        self._fitted = False

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None or self._training_outputs is None:
            raise ValueError("Missing training data.")

        self._clf.fit(self._training_inputs, self._training_outputs)
        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        return CallResult(self._clf.predict(inputs))

    def produce_log_proba(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        return CallResult(self._clf.predict_log_proba(inputs))

    def get_params(self) -> Params:
        return Params(
            support=self._clf.support_,
            support_vectors=self._clf.support_vectors_,
            n_support=self._clf.n_support_,
            dual_coef=self._clf.dual_coef_,
            coef=self._clf.coef_,
            intercept=self._clf.intercept_,
        )

    def set_params(self, *, params: Params) -> None:
        self._clf.support_ = params.support
        self._clf.support_vectors_ = params.support_vectors
        self._clf.n_support_ = params.n_support
        self._clf.dual_coef_ = params.dual_coef
        self._clf.intercept_ = params.intercept

BBNSVC.__doc__ = SVC.__doc__
