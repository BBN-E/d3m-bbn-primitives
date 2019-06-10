from typing import Any, Callable, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy as np
import typing
import logging
import stopit

# Custom import commands if any
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing.label import LabelBinarizer

from d3m.container import List as List
from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.metadata import base as metadata_module
from d3m import utils
from d3m.primitive_interfaces.base import CallResult, DockerContainer
import common_primitives.utils as common_utils
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin

from . import __author__, __version__

Inputs = d3m_dataframe
Outputs = d3m_dataframe

_logger = logging.getLogger('d3m.primitives.classification.mlp.BBN)

class Params(params.Params):
    coefs_: typing.List[np.ndarray]
    n_outputs_: int
    n_layers_: int
    intercepts_: typing.List[np.ndarray]
    out_activation_: str
    classes_: np.ndarray
    _label_binarizer: LabelBinarizer
    _target_names: Optional[Sequence[Any]]

class Hyperparams(hyperparams.Hyperparams):
    hidden_layer_sizes = hyperparams.Hyperparameter[List](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ],
        default=List([30,30]),
    )
    activation = hyperparams.Enumeration[str](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ],
        values=['identity', 'logistic', 'tanh','relu'],
        default='relu',
        description='Activation function for the hidden layer.'
    )
    solver = hyperparams.Enumeration[str](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ],
        values=['lbfgs', 'sgd', 'adam'],
        default='adam',
        description=''
    )
    learning_rate = hyperparams.Enumeration[str](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ],
        values=['constant', 'invscaling', 'adaptive'],
        default='constant',
        description=''
    )
    alpha = hyperparams.Hyperparameter[float](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ],
        default=0.0001,
        description=''
    )
    beta_1 = hyperparams.Hyperparameter[float](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ],
        default=0.9,
        description=''
    )
    beta_2 = hyperparams.Hyperparameter[float](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ],
        default=0.999,
        description=''
    )
    learning_rate_init = hyperparams.Hyperparameter[float](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ],
        default=0.001,
    )
    tol = hyperparams.Hyperparameter[float](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ],
        default=0.0001,
    )
    max_iter = hyperparams.UniformInt(
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/ControlParameter',
          'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
        ],
        default=200,
        lower=1,
        upper=800,
        description='The maximum number of passes over the training data  '
    )
    early_stopping = hyperparams.Hyperparameter[bool](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        ],
        default=False,
    )
    shuffle = hyperparams.Hyperparameter[bool](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        ],
        default=True,
        description='Whether or not the training data should be shuffled after each iteration. '
    )
    warm_start = hyperparams.Hyperparameter[bool](
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        ],
        default=False,
        description='When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. '
    )
    #hidden_layer_sizes=hyperparams.Hyperparameter[Tuple[int, int]](
    #    default=(100,1)
    #)
    epsilon = hyperparams.LogUniform(
        semantic_types = [
          'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ],
        default=1e-8,
        lower=1e-08,
        upper=0.1,
        description='Value for numerical stability in adam. Only used when solver=’adam’'
    )

    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned?",
    )
    use_semantic_types = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe."
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )

class BBNMLPClassifier(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):

    """
    BBN D3M BBN MLP Classifier Primitive is wrapper of sklearn MLPClassfier.

    Arguments:
    Input: dataframe
    Output: dataframe
    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': 'cdb18166-e2ca-4418-b5a4-fffbe98f7844',
        'version': __version__,
        'name': "BBN MLP Classifier",
        'description': "BBN D3M BBN MLP Classifier Primitive is wrapper of sklearn MLPClassfier.",
        'keywords': [],
        'source': {
            'name': __author__,
            'contact':'mailto:prasannakumar.muthukumar@raytheon.com',
            'uris': [
                'https://github.com/BBN-E/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/sklearn_wrap/mlp_classifier.py'.format(
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
        'python_path': 'd3m.primitives.classification.mlp.BBN', #'d3m.primitives.bbn.sklearn_wrap.BBNMLPClassifier', #'d3m.primitives.classification.mlp_classifier.BBN',
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.MODULAR_NEURAL_NETWORK], # TODO: replace by a new algorithm_type, e.g. ?
        'primitive_family': metadata_module.PrimitiveFamily.CLASSIFICATION,
    })

    def __init__(
        self, *, hyperparams: Hyperparams, random_seed: int = 0,
        docker_containers: typing.Dict[str, DockerContainer] = None
    ) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._model = MLPClassifier(activation=self.hyperparams['activation'], alpha=self.hyperparams['alpha'], batch_size='auto',
                        beta_1=self.hyperparams['beta_1'], beta_2=self.hyperparams['beta_2'],
                        epsilon=self.hyperparams['epsilon'], hidden_layer_sizes=self.hyperparams['hidden_layer_sizes'], learning_rate=self.hyperparams['learning_rate'],
                        learning_rate_init=self.hyperparams['learning_rate_init'], max_iter=200, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=self.hyperparams['shuffle'],
                        solver=self.hyperparams['solver'], tol=self.hyperparams['tol'], validation_fraction=0.1, verbose=False,
                        warm_start=self.hyperparams['warm_start'], early_stopping=self.hyperparams['early_stopping'])
        self._fitted: bool = False
        return


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs, self._training_indices = self._get_columns_to_fit(inputs, self.hyperparams)
        self._training_outputs, self._target_names = self._get_targets(outputs, self.hyperparams)
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None or self._training_outputs is None:
            raise ValueError("Missing training data.")
        sk_training_output = d3m_ndarray(self._training_outputs)

        with stopit.ThreadingTimeout(timeout) as timer:
            shape = sk_training_output.shape
            if len(shape) == 2 and shape[1] == 1:
                sk_training_output = np.ravel(sk_training_output)

            self._model.fit(self._training_inputs, sk_training_output)
            self._fitted = True

        if timer.state == timer.EXECUTED:
            return CallResult(None)
        else:
            raise TimeoutError('BBNMLPClassifier exceeded time limit')

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        with stopit.ThreadingTimeout(timeout) as timer:
            sk_inputs = inputs
            if self.hyperparams['use_semantic_types']:
                sk_inputs = inputs.iloc[:, self._training_indices]
            sk_output = self._model.predict(sk_inputs)
            if sparse.issparse(sk_output):
                sk_output = sk_output.toarray()
            output = d3m_dataframe(sk_output, columns=self._target_names if self._target_names else None,
                                    generate_metadata=False)
            output.metadata = inputs.metadata.clear( for_value=output, generate_metadata=True)
            output.metadata = self._add_target_semantic_types(metadata=output.metadata, target_names=self._target_names, source=self)
            outputs = common_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                                add_index_columns=self.hyperparams['add_index_columns'],
                                                inputs=inputs, column_indices=[], columns_list=[output])

        if timer.state == timer.EXECUTED:
            return CallResult(outputs)
        else:
            raise TimeoutError('BBNMLPClassifier exceeded time limit')

    def get_params(self) -> Params:
        return Params(
                     coefs_ = self._model.coefs_,
                     n_outputs_ = self._model.n_outputs_,
                     n_layers_ = self._model.n_layers_,
                     intercepts_ = self._model.intercepts_,
                     out_activation_ = self._model.out_activation_,
                     classes_ = self._model.classes_,
                     _label_binarizer = self._model._label_binarizer,
                     _target_names = self._target_names
                )

    def set_params(self, *, params: Params) -> None:
        self._model.coefs_ = params['coefs_']
        self._model.n_outputs_ = params['n_outputs_']
        self._model.n_layers_ = params['n_layers_']
        self._model.intercepts_ = params['intercepts_']
        self._model.out_activation_ = params['out_activation_']
        self._model.classes_ = params['classes_']
        self._model._label_binarizer = params['_label_binarizer']
        self._target_names = params['_target_names']

    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams):
        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = common_utils.get_columns_to_use(inputs_metadata,
                                                                             use_columns=hyperparams['use_columns'],
                                                                             exclude_columns=hyperparams['exclude_columns'],
                                                                             can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce

    @classmethod
    def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int, hyperparams: Hyperparams) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        accepted_structural_types = [int, float, np.int64, np.float64]
        if column_metadata['structural_type'] not in accepted_structural_types:
            return False

        semantic_types = column_metadata.get('semantic_types', [])
        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        if "https://metadata.datadrivendiscovery.org/types/Attribute" in semantic_types:
            return True

        return False

    @classmethod
    def _get_targets(cls, data: d3m_dataframe, hyperparams: Hyperparams):
        metadata = data.metadata
        if not hyperparams['use_semantic_types']:
            target_names = [ metadata.query((metadata_base.ALL_ELEMENTS, c))['name']
                  for c in range(len(data.columns))
                  if c is not metadata_base.ALL_ELEMENTS ]
            return data, target_names
        target_names = []
        target_column_indices = []
        target_column_indices.extend(metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        target_column_indices.extend(metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/RedactedTarget'))
        target_column_indices.extend(
            metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget'))
        target_column_indices = list(set(target_column_indices))
        for column_index in target_column_indices:
            if column_index is metadata_base.ALL_ELEMENTS:
                continue
            column_index = typing.cast(metadata_base.SimpleSelectorSegment, column_index)
            column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            target_names.append(column_metadata.get('name', str(column_index)))

        targets = data.iloc[:, target_column_indices]
        return targets, target_names

    @classmethod
    def _add_target_semantic_types(cls, metadata: metadata_base.DataMetadata,
                            source: typing.Any,  target_names: typing.List = None,) -> metadata_base.DataMetadata:
        for column_index in range(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']):
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/Target',
                                                  )
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/PredictedTarget',
                                                  )
            if target_names:
                metadata = metadata.update((metadata_base.ALL_ELEMENTS, column_index), {
                    'name': target_names[column_index],
                })
        return metadata

