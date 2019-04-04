import typing

import numpy as np
import scipy.stats
import stopit
import sys, os

import sklearn.svm
import sklearn.naive_bayes
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from .time_series_common import *
from .segmentation_common import *
from .signal_framing import SignalFramer

from d3m.container import ndarray as d3m_ndarray, Dataset, List
from d3m.metadata import hyperparams, params, utils
from d3m import container
from d3m.metadata import base as metadata_modul
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase


from . import __author__, __version__

Inputs = List[d3m_ndarray]
Outputs = List[int]

class Params(params.Params):
    feature_log_prob: d3m_ndarray
    classes: d3m_ndarray
    class_log_prior: d3m_ndarray
    vocab: dict

class Hyperparams(hyperparams.Hyperparams):
    alpha = hyperparams.Bounded(
        default = 1.0,
        lower = 0.0, upper = 1.0,
        description = 'Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing)'
    )
    fit_prior = hyperparams.Hyperparameter(
        default = True,
        _structural_type = bool,
        description = 'Whether to learn class prior probabilities or not. If false, a uniform prior will be used.'
    )

class SequenceBagOfWordsClassifier(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):

    """
    BBN D3M Naive Sequence Classifier
    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': '659b3e83-7af4-4450-99a2-e214a911b0a1',
        'version': __version__,
        'name': "Naive Sequence Classifier",
        'description': "BBN D3M Naive Sequence Classifier classifies bag of words,features calculated from the Bag-of-words model is term frequency",
        'keywords': [],
        'source': {
            'name': __author__,
            'uris': [
                'https://gitlab.datadrivendiscovery.org/BBN/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/sequence_bow_classifier.py'.format(
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
        'python_path': 'd3m.primitives.bbn.time_series.SequenceBagOfWordsClassifier',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': ['MULTINOMIAL_NAIVE_BAYES'],
        'primitive_family': 'TIMESERIES_CLASSIFICATION',
    })

    def __init__(
        self, *, hyperparams: Hyperparams, random_seed: int = 0,
        docker_containers: typing.Dict[str, str] = None
    ) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._training_inputs: Inputs = None
        self._training_outputs: Outputs = None
        #self._tfidf = False
        self._vocab: dict = None
        self._model: MultinomialNB = MultinomialNB(
													alpha = self.hyperparams['alpha'],
													fit_prior = self.hyperparams['fit_prior']
											)
        self._fitted: bool = False

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        """
        Arguments
            - inputs: List(d3m_ndarray)
        """
        if self._fitted:
            return base.CallResult(None)

        if self._training_inputs is None or self._training_outputs is None:
            raise Exception('Missing training data')

        with stopit.ThreadingTimeout(timeout) as timer:
#            if self.hp_splice > 1:
#                spliced_data = list()
#                for cinput in self._training_inputs:
#                    framer = SignalFramer(
#                        sampling_rate = 1,
#                        frame_length_s = self.hp_splice,
#                        frame_shift_s = 1,
#                        flatten_output = True,
#                    )
#                    cdata = frames.produce([cinput])[0]
#                    spliced_data.append(cdata)
#            else:
#                spliced_data = self._training_inputs
            self._vocab = seq_vocab(self._training_inputs)
            train_x = seq_to_tokenfreq_csr(self._training_inputs, self._vocab)
            train_y = self._training_outputs

            #self._itfidf = TfidfTransformer(norm='l1')
            #train_x_tfid = self._itfidf.fit(train_x).transform(train_x)
            train_x_tfid = train_x

            # Build a classification model
            self._model.fit(train_x_tfid, train_y)

            self._fitted = True

        if timer.state == timer.EXECUTED:
            return
        else:
            raise TimeoutError('SequenceBagOfWordsClassifier exceeded time limit')

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Arguments:
            - inputs: List(d3m_ndarray)

        Returns:
            - List(int)
        """
        with stopit.ThreadingTimeout(timeout) as timer:
            x = seq_to_tokenfreq_csr(inputs, self._vocab)
            #x_tfid = self._itfidf.transform(x)
            x_tfid = x
            #print(x_tfid.shape)
            pred = self._model.predict(x_tfid)
            outputs = List([ cpred for cpred in pred ])

            metadata = inputs.metadata.clear({
                'schema': metadata_module.CONTAINER_SCHEMA_VERSION,
                'structural_type': type(outputs),
                'dimension': {
                    'length': len(outputs)
                }
            }, for_value=outputs, source=self).update((metadata_module.ALL_ELEMENTS,), {
                'structural_type': int,
            }, source=self)

            # Set metadata attribute.
            outputs.metadata = metadata

        if timer.state == timer.EXECUTED:
            return base.CallResult(outputs)
        else:
            raise TimeoutError('SequenceBagOfWordsClassifier exceeded time limit')


    def get_params(self) -> Params:
        return Params(
                  feature_log_prob = self._model.feature_log_prob_,
                  classes = self._model.classes_,
                  class_log_prior = self._model.class_log_prior,
                  vocab = self._vocab
                )

    def set_params(self, *, params: Params) -> None:
        self._model.feature_log_prob_ = params.feature_log_prob
        self._model.classes_ = params.classes
        self._model.class_log_prior_ = params.class_log_prior
        self._vocab = params.vocab

#    def seq_to_tokenfreq_csr(self, inputs: Inputs):
#        X = list()
#        Y = list()
#        data = list()
#        for i in range(len(inputs)):
#            a = scipy.stats.itemfreq(inputs[i])
#            Y.append(a[:,0])
#            X.append(np.ones_like(a[:,0], dtype=int)*i)
#            data.append(a[:,1])
#
#        X = np.hstack(X)
#        Y = np.hstack(Y)
#        data = np.hstack(data)
#
#        return scipy.sparse.coo_matrix((data, (X,Y)))
