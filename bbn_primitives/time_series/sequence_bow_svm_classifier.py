import numpy as np
import stopit
import sys

from typing import NamedTuple

import sklearn.svm
from sklearn.feature_extraction.text import TfidfTransformer

from .time_series_common import *
from .segmentation_common import *
from .signal_framing import SignalFramer

from d3m_types.sequence import ndarray as d3m_ndarray
from d3m_types.sequence import List
from primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

Inputs = List[d3m_ndarray]
Outputs = List[int]
Params = NamedTuple('Params', [
          ('coefficient', d3m_ndarray),
        ])

class SequenceBagOfWordsSVMClassifier(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params]):

    """
    BBN D3M Naive Sequence Classifier

    Arguments:
        hp_seed ... 
        hp_splice ... 
    """

    __author__ = 'BBN'
    __metadata__ = {
            "common_name": "Discontinuity-based segmentation",
            "algorithm_type": ["Bayesian"],
            "original_name": "bbn_primitives.time_series.SequenceBagOfWordsSVMClassifier",
            "task_type": ["Modeling"],
            "learning_type": ["Supervised learning"],
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
        hp_seed: int = 0,
        hp_splice: int = 0,
    ):
        super().__init__()

        np.random.seed(hp_seed)

        self.hp_splice = hp_splice

        self.training_inputs = None
        self.training_outputs = None
        self.tfidf = None
        self.vocab = None
        self.model = None
        self.fitted = False

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self.training_inputs = inputs
        self.training_outputs = outputs
        self.fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        """
        Arguments
            - inputs: List(d3m_ndarray)
        """
        if self.fitted:
            return

        if self.training_inputs is None or self.training_outputs is None:
            raise Exception('Missing training data')

        with stopit.ThreadingTimeout(timeout) as timer:
#            if self.hp_splice > 1:
#                spliced_data = list()
#                for cinput in self.training_inputs:
#                    framer = SignalFramer(
#                        sampling_rate = 1,
#                        frame_length_s = self.hp_splice,
#                        frame_shift_s = 1,
#                        flatten_output = True,
#                    )
#                    cdata = frames.produce([cinput])[0]
#                    spliced_data.append(cdata)
#            else:
#                spliced_data = self.training_inputs
            self.vocab = seq_vocab(self.training_inputs)
            train_x = seq_to_tokenfreq_csr(self.training_inputs, self.vocab)
            train_y = self.training_outputs

            self.tfidf = TfidfTransformer(norm='l1')
            train_x_tfid = self.tfidf.fit(train_x).transform(train_x)

            # Build a classification model
            svm = sklearn.svm.SVC(probability = True)
            svm.fit(train_x_tfid, train_y)

            self.model = svm
            self.fitted = True

        if timer.state == timer.EXECUTED:
            return
        else:
            raise TimeoutError('SequenceBagOfWordsSVMClassifier exceeded time limit')

    def produce(self, inputs: Inputs, timeout: float = None, iterations: int = None) -> Outputs:
        """
        Arguments:
            - inputs: List(d3m_ndarray)

        Returns:
            - List(int)
        """
        with stopit.ThreadingTimeout(timeout) as timer:
            x = seq_to_tokenfreq_csr(inputs, self.vocab)
            train_x_tfid = self.tfidf.transform(x)
            pred = self.model.predict(train_x_tfid)
            outputs = [ cpred for cpred in pred ]

        if timer.state == timer.EXECUTED:
            return outputs
        else:
            raise TimeoutError('SequenceBagOfWordsSVMClassifier exceeded time limit')

    def get_params(self) -> Params:
        return Params(coefficient=self.model.coef_)

    def set_params(self, *, params: Params) -> None:
        self.model.coef_ = params.coefficient

