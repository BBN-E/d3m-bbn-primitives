import numpy as np
import stopit
import sys

from typing import NamedTuple

import scipy.cluster
import sklearn.svm
import logging

from .time_series_common import *
from .segmentation_common import *
from .signal_framing import SignalFramer

from d3m.container import ndarray as d3m_ndarray, Dataset, List
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

Inputs = List[List[CurveFitting]]
Outputs = List[d3m_ndarray]
Params = NamedTuple('Params', [
          ('coefficient', d3m_ndarray),
        ])

_logger = logging.getLogger('d3m.primitives.bbn.time_series.ClusterCurveFitting')

class ClusterCurveFitting(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params]):

    """
    BBN D3M Curve Fitting Clusterer

    For details, refer to Gish, H. and Ng, K., 1996, October. Parametric trajectory models for speech recognition. In Spoken Language, 1996. ICSLP 96. Proceedings., Fourth International Conference on (Vol. 1, pp. 466-469). IEEE.
    or Yeung, S.K.A., Li, C.F. and Siu, M.H., 2005, March. Sub-phonetic polynomial segment model for large vocabulary continuous speech recognition. In Acoustics, Speech, and Signal Processing, 2005. Proceedings.(ICASSP'05). IEEE International Conference on (Vol. 1, pp. I-193). IEEE.

    Arguments:
        hp_seed ... 
    """

    __author__ = 'BBN'
    __metadata__ = {
            "common_name": "Discontinuity-based segmentation",
            "algorithm_type": ["Clustering"],
            "original_name": "bbn_primitives.time_series.ClusterCurveFitting",
            "task_type": ["Modeling"],
            "learning_type": ["Unsupervised learning"],
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
            "handles_classification": True,
            "handles_multiclass": False,
            "handles_multilabel": False,
          }

    def __init__(
        self,
        hp_seed = 0,
    ):
        super().__init__()

        np.random.seed(hp_seed)

        self.hp_linkageMethod = 'average'
        #self.hp_clustCriterion = 'distance'
        #self.hp_clustThresh = 0.5
        self.hp_clustCriterion = 'maxclust'
        self.hp_clustThresh = 32

        self.training_inputs = None
        self.model = None
        self.fitted = False

    def set_training_data(self, *, inputs: Inputs) -> None:
        self.training_inputs = inputs
        self.model = None
        self.fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        """
        Arguments
            - inputs: List( # Data
                         List( # Segments
                           CurveFitting
                         )
                       ),
        """
        if self.fitted:
            return

        if self.training_inputs is None:
            raise Exception('Missing training data')

        with stopit.ThreadingTimeout(timeout) as timer:
            inputs_curve_fitting = self.training_inputs
            num_data = sum([ len(x) for x in inputs_curve_fitting ]) # number of segments, each segment if formed by multiple data samples
            deg, num_feats = inputs_curve_fitting[0][0].beta.shape
            betas = np.vstack([
                        np.array([
                            segment.beta for segment in cinput
                        ]) for cinput in inputs_curve_fitting if len(cinput) > 0
                    ])
            sigmas = np.vstack([
                        np.array([
                            segment.sigma for segment in cinput
                        ]) for cinput in inputs_curve_fitting if len(cinput) > 0
                    ])
            Ns = np.hstack([
                      np.array([
                          segment.N for segment in cinput
                      ]) for cinput in inputs_curve_fitting if len(cinput) > 0
                 ])

            # Compute diff_mat
            # TODO: review if computation of synth_data, cbeta, csynth_data is necessary
            #       or if this can be expressed in a different, more efficient form
            synth_data = List[d3m_ndarray]()
            for i in range(num_data):
                synth_data.append(applyFitting(Ns[i], betas[i]))

            diff_mat = np.zeros((num_data, num_data))
            Ws = np.multiply(sigmas.T, Ns).T
            for i in range(num_data):
                for j in range(i):
                    #print('%d, %d / %d' % (i, j, num_data))
                    cN = Ns[i] + Ns[j]
                    cW = (Ns[i] * Ws[i] + Ns[j] * Ws[j]) / cN
                    cbeta = (Ns[i] * betas[i] + Ns[j] * betas[j]) / cN
                    csynth_data = applyFitting(Ns[i], cbeta)
                    Ei = synth_data[i]-csynth_data
                    csynth_data = applyFitting(Ns[j], cbeta)
                    Ej = synth_data[j]-csynth_data
                    D = (np.dot(Ei.T, Ei) + np.dot(Ej.T, Ej)) / cN
                    lambda_traj = cN/2*np.log(np.linalg.det(
                                    np.eye(num_feats) + np.dot(np.linalg.inv(cW), D)
                                  ))
                    diff_mat[i, j] = lambda_traj
                    diff_mat[j, i] = lambda_traj

            _logger.info('DEBUG: %s' % np.sum(diff_mat < 0.0))

            # Hierarchical clustering
            clust_thresh = 0.5
            diff_sqf = scipy.spatial.distance.squareform(diff_mat,checks=False)
            seg_link = scipy.cluster.hierarchy.linkage(diff_sqf, method=self.hp_linkageMethod)
            seg_clust = scipy.cluster.hierarchy.fcluster(seg_link, self.hp_clustThresh,
                                criterion=self.hp_clustCriterion)

            # Build a classification model
            svm = sklearn.svm.SVC(probability = True)
            svm.fit(betas, seg_clust)

            self.model = svm
            self.fitted = True

        if timer.state == timer.EXECUTED:
            return
        else:
            raise TimeoutError('ClusterCurveFitting exceeded time limit')

    def produce(self, inputs: Inputs, timeout=None, iterations=None) -> Outputs:
        """
        Arguments:
            - inputs: List( # Data
                        List( # Segments
                          CurveFitting
                        )
                      )


        Returns:
            - List(d3m_ndarray)
        """
        with stopit.ThreadingTimeout(timeout) as timer:
            outputs = List()
            for cinput in inputs:
                deg, num_feats = cinput[0].beta.shape
                betas = np.array([ segment.beta for segment in cinput ]).reshape((-1, num_feats))
                outputs.append(self.model.predict(betas))

        if timer.state == timer.EXECUTED:
            return d3m_ndarray(outputs)
        else:
            raise TimeoutError('ClusterCurveFitting exceeded time limit')


    def get_params(self) -> Params:
        return Params(coefficient=self.model.coef_)

    def set_params(self, *, params: Params) -> None:
        self.model.coef_ = params.coefficient

