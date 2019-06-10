import typing

import numpy as np
import stopit
import sys, os
import subprocess as sp
import logging
import re

from d3m.container import ndarray as d3m_ndarray, Dataset, List
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params
from d3m import utils
from d3m.metadata import base as metadata_module
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase


from . import __author__, __version__

DEVNULL = open(os.devnull, 'w')

Inputs = Dataset
Outputs = d3m_dataframe

#_logger = logging.getLogger(TargetsReader.metadata.query()['python_path'])
_logger = logging.getLogger('d3m.primitives.data_preprocessing.targets_reader.BBN')

class Hyperparams(hyperparams.Hyperparams):
    pass

class TargetsReader(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    """
    BBN D3M Targets Reader Primitive reads the targets from a D3M Dataset

    Arguments:
    Input: Dataset
    Output: DataFrame
    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': '952b659c-d290-465b-9e89-160947e29c06',
        'version': __version__,
        'name': "Targets Reader",
        'description': "BBN D3M Targets Reader Primitive reads the targets from a D3M Dataset",
        'keywords': [],
        'source': {
            'name': __author__,
            'contact':'mailto:prasannakumar.muthukumar@raytheon.com',
            'uris': [
                'https://github.com/BBN-E/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/targets_reader.py'.format(
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
        'python_path': 'd3m.primitives.data_preprocessing.targets_reader.BBN',#'d3m.primitives.bbn.time_series.TargetsReader', #'d3m.primitives.data_preprocessing.targets_reader.BBN',
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.DATA_CONVERSION],
        'primitive_family': metadata_module.PrimitiveFamily.DATA_PREPROCESSING,
    })

    def __init__(
        self, *, hyperparams: Hyperparams, random_seed: int = 0,
        docker_containers: typing.Dict[str, DockerContainer] = None
    ) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._metadata_lookup = None
        return


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Arguments:
            - inputs: Dataset

        Returns:
            - [ num_samples, num_channels ]
        """

        with stopit.ThreadingTimeout(timeout) as timer:
            metadata_lookup = self.__class__._parse_metadata(
                                metadata = inputs.metadata)
            if not metadata_lookup:
                return None

            primary_key_name = inputs.metadata.query(metadata_lookup['primary_key']['selector'])['name']
            targets_name = inputs.metadata.query(metadata_lookup['targets']['selector'])['name']
            #outputs = d3m_dataframe(inputs[metadata_lookup['targets']['selector'][0]][metadata_lookup['targets']['selector'][-1]])
            outputs = d3m_dataframe({
                targets_name:
                    inputs[metadata_lookup['targets']['selector'][0]][targets_name]
            })
            outputs.metadata = outputs.metadata.update(
              (metadata_module.ALL_ELEMENTS, 0),
                inputs.metadata.query(metadata_lookup['targets']['selector'])
            )

        if timer.state == timer.EXECUTED:
            return CallResult(outputs)
        else:
            raise TimeoutError('TargetsReader exceeded time limit')

    @classmethod
    def can_accept(cls, *, method_name: str,
    		   arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]],
           hyperparams: Hyperparams) -> typing.Optional[metadata_module.DataMetadata]:

        output_metadata = super().can_accept(method_name=method_name, arguments=arguments,
        hyperparams=hyperparams)

        return cls._can_accept(self = cls, method_name = method_name,
                               arguments = arguments, hyperparams = hyperparams,
                               outputs = None)

    @classmethod
    def _can_accept(cls, *, self, method_name: str,
    		    arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]],
            hyperparams: Hyperparams, outputs: Outputs) -> typing.Optional[metadata_module.DataMetadata]:
        output_metadata = super().can_accept(method_name=method_name, arguments=arguments,
						hyperparams=hyperparams)

        if 'inputs' not in arguments:
            return output_metadata

        inputs_metadata = typing.cast(metadata_module.DataMetadata, arguments['inputs'])

        metadata_lookup = cls._parse_metadata(metadata = inputs_metadata)
        #try:
        #    cls._parse_metadata(metadata = inputs_metadata)
        #except:
        #    return None

        num_data = inputs_metadata.query(metadata_lookup['primary_resource_id']['selector'])['dimension']['length']

        outputs = d3m_dataframe(data={})
        metadata = outputs.metadata.update(
          (metadata_module.ALL_ELEMENTS, 0),
            inputs_metadata.query(metadata_lookup['targets']['selector'])
        )

        return metadata

    @classmethod
    def _update_metadata_lookup(cls, metadata_lookup, key, selector):
        if key not in metadata_lookup:
            raise Exception('Updating unknown key %s' % key)

        metadata_lookup[key]['found'] = True
        metadata_lookup[key]['selector'] = selector

    @classmethod
    def _valid_metadata_lookup(cls, metadata_lookup):
        for k in metadata_lookup.keys():
            if metadata_lookup[k]['required'] and not metadata_lookup[k]['found']:
                return False
        return True

    @classmethod
    def _init_metadata_lookup(cls):
        metadata_lookup = dict()
        metadata_lookup['primary_key'] = {
            'required': True, 'found': False, 'selector': None,
        }
        metadata_lookup['primary_resource_id'] = {
            'required': True, 'found': False, 'selector': None,
        }
        metadata_lookup['targets'] = {
            'required': True, 'found': False, 'selector': None,
        }

        return metadata_lookup

    @classmethod
    def _parse_metadata(cls, *, metadata: metadata_module.DataMetadata):
        flatten = lambda l: [item for sublist in l for item in sublist]

        mdlu = cls._init_metadata_lookup()

        num_res = metadata.query(())['dimension']['length']
        resources = [ str(x) for x in range(num_res-1) ]
        resources.append('learningData')
        #primary_key = [ [ (res_id, metadata_module.ALL_ELEMENTS, col_id) for x in range(metadata.query((res_id, metadata_module.ALL_ELEMENTS))['dimension']['length'])
        #                              if 'semantic_types' in metadata.query((res_id, metadata_module.ALL_ELEMENTS, col_id)) and primary_sem_type in metadata.query((res_id, metadata_module.ALL_ELEMENTS, col_id))['semantic_types'] ]
        #                           for res_id in resources ]
        primary_key = [ [ (res_id, metadata_module.ALL_ELEMENTS, col_id) for col_id in range(metadata.query((res_id, metadata_module.ALL_ELEMENTS))['dimension']['length'])
                                      if 'd3mIndex' == metadata.query((res_id, metadata_module.ALL_ELEMENTS, col_id))['name'] ]
                                   for res_id in resources ]
        primary_key = flatten(primary_key)
        if len(primary_key) != 1:
            raise Exception('One primary key supported')
        cls._update_metadata_lookup(mdlu, 'primary_key', primary_key[0])
        cls._update_metadata_lookup(mdlu, 'primary_resource_id', (primary_key[0][0], ))

        primary_resource_cols = metadata.query((mdlu['primary_resource_id']['selector'][0], metadata_module.ALL_ELEMENTS))
        for col_id in range(primary_resource_cols['dimension']['length']):
            cmd = metadata.query((mdlu['primary_resource_id']['selector'][0], metadata_module.ALL_ELEMENTS, col_id))
            col_name = cmd['name']
            if 'semantic_types' in cmd:
                st = cmd['semantic_types']
                if 'https://metadata.datadrivendiscovery.org/types/TrueTarget' in st:
                    cls._update_metadata_lookup(mdlu,
                        'targets',
                        #(mdlu['primary_resource_id']['selector'][0], metadata_module.ALL_ELEMENTS, col_name)
                        (mdlu['primary_resource_id']['selector'][0], metadata_module.ALL_ELEMENTS, col_id)
                        )

        return mdlu if cls._valid_metadata_lookup(mdlu) else None

