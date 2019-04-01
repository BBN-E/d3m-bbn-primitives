import typing

import numpy as np
import stopit
import sys, os
import pandas as pd
import logging
import re

from d3m.container import ndarray as d3m_ndarray, Dataset, List
from d3m.metadata import hyperparams, params
from d3m import utils
from d3m.metadata import base as metadata_module
from d3m.primitive_interfaces.base import CallResult, DockerContainer
#from d3m.container import ndarray as d3m_ndarray, Dataset, List
#from d3m.metadata import hyperparams, params, utils
#from d3m import container
#from d3m.metadata import base as metadata_modul
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase

from . import __author__, __version__

Inputs = Dataset
Outputs = List[d3m_ndarray]


_logger = logging.getLogger('d3m.primitives.bbn.time_series.CSVReader')

class Hyperparams(hyperparams.Hyperparams):
    resampling_rate = hyperparams.Bounded[float](
        default = 1.0,
        lower = 0.0, upper = None,
        description = 'Resampling rate'
    )
    read_as_mono = hyperparams.Hyperparameter[bool](
        default = True,
        #_structural_type = bool,
        description = 'Read csv'
    )

class CSVReader(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    """
    BBN D3M CSV Reader Primitive

    Arguments:
    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': '503e69a1-5fc4-4f14-912a-4b564cb1b171',
        'version': __version__,
        'name': "CSV Reader",
        'description': "BBN D3M CSV Reader Primitive.",
        'keywords': [],
        'source': {
            'name': __author__,
            'uris': [
                'https://gitlab.datadrivendiscovery.org/BBN/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/csv_reader.py'.format(
                    git_commit=__git_commit__
                ),
                'https://gitlab.datadrivendiscovery.org/BBN/d3m-bbn-primitives.git',
            ],
        },
        'installation': [{
                'type': 'UBUNTU',
                'package': 'ffmpeg',
                'version': '7:2.8.11-0',
            },
            {
                'type': 'PIP',
                'package_uri': 'git+https://gitlab.datadrivendiscovery.org/BBN/d3m-bbn-primitives.git@{git_commit}#egg={egg}'.format(
                    git_commit=__git_commit__, egg='bbn_primitives'
            ),
        }],
        'python_path': 'd3m.primitives.bbn.time_series.CSVReader',
        'algorithm_types': ['DATA_CONVERSION'], # TODO: replace by a new algorithm_type, e.g. ?
        'primitive_family': 'DATA_PREPROCESSING',
    })

    def __init__(
        self, *, hyperparams: Hyperparams, random_seed: int = 0,
        docker_containers: typing.Dict[str, DockerContainer] = None
    ) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._metadata_lookup = None
        return


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Arguments:
            - inputs: [ num_samples, num_channels ]

        Returns:
            - [ num_samples ]
        """

        with stopit.ThreadingTimeout(timeout) as timer:
            metadata_lookup = self.__class__._parse_metadata(
                                metadata = inputs.metadata)
            if not metadata_lookup:
                return None

            outputs = Outputs()
            metadata = self.__class__._can_accept(self = self,
                          method_name = 'produce',
                          arguments = { 'inputs': inputs.metadata, },
                          outputs = outputs
                        )

            csv_location_base_uris = inputs.metadata.query(metadata_lookup['location_base_uris']['selector'])['location_base_uris'][0]
            for idx, row in inputs[metadata_lookup['primary_resource_id']['selector'][0]].iterrows():
            #for idx in range(len(inputs[metadata_lookup['primary_resource_id']['selector'][0]])):
                #row = inputs[metadata_lookup['primary_resource_id']['selector'][0]][idx]
                #d3mIndex = row[metadata_lookup['primary_key']['selector'][-1]]
                d3mIndex=row['d3mIndex']
                csv_fn = row[metadata_lookup['csv_fn']['selector'][-1]]

                filename = os.path.join(csv_location_base_uris, csv_fn)
                filename = re.sub('^file://', '', filename)
                print(filename)
                #csv_file= csv.load(filename)
                csv_file=pd.read_csv(filename,index_col=0)
                start = 0
                end = len(csv_file)

                outputs.append(csv_file)
                metadata = metadata.update((idx,), { 'sampling_rate': 1 })

            metadata = metadata.update((), { 'dimension': { 'length': len(outputs) } })
            # Set metadata attribute.
            outputs.metadata = metadata

        if timer.state == timer.EXECUTED:
            return base.CallResult(outputs)
        else:
            raise TimeoutError('Reader exceeded time limit')

    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]]) -> typing.Optional[metadata_module.DataMetadata]:
        output_metadata = super().can_accept(method_name=method_name, arguments=arguments)

        return cls._can_accept(self = cls, method_name = method_name,
                                arguments = arguments, outputs = Outputs())

    @classmethod
    def _can_accept(cls, *, self, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], outputs: Outputs) -> typing.Optional[metadata_module.DataMetadata]:
        output_metadata = super().can_accept(method_name=method_name, arguments=arguments)

        if 'inputs' not in arguments:
            return output_metadata

        inputs_metadata = typing.cast(metadata_module.DataMetadata, arguments['inputs'])

        metadata_lookup = cls._parse_metadata(metadata = inputs_metadata)

        num_data = inputs_metadata.query(metadata_lookup['primary_resource_id']['selector'])['dimension']['length']

        metadata = inputs_metadata.clear({
            'schema': metadata_module.CONTAINER_SCHEMA_VERSION,
            'structural_type': Outputs,
            'dimension': {
                'length': num_data,
            }
        }, for_value=outputs, source=self).update((metadata_module.ALL_ELEMENTS,), {
            'structural_type': d3m_ndarray,
            'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Timeseries', )
        }, source=self)

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
        metadata_lookup['csv_fn'] = {
            'required': True, 'found': False, 'selector': None,
        }
        metadata_lookup['location_base_uris'] = {
            'required': True, 'found': False, 'selector': None,
        }

        return metadata_lookup

    @classmethod
    def _parse_metadata(cls, *, metadata: metadata_module.DataMetadata):
        flatten = lambda l: [item for sublist in l for item in sublist]

        mdlu = cls._init_metadata_lookup()

        num_res = metadata.query(())['dimension']['length']
        resources = [ str(x) for x in range(num_res) ]
        primary_key = [ [ (res_id, metadata_module.ALL_ELEMENTS, col_id) for col_id in range(metadata.query((res_id, metadata_module.ALL_ELEMENTS))['dimension']['length'])
                                      if 'd3mIndex' == metadata.query((res_id, metadata_module.ALL_ELEMENTS, col_id))['name'] ]
                                   for res_id in resources ]
        primary_key = flatten(primary_key)
        if len(primary_key) != 1:
            raise Exception('One primary key supported')
        cls._update_metadata_lookup(mdlu, 'primary_key', primary_key[0])
        cls._update_metadata_lookup(mdlu, 'primary_resource_id', (primary_key[0][0], ))

        csv_res_type = 'https://metadata.datadrivendiscovery.org/types/Timeseries'
        primary_resource_cols = metadata.query((mdlu['primary_resource_id']['selector'][0], metadata_module.ALL_ELEMENTS))
        for col_id in range(primary_resource_cols['dimension']['length']):
            cmd = metadata.query((mdlu['primary_resource_id']['selector'][0], metadata_module.ALL_ELEMENTS, col_id))
            if 'semantic_types' in cmd:
                st = cmd['semantic_types']
                if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in st:
                    # we already found primary key
                    pass
                elif 'https://metadata.datadrivendiscovery.org/types/Attribute' in st:
                    if 'foreign_key' in cmd and cmd['foreign_key']['type'] == 'COLUMN':
                        foreign_resource_id = cmd['foreign_key']['resource_id']
                        foreign_resource_md = metadata.query((foreign_resource_id,))
                        foreign_col_selector = (foreign_resource_id, metadata_module.ALL_ELEMENTS, cmd['foreign_key']['column_index'])
                        foreign_col_md = metadata.query(foreign_col_selector)
                        if csv_res_type in foreign_resource_md['semantic_types'] and \
                            'https://metadata.datadrivendiscovery.org/types/FileName' in foreign_col_md['semantic_types']:
                                cls._update_metadata_lookup(mdlu,
                                    'csv_fn',
                                    (mdlu['primary_resource_id']['selector'][0], metadata_module.ALL_ELEMENTS, col_id)
                                  )
                                cls._update_metadata_lookup(mdlu,
                                    'location_base_uris',
                                     foreign_col_selector
                                  )
                        else:
                            _logger.warning('Expected foreign resource of type %s and column of semantic type Filename' % (csv_res_type))
                    else:
                        _logger.warning('Unexpected semantic type Attribute')
                elif 'https://metadata.datadrivendiscovery.org/types/InstanceWeight' in st:
                    _logger.warning('Semantic type InstanceWeight recognized but unused in the current implementation')
                elif 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in st:
                    _logger.info('Semantic type SuggestedTarget is ignored by this primitive')
                else:
                    raise Exception('Semantic type(s) %s does not match any supported types' % (st))

       
        return mdlu if cls._valid_metadata_lookup(mdlu) else None


