import typing

import numpy as np
import stopit
import sys, os
import subprocess as sp
import logging
import re

from d3m.container import ndarray as d3m_ndarray, Dataset, List
from d3m.metadata import hyperparams, params
from d3m import utils
from d3m.metadata import base as metadata_module
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase


from . import __author__, __version__

DEVNULL = open(os.devnull, 'w')

Inputs = Dataset
Outputs = List #List[d3m_ndarray]

#_logger = logging.getLogger(AudioReader.metadata.query()['python_path'])
_logger = logging.getLogger('d3m.primitives.data_preprocessing.audio_reader.AudioReader') #('d3m.primitives.bbn.time_series.AudioReader')

class Hyperparams(hyperparams.Hyperparams):
    resampling_rate = hyperparams.Bounded[float](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = 16000.0,
        lower = 0.0, upper = None,
        description = 'Resampling rate'
    )
    read_as_mono = hyperparams.Hyperparameter[bool](
	semantic_types = [
		'https://metadata.datadrivendiscovery.org/types/ControlParameter',
	],
        default = True,
        description = 'Read audio as mono'
    )

class AudioReader(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    """
    BBN D3M Audio Reader Primitive reads the audio using ffmpeg via pipe from the Audio input Dataset.
    
    Arguments:
    Input : Dataset Object pointing to raw audio files
    Output: List of arrays with samples of shape [ num_samples, num_channels ]
    Applications include: audio, time-series classification

    """

    __git_commit__=utils.current_git_commit(os.path.dirname(__file__))
    metadata = metadata_module.PrimitiveMetadata({
        'id': '503e69a1-5fc4-4f14-912a-4b564cb1b171',
        'version': __version__,
        'name': "Audio Reader",
        'description': """BBN D3M Audio Reader Primitive reads the audio using ffmpeg via pipe from the Audio input Dataset.\n
                        Input : Dataset Object pointing to raw audio files\n
			Output: List of arrays with samples of shape [ num_samples, num_channels ]\n
			Applications include: audio, time-series classification""",
        'keywords': [],
        'source': {
            'name': __author__,
            'contact':'mailto:prasannakumar.muthukumar@raytheon.com',
            'uris': [
                'https://github.com/BBN-E/d3m-bbn-primitives/blob/{git_commit}/bbn_primitives/time_series/audio_reader.py'.format(
                    git_commit=__git_commit__
                ),
                'https://github.com/BBN-E/d3m-bbn-primitives.git',
            ],
        },
        'installation': [{
                'type': 'UBUNTU',
                'package': 'ffmpeg',
                'version': '7:2.8.11-0',
            },
            {
                'type': 'PIP',
                'package_uri': 'git+https://github.com/BBN-E/d3m-bbn-primitives.git@{git_commit}#egg={egg}'.format(
                    git_commit=__git_commit__, egg='bbn_primitives'
            ),
        }],
        'python_path': 'd3m.primitives.data_preprocessing.audio_reader.AudioReader', #'d3m.primitives.bbn.time_series.AudioReader', #'d3m.primitives.data_preprocessing.audio_reader.BBN',
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.DATA_CONVERSION], #['DATA_CONVERSION'], #  replaced 'AUDIO_MIXING'
        'primitive_family': metadata_module.PrimitiveFamily.DATA_PREPROCESSING, #'DATA_PREPROCESSING',
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

            outputs = Outputs()
            metadata = self.__class__._can_accept(self = self,
                          method_name = 'produce',
                          arguments = { 'inputs': inputs.metadata, },
			  hyperparams = self.hyperparams,
                          outputs = outputs
                        )

            audio_location_base_uris = inputs.metadata.query(metadata_lookup['audio_location_base_uris']['selector'])['location_base_uris'][0]
            for idx, row in inputs[metadata_lookup['primary_resource_id']['selector'][0]].iterrows():
                #row = inputs[metadata_lookup['primary_resource_id']['selector'][0]][idx]
                #d3mIndex = row[metadata_lookup['primary_key']['selector'][-1]]
                d3mIndex = row['d3mIndex']
                audio_fn = row[metadata_lookup['audio_fn']['selector'][-1]]

                filename = os.path.join(audio_location_base_uris, audio_fn)
                filename = re.sub('^file://', '', filename)
                _logger.info('Processing file %s' % filename)
                audio_clip,sampling_rate = ffmpeg_load_audio(filename, mono=self.hyperparams['read_as_mono']);
                start = 0
                end = len(audio_clip)
                if metadata_lookup['audio_start_time']['found']:
                    start = int(sampling_rate * float(row[metadata_lookup['audio_start_time']['selector'][-1]]))
                if metadata_lookup['audio_end_time']['found']:
                    end = int(sampling_rate * float(row[metadata_lookup['audio_end_time']['selector'][-1]]))
                audio_clip = audio_clip[start:end]

                outputs.append(d3m_ndarray(audio_clip))
                metadata = metadata.update((idx,), { 'sampling_rate': sampling_rate })

            metadata = metadata.update((), { 'dimension': { 'length': len(outputs) } })
            # Set metadata attribute.
            outputs.metadata = metadata

        if timer.state == timer.EXECUTED:
            return CallResult(outputs)
        else:
            raise TimeoutError('AudioReader exceeded time limit')

    @classmethod
    def can_accept(cls, *, method_name: str,
    		   arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]],
		   hyperparams: Hyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        output_metadata = super().can_accept(method_name=method_name, arguments=arguments,
						hyperparams=hyperparams)

        return cls._can_accept(self = cls, method_name = method_name,
                                arguments = arguments, hyperparams = hyperparams,
				outputs = Outputs())

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

        metadata = inputs_metadata.clear({
            'schema': metadata_module.CONTAINER_SCHEMA_VERSION,
            'structural_type': Outputs,
            'dimension': {
                'length': num_data,
            }
        }, for_value=outputs).update((metadata_module.ALL_ELEMENTS,), {
            'structural_type': d3m_ndarray,
            'semantic_types': ('http://schema.org/AudioObject', )
        })

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
        metadata_lookup['audio_fn'] = {
            'required': True, 'found': False, 'selector': None,
        }
        #metadata_lookup['audio_fn_resource_id'] = {
        #    required: True, found: False, 'selector': None,
        #}
        metadata_lookup['audio_start_time'] = {
            'required': False, 'found': False, 'selector': None,
        }
        metadata_lookup['audio_end_time'] = {
            'required': False, 'found': False, 'selector': None,
        }
        metadata_lookup['audio_location_base_uris'] = {
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

        audio_res_type = 'https://metadata.datadrivendiscovery.org/types/FilesCollection'
        primary_resource_cols = metadata.query((mdlu['primary_resource_id']['selector'][0], metadata_module.ALL_ELEMENTS))
        for col_id in range(primary_resource_cols['dimension']['length']):
            cmd = metadata.query((mdlu['primary_resource_id']['selector'][0], metadata_module.ALL_ELEMENTS, col_id))
            col_name = cmd['name']
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
                        if audio_res_type in foreign_resource_md['semantic_types'] and \
                            'https://metadata.datadrivendiscovery.org/types/FileName' in foreign_col_md['semantic_types']:
                                #mdlu['audio_location_base_uris']['value'] = foreign_col_md['location_base_uris']
                                cls._update_metadata_lookup(mdlu,
                                    'audio_fn',
                                    (mdlu['primary_resource_id']['selector'][0], metadata_module.ALL_ELEMENTS, col_name)
                                  )
                                cls._update_metadata_lookup(mdlu,
                                    'audio_location_base_uris',
                                     foreign_col_selector
                                  )
                            #for fcol_id in range(foreign_resource_md['dimension']['length']):
                            #    fcmd = metadata.query((foreign_resource_id, metadata_module.ALL_ELEMENTS, fcol_id))
                            #    if ''
                        else:
                            _logger.warning('Expected foreign resource of type %s and column of semantic type Filename' % (audio_res_type))
                    else:
                        _logger.warning('Unexpected semantic type Attribute')
                elif 'https://metadata.datadrivendiscovery.org/types/Boundary' in st:
                    if cmd['name'] == 'start':
                        cls._update_metadata_lookup(mdlu, 'audio_start_time',
                            (mdlu['primary_resource_id']['selector'][0], metadata_module.ALL_ELEMENTS, col_name)
                          )
                        pass
                    elif cmd['name'] == 'end':
                        cls._update_metadata_lookup(mdlu, 'audio_end_time',
                            (mdlu['primary_resource_id']['selector'][0], metadata_module.ALL_ELEMENTS, col_name)
                          )
                        pass
                    else:
                        raise Exception('Unsupported Boundary %s' %s (cmd['name']))
                elif 'https://metadata.datadrivendiscovery.org/types/InstanceWeight' in st:
                    _logger.warning('Semantic type InstanceWeight recognized but unused in the current implementation')
                elif 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in st:
                    _logger.info('Semantic type SuggestedTarget is ignored by this primitive')
                else:
                    raise Exception('Semantic type(s) %s does not match any supported types' % (st))

        #audio_res_type = 'http://schema.org/AudioObject'
        #audio_fn_resource_ids = [ res_id for res_id in resources
        #                            if metadata.query((res_id,))['semantic_types'] == (audio_res_type,) ]
        #if len(audio_fn_resource_ids) != 1:
        #    raise Exception('One resource of semantic_types = (%s,) supported' % audio_res_type)
        #mdlu['audio_fn_resource_id']['selector'] = (audio_fn_resource_ids[0], )

        return mdlu if cls._valid_metadata_lookup(mdlu) else None


def ffmpeg_load_audio(filename, sr=16000, mono=False, normalize=True, in_type=np.int16, out_type=np.float32):
    channels = 1 if mono else 2
    command = [
        'ffmpeg',
        '-i', filename,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', str(sr),
        '-ac', str(channels),
        '-']
    try:
        p=sp.check_output(command, stderr=DEVNULL)
        audio=np.frombuffer(p, dtype=in_type).astype(out_type)
    except sp.CalledProcessError as e:
        _logger.error(e)
        _logger.error(e.cmd)
        _logger.error(e.output)
        raise Exception("Could not open audio file " + str(filename))
    except Exception as e:
        _logger.error(e)
    #p = sp.Popen(command, stdout=sp.PIPE, stderr=DEVNULL, bufsize=10**8, close_fds=True)
    #bytes_per_sample = np.dtype(in_type).itemsize
    #frame_size = bytes_per_sample * channels
    #chunk_size = frame_size * sr
    #raw = b''
    #with p.stdout as stdout:
    #    while True:
    #        data = stdout.read(chunk_size)
    #        if data:
    #            raw += data
    #        else:
    #            break
    #audio = np.fromstring(raw, dtype=in_type).astype(out_type)
    if channels > 1:
        audio = audio.reshape((-1, channels)).transpose()
    if audio.size == 0:
        return audio, sr
    if issubclass(out_type, np.floating):
        if normalize:
            peak = np.abs(audio).max()
            if peak > 0:
                audio /= peak
        elif issubclass(in_type, np.integer):
            audio /= np.iinfo(in_type).max
    del p
    #del raw
    return audio, sr
