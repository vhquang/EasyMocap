from typing import Any
from easymocap.config import Config, load_object
from easymocap.mytools.debug_utils import mywarn, log
import numpy as np
import time
from tabulate import tabulate

import typing

if typing.TYPE_CHECKING:
    from myeasymocap.operations.optimizer import Optimizer

class Timer:
    def __init__(self, record, verbose) -> None:
        self.keys = list(record.keys())
        self.header = self.keys
        self.verbose = verbose

    def update(self, timer):
        if not self.verbose:
            return
        contents = []
        for key in self.keys:
            if key not in timer:
                contents.append('skip')
            else:
                contents.append('{:.3f}s'.format(timer[key]))
        print(tabulate(headers=self.header, tabular_data=[contents], tablefmt='fancy_grid'))

class MultiStage:
    def load_final(self):
        at_finals = {}
        for key, val in self._at_final.items():
            if 'module' not in val.keys():
                continue
            if val['module'] == 'skip':
                mywarn('Stage {} is not used'.format(key))
                continue
            log('[{}] loading {}'.format(self.__class__.__name__, key))
            model = load_object(val['module'], val['args'])
            model.output = self.output
            at_finals[key] = model
        self.model_finals = at_finals

    def __init__(self, output, at_step, at_final, keys_keep=[], timer=False) -> None:
        log('[{}] writing the results to {}'.format(self.__class__.__name__, output))
        at_steps = {}
        for key, val in at_step.items():
            if val['module'] == 'skip':
                mywarn('Stage {} is not used'.format(key))
                continue
            log('[{}] loading module {}'.format(self.__class__.__name__, key))
            model = load_object(val['module'], val['args'])
            model.output = output
            at_steps[key] = model
        self.output = output
        self.model_steps = at_steps
        self._at_step = at_step
        self._at_final = at_final
        self.keys_keep = keys_keep
        self.timer = Timer(at_steps, verbose=timer)

    def at_step(self, data, index) -> dict:
        ret = {}
        if 'meta' in data:
            ret['meta'] = data['meta']
        for key in self.keys_keep:
            ret[key] = data[key]
        timer = {}
        for key, model in self.model_steps.items():
            for k in self._at_step[key].get('key_keep', []):
                ret[k] = data[k]
            if self._at_step[key].get('skip', False):
                continue
            inputs = {}
            for k in self._at_step[key].get('key_from_data', []):
                inputs[k] = data[k]
            for k in self._at_step[key].get('key_from_previous', []):
                inputs[k] = ret[k]
            start = time.time()
            try:
                output = model(**inputs)
            except:
                print('[{}] Error in {}'.format('Stages', key))
                raise Exception
            timer[key] = time.time() - start
            if output is not None:
                ret.update(output)

        self.timer.update(timer)
        return ret

    @staticmethod
    def merge_data(infos_all):
        info0 = infos_all[0]
        data = {}
        for key, val in info0.items():
            data[key] = [info[key] for info in infos_all]
            if isinstance(val, np.ndarray):
                try:
                    data[key] = np.stack(data[key])
                except ValueError:
                    print('[{}] Skip merge {}'.format('Stages', key))
                    pass
            elif isinstance(val, dict):
                data[key] = MultiStage.merge_data(data[key])
        return data

    def at_final(self, infos_all) -> dict:
        self.load_final()
        data = self.merge_data(infos_all)
        log('Keep keys: {}'.format(list(data.keys())))
        ret = {}
        """
        load_body_model <myeasymocap.io.model.SMPLLoader object at 0x7d4c3ffa67d0>
        init_translation <myeasymocap.operations.init.InitTranslation object at 0x7d4b127b5060>
        smooth <myeasymocap.operations.smooth.SmoothPoses object at 0x7d4b127b4490>
        mean_param <myeasymocap.operations.init.MeanShapes object at 0x7d4b127a3f10>
        init_RT <myeasymocap.operations.optimizer.Optimizer object at 0x7d4b127a0310>
        refine_poses <myeasymocap.operations.optimizer.Optimizer object at 0x7d4b127a0610>
        write <myeasymocap.io.write.WriteSMPL object at 0x7d4b127a0460>
        render <myeasymocap.io.vis3d.Render object at 0x7d4b127a0f10>
        make_video <myeasymocap.io.video.MakeVideo object at 0x7d4b127a13f0>
        """
        for key, model in self.model_finals.items():
            if self._at_final[key].get('skip', False):
                continue
            at_final = self._at_final[key]
            iter_num = at_final.get('repeat', 1)
            # TODO: check if model mutates the data.
            # key_from_data = at_final.get('key_from_data', [])
            # key_from_previous = at_final.get('key_from_previous', [])

            start = time.time()
            for iter_ in range(iter_num):
                inputs = {}
                model.iter = iter_
                for k in at_final.get('key_from_data', []):
                    inputs[k] = data[k]
                for k in at_final.get('key_from_previous', []):
                    inputs[k] = ret[k]
                try:
                    output = model(**inputs)
                except:
                    print('[{}] Error in {}'.format('Stages', key))
                    raise Exception
                if output is not None:
                    ret.update(output)
            duration = time.time() - start
            print(f'{model.__class__.__name__} runs {iter_num} iter in {duration:.3f}s \n')
        
        return ret


class StageForFittingEach:
    def __init__(self, stages, keys_keep) -> None:
        stages_ = {}
        for key, val in stages.items():
            if val['module'] == 'skip':
                mywarn('Stage {} is not used'.format(key))
                continue
            model = load_object(val['module'], val['args'])
            stages_[key] = model
        self.stages = stages_
        self.stages_args = stages
        self.keys_keep = keys_keep
    
    def __call__(self, results, **ret):
        for pid, result in results.items():
            print('[{}] Optimize person {} with {} frames'.format(self.__class__.__name__, pid, len(result['frames'])))
            ret0 = {}
            ret0.update(ret)
            for key, stage in self.stages.items():
                for iter_ in range(self.stages_args[key].get('repeat', 1)):
                    inputs = {}
                    stage.iter = iter_
                    for k in self.stages_args[key].get('key_from_data', []):
                        inputs[k] = result[k]
                    for k in self.stages_args[key].get('key_from_previous', []):
                        inputs[k] = ret0[k]
                    output = stage(**inputs)
                    if output is not None:
                        ret0.update(output)
            for key in self.keys_keep:
                result[key] = ret0[key]
        return {'results': results}