import datetime
import os

import torch.distributed as dist
import typing
from accelerate import Accelerator
from .server import Server
from .server.utils import InstructionsEnum

import logging
lamorel_logger = logging.getLogger('lamorel_logger')

class Caller:
    '''
    This class should be called by each process.
    It will instantiate the different distributed groups.
    If the current process belongs to the LLM's processes, it will launch the LLM and wait for requests.
    '''
    def __init__(self, config, custom_updater=None, custom_module_functions={}, custom_model_initializer=None):
        self.accelerator = Accelerator()
        assert dist.is_initialized(), "torch distributed must be used!"
        self.__config = config
        self.__grad_fn_model = None

        # Set log level
        numeric_log_level = getattr(logging, config.log_level.upper(), None)
        if not isinstance(numeric_log_level, int):
            raise ValueError('Invalid log level: %s' % config.log_level)
        lamorel_logger.setLevel(numeric_log_level)

        # Initialize distributed groups
        if "gloo_timeout" in config:
            lamorel_logger.info(f"Setting the GLOO timeout to {int(config.gloo_timeout)} seconds.")
            gloo_timeout = datetime.timedelta(seconds=int(config.gloo_timeout))
        else:
            lamorel_logger.info(f"No configuration found for the GLOO timeout, setting it to default: 1800 seconds.")
            gloo_timeout = datetime.timedelta(seconds=1800)

        # RL processes are considered as the first n processes
        rl_processes = list(range(config.distributed_setup_args.n_rl_processes))
        llm_processes = list(range(
            config.distributed_setup_args.n_rl_processes,
            config.distributed_setup_args.n_rl_processes + config.distributed_setup_args.n_llm_processes))

        lamorel_logger.info("Init rl group for process {}".format(self.accelerator.process_index))
        self._rl_group = dist.new_group(
            ranks=rl_processes,
            backend='gloo',
            timeout=gloo_timeout
        )
        lamorel_logger.info("Init llm group for process {}".format(self.accelerator.process_index))
        self._llm_group = dist.new_group(
            ranks=llm_processes,
            backend='gloo',
            timeout=gloo_timeout
        )
        lamorel_logger.info("Init rl-llm group for process {}".format(self.accelerator.process_index))
        self._llm_master_process = rl_processes[-1] + 1  # First LLM process is considered as master
        self._rl_llm_group = dist.new_group(
            ranks=rl_processes + [self._llm_master_process],
            backend='gloo',
            timeout=gloo_timeout
        )

        if self.accelerator.process_index in llm_processes:
            Server(
                config,
                llm_processes.index(self.accelerator.process_index),
                self._llm_group,
                self._llm_master_process,
                self._rl_llm_group,
                len(rl_processes) + 1,
                custom_updater,
                custom_module_functions,
                custom_model_initializer
            )

    def get_rl_dist_group(self):
        '''
        Use this group for communication between RL processes
        :return: torch distributed group
        '''
        return self._rl_group

    def score(self, contexts: typing.List[str], candidates: typing.List[typing.List[str]],
              additional_module_function_keys: typing.List[str] = [], **kwargs):
        '''
        Returns log probabilities for each candidate to follow its context.
        '''
        module_function_keys = ["__score"]
        module_function_keys.extend(additional_module_function_keys)
        result = self.__call_model(InstructionsEnum.FORWARD, True, module_function_keys=module_function_keys,
                                   contexts=contexts, candidates=candidates, **kwargs)
        if additional_module_function_keys == []:
            result = [_r['__score'] for _r in result]
        return result

    def generate(self, contexts: list, return_logprobs: bool = False, **kwargs):
        '''
        Returns for each context a list of dict containing for each generated sequence:
        - `tokens`: the sampled tokens
        - `text`: the generated text as a string (once the LLM's tokenizer has been used on `tokens`)

        If `return_logprobs=False` is passed, this dict contains two additional keys:
        - `tokens_probability`: the probability of each token in the sequence
        - `text_probability`: the probability of the whole sequence (i.e. the product of `tokens_probability`)

        Otherwise, the dict contains the following two additional keys:
        - `tokens_logprob: the log probability of each token in the sequence
        - `text_logprob: the log probability of the whole sequence (i.e. the sum of `tokens_logprob`)
        '''
        return self.__call_model(InstructionsEnum.GENERATE, True,
                                 contexts=contexts, return_logprobs=return_logprobs, **kwargs)

    def update(self, contexts: typing.List[str], candidates: typing.List[typing.List[str]], **kwargs):
        result = self.__call_model(InstructionsEnum.UPDATE, True, contexts=contexts, candidates=candidates, **kwargs)
        if not isinstance(result, list):
            result = [result]
        return result

    def close(self):
        self.__call_model(InstructionsEnum.CLOSE, False)

    def custom_module_fns(self, module_function_keys : typing.List[str], contexts: typing.List[str], **kwargs):
        return self.__call_model(InstructionsEnum.FORWARD, True, module_function_keys=module_function_keys,
                                 contexts=contexts, **kwargs)

    def __call_model(self, instruction, expect_answer, **kwargs):
        dist.gather_object(
            obj={
                "instruction": instruction,
                **kwargs
            }, object_gather_list=None, dst=self._llm_master_process, group=self._rl_llm_group
        )

        if expect_answer:
            results = [None for _ in range(dist.get_world_size(group=self._rl_llm_group))]
            dist.broadcast_object_list(object_list=results, src=self._llm_master_process, group=self._rl_llm_group)
            return results[self.accelerator.process_index]
        else:
            return None


