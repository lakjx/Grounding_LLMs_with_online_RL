'''
PPO implementation taken from https://github.com/openai/spinningup
'''
import itertools
from collections import OrderedDict
from typing import List
import hydra
from utils.ppo_buffer import PPOBuffer
from utils.generate_prompt import generate_prompt
from utils.scoring_utils import scores_stacking
import torch
import numpy as np
import logging
import wandb
from tqdm import tqdm
import time
import pickle
import math
import os
import functools as f
from operator import add
from transformers import set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from babyai_text_env import UDTSEnv
from macprotocol import MacProtocolEnv
from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater, BaseModuleFunction, BaseModelInitializer

lamorel_init()

from accelerate import Accelerator
accelerator = Accelerator()

class LogScoringModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pad_token = 0
        self._pre_encoded_input = pre_encoded_input

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        if self._model_type == "causal":
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"]) # inputs are padded so all of same size

            logits = forward_outputs["logits"][:, end_of_context_position:-1, :]
            output_tokens = minibatch["input_ids"][:, end_of_context_position+1:]
        else:
            logits = forward_outputs["logits"][:, :-1, :]  # skip </s> token appended by tokenizer
            output_tokens = minibatch["decoder_input_ids"][:, 1:]  # skip pad token

        tokens_logprobs = \
            torch.gather(logits, 2, output_tokens[:, :, None]).squeeze(-1).to(torch.float32)  # filter with sequence tokens

        # Compute mask to assign probability 1 to padding tokens
        mask = torch.ones(tokens_logprobs.shape, dtype=torch.bool, device=self.device)
        for i, _output in enumerate(output_tokens):
            for j, _token in enumerate(_output):
                if _token != self._pad_token:
                    mask[i, j] = False
        masked_token_probs = tokens_logprobs.masked_fill(mask, 0.0)  # apply mask
        minibatch_probs = masked_token_probs.sum(-1)  # compute final sequences' probability

        return minibatch_probs.cpu()

class ValueHeadModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pre_encoded_input = pre_encoded_input

    def initialize(self):
        if 'hidden_size' in self.llm_config.attribute_map:
            _hidden_size_key = self.llm_config.attribute_map['hidden_size']
        else:
            if "word_embed_proj_dim" in self.llm_config.to_dict():
                _hidden_size_key = "word_embed_proj_dim"
            elif "hidden_size" in self.llm_config.to_dict():
                _hidden_size_key = "hidden_size"
            else:
                print(self.llm_config.to_dict())
                raise NotImplementedError("Unknown hidden size key")

        self._llm_hidden_size = self.llm_config.to_dict()[_hidden_size_key]
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(self._llm_hidden_size, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1024, 1),
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        # Get last layer's hidden from last token in context
        if self._model_type == "causal":
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"])  # inputs are padded so all of same size

            model_head = forward_outputs['hidden_states'][-1][:, end_of_context_position, :]
        else:
            model_head = forward_outputs["decoder_hidden_states"][-1][:, 0, :]

        value = self.value_head_op(model_head.to(torch.float32).to(self.device))
        return value.cpu()

class SequentialInitializer(BaseModelInitializer):
    def __init__(self, initializers:List[BaseModelInitializer]):
        super().__init__()
        self._initializers = initializers

    def initialize_model(self, model):
        for _initializer in self._initializers:
            model = _initializer.initialize_model(model)

        return model

class WeightsLoaderInitializer(BaseModelInitializer):
    def __init__(self, weights_path):
        super().__init__()
        self._weights_path = weights_path
        self.is_pretrained = True
    def initialize_model(self, model):
        if self._weights_path is not None:
            loaded_ddp_dict = torch.load(self._weights_path + "/model.checkpoint")
            hf_llm_module_dict = {_k.replace('module.', ''): _v for _k, _v in loaded_ddp_dict.items()}
            model.load_state_dict(state_dict=hf_llm_module_dict, strict=False)
        if not self.is_pretrained:
            model.apply(self.init_weights)
        return model

    def init_weights(self, module):
        if isinstance(module, (torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, (torch.nn.LayerNorm)):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
class PeftInitializer(BaseModelInitializer):
    def __init__(self, model_type, model_name, use_lora, use_4bit, r, alpha, use_cache=True):
        super().__init__()
        self._model_type = model_type
        self._model_name = model_name
        self._use_lora = use_lora
        self._use_4bit = use_4bit
        self._r = r
        self._alpha = alpha
        self._use_cache = use_cache

    def _print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def _get_model_config(self):
        if "t5" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=["q", "v"],
                lora_dropout=0.0,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
        elif "opt" in self._model_name or "Llama" in self._model_name or "Mistral" in self._model_name:
            return LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )
        else:
            raise NotImplementedError()

    def initialize_model(self, model):
        if self._use_lora:
            llm_module = model._modules['_LLM_model']
            if self._model_type == "seq2seq" or not self._use_cache:
                llm_module.gradient_checkpointing_enable()  # reduce number of stored activations

            if self._use_4bit:
                llm_module = prepare_model_for_kbit_training(llm_module)

            # Init adapters #
            config = self._get_model_config()
            peft_model = get_peft_model(llm_module, config)
            parent_module_device = None
            for name, param in peft_model.named_modules():
                if name.split(".")[-1].startswith("lora_"):
                    if hasattr(param, "weight"):
                        param.to(parent_module_device)
                else:
                    if hasattr(param, "weight"):
                        parent_module_device = param.weight.device
                    else:
                        parent_module_device = None

            model._modules['_LLM_model'] = peft_model

        model.eval()  # Important to ensure ratios are 1 in first minibatch of PPO (i.e. no dropout)
        model._modules['_LLM_model'].config.use_cache = self._use_cache
        self._print_trainable_parameters(model)
        return model

class PPOUpdater(BaseUpdater):
    def __init__(self, model_type, minibatch_size, gradient_batch_size, gradient_minibatch_size=None):
        super(PPOUpdater, self).__init__()
        self._model_type = model_type
        self._minibatch_size = minibatch_size
        self._gradient_batch_size = gradient_batch_size
        self._gradient_minibatch_size = gradient_minibatch_size

    def _get_trainable_params(self, model, return_with_names=False):
        if return_with_names:
            return filter(lambda p: p[1].requires_grad, model.named_parameters())
        else:
            return filter(lambda p: p.requires_grad, model.parameters())
        
    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, 'optimizer'):
            self._iterator_named_trainable_params = lambda: self._get_trainable_params(self._llm_module, True)
            self._iterator_trainable_params = (p for n, p in self._iterator_named_trainable_params())
            self.optimizer = torch.optim.Adam(self._iterator_trainable_params, lr=kwargs["lr"])

            if os.path.exists(kwargs["loading_path"] + "/optimizer.checkpoint"):
                self.optimizer.load_state_dict(torch.load(kwargs["loading_path"] + "/optimizer.checkpoint"))

        current_process_buffer = {}
        for k in ['actions', 'advantages', 'returns', 'logprobs', 'values']:
            current_process_buffer[k] = kwargs[k][_current_batch_ids]

        epochs_losses = {
            "value": [],
            "policy": [],
            "loss": []
        }

        n_minibatches = math.ceil(len(contexts) / self._minibatch_size)
        for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
            for step in range(n_minibatches):
                _minibatch_start_idx = step * self._minibatch_size
                _minibatch_end_idx = min(
                    (step + 1) * self._minibatch_size,
                    len(contexts))

                self.optimizer.zero_grad()
                gradient_accumulation_steps = math.ceil(
                    (_minibatch_end_idx - _minibatch_start_idx) / self._gradient_batch_size)
                for accumulated_batch in range(gradient_accumulation_steps):
                    _start_idx = _minibatch_start_idx + accumulated_batch * self._gradient_batch_size
                    _stop_idx = _minibatch_start_idx + min(
                        (accumulated_batch + 1) * self._gradient_batch_size, _minibatch_end_idx)
                    #TODO Batch size !=1
                    # _contexts = contexts[_start_idx:_stop_idx][0]
                    # _candidates = candidates[_start_idx:_stop_idx][0]
                    _contexts = list(itertools.chain.from_iterable(contexts[_start_idx:_stop_idx]))
                    _candidates = list(itertools.chain.from_iterable(candidates[_start_idx:_stop_idx]))
                    if len(_contexts) == 0: break
                    if self._gradient_minibatch_size is None:
                        _batch_size = sum(len(_c) for _c in _candidates)
                    else:
                        _batch_size = self._gradient_minibatch_size
                    # Use LLM to compute again action probabilities and value
                    output = self._llm_module(['score', 'value'], contexts=_contexts, candidates=_candidates,
                                              require_grad=True)
                    # scores = torch.stack([_o['score'] for _o in output]).squeeze()
                    # probas = torch.distributions.Categorical(logits=scores)
                    # values = torch.stack([_o["value"][0] for _o in output]).squeeze()
                    scores = scores_stacking([_o['score'] for _o in output]).reshape(self._gradient_batch_size, 2*kwargs["num_UE"], -1)
                    probas = torch.distributions.Categorical(logits=scores)                   
                    values = torch.stack([_o["value"][0] for _o in output]).reshape(self._gradient_batch_size, 2*kwargs["num_UE"], -1).mean(1).squeeze()
                    # Compute policy loss
                    entropy = probas.entropy().mean()
                    # log_prob = torch.mean(probas.log_prob(current_process_buffer['actions'][_start_idx:_stop_idx])).unsqueeze(0) # Use logprobs from dist as they were normalized
                    log_prob = probas.log_prob(current_process_buffer['actions'][_start_idx:_stop_idx]).mean(1)
                    ratio = torch.exp(log_prob - current_process_buffer['logprobs'][_start_idx:_stop_idx])
                    # assert not (i == 0 and step == 0 and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1)))
                    if i == 0 and step == 0 and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1)):
                        logging.warning("PPO ratio != 1 !!")

                    clip_adv = torch.clamp(ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]) * current_process_buffer['advantages'][_start_idx:_stop_idx]
                    policy_loss = -(torch.min(ratio * current_process_buffer['advantages'][_start_idx:_stop_idx], clip_adv)).mean()
                    epochs_losses["policy"].append(policy_loss.detach().cpu().item())

                    # Compute value loss
                    unclipped_value_error = ((values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                    clipped_values = current_process_buffer['values'][_start_idx:_stop_idx] + \
                                     torch.clamp(values - current_process_buffer['values'][_start_idx:_stop_idx],
                                                 -kwargs["clip_eps"], kwargs["clip_eps"])
                    clipped_value_error = ((clipped_values - current_process_buffer['returns'][_start_idx:_stop_idx]) ** 2)
                    value_loss = torch.max(unclipped_value_error, clipped_value_error).mean()
                    epochs_losses["value"].append(value_loss.detach().cpu().item())

                    # Compute final loss
                    loss = policy_loss - kwargs["entropy_coef"] * entropy + kwargs["value_loss_coef"] * value_loss
                    loss = loss / gradient_accumulation_steps
                    epochs_losses["loss"].append(loss.detach().cpu().item())

                    # Backward
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self._iterator_trainable_params, kwargs["max_grad_norm"])
                self.optimizer.step()

        if kwargs["save_after_update"] and accelerator.process_index == 1:
            print("Saving model...")
            model_state_dict = OrderedDict({
                    k: v for k, v in self._iterator_named_trainable_params()
                })
            torch.save(model_state_dict, kwargs["output_dir"] + "/model.checkpoint")
            torch.save(self.optimizer.state_dict(), kwargs["output_dir"] + "/optimizer.checkpoint")
            print("Model saved")

        return {'loss': np.mean(epochs_losses["loss"]), 'value_loss': np.mean(epochs_losses["value"]),
                'policy_loss': np.mean(epochs_losses["policy"])}

def reset_history():
    return {
        "ep_len": [],
        "ep_ret": [],
        "goal": [],
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "possible_actions": [],
        "actions": [],
        "prompts": [],
    }

@hydra.main(config_path='config', config_name='config')
def main(config_args):
    if config_args.rl_script_args.test:
        test_Env(config_args)
        return 0
    # Random seed
    seed = config_args.rl_script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    # Instantiate environment
    envs = UDTSEnv(config_args.rl_script_args)
    # envs = MacProtocolEnv(config_args.macEnv_args)

    # Create LLM agent
    lm_server = Caller(config_args.lamorel_args,
                       custom_updater=PPOUpdater(config_args.lamorel_args.llm_args.model_type,
                                                 config_args.rl_script_args.minibatch_size,
                                                 config_args.rl_script_args.gradient_batch_size),
                        # custom_model_initializer=WeightsLoaderInitializer(config_args.rl_script_args.loading_path),
                        custom_model_initializer=SequentialInitializer([
                            PeftInitializer(config_args.lamorel_args.llm_args.model_type,
                                            config_args.lamorel_args.llm_args.model_path,
                                            config_args.rl_script_args.use_lora,
                                            config_args.lamorel_args.llm_args.load_in_4bit,
                                            config_args.rl_script_args.lora_r,
                                            config_args.rl_script_args.lora_alpha,
                                            config_args.rl_script_args.use_cache),
                            WeightsLoaderInitializer(config_args.rl_script_args.loading_path)
                        ]),
                       custom_module_functions={
                           'score': LogScoringModuleFn(config_args.lamorel_args.llm_args.model_type,
                                                       config_args.lamorel_args.llm_args.pre_encode_inputs),
                           'value': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type,
                                                      config_args.lamorel_args.llm_args.pre_encode_inputs)
                       })

    # wandb init
    if config_args.rl_script_args.wandb_init:
        wandb.init(
            project=config_args.rl_script_args.wandb_project,
            name=config_args.rl_script_args.wandb_name,
            config={**config_args.rl_script_args}
            )

    # Set up experience buffer
    buffers = [
        PPOBuffer(config_args.rl_script_args.steps_per_epoch // config_args.rl_script_args.number_envs, config_args.rl_script_args.ue_num[ii],
                  config_args.rl_script_args.gamma, config_args.rl_script_args.lam)
        for ii in range(config_args.rl_script_args.number_envs)
    ]

    # Prepare for interaction with environment
    (o, infos), ep_ret, ep_len = envs.reset(), \
        [0 for _ in range(config_args.rl_script_args.number_envs)], \
        [0 for _ in range(config_args.rl_script_args.number_envs)]

    # history = reset_history()

    Action_candidates= ['nothing', 'transmit', 'delete']
    ue_signal_candidate = ['null', 'schedule_request']
    bs_signal_candidate = ['null', 'schedule_grant','acknowledge']
    # ue_signal_candidate = ['Msg_#0', 'Msg_#1', 'Msg_#2', 'Msg_#3']
    # bs_signal_candidate = ['Msg_#0', 'Msg_#1', 'Msg_#2', 'Msg_#3', 'Msg_#4', 'Msg_#5']
    

    for epoch in range(config_args.rl_script_args.epochs):
        __time = time.time()
        for t in tqdm(range(config_args.rl_script_args.steps_per_epoch // config_args.rl_script_args.number_envs),
                      ascii=" " * 9 + ">", ncols=100):
            possible_actions,prompts,env_act,values,log_probs,actions_id = [],[],[],[],[],[]
            for info_a in infos:
                possible_actions_a,prompts_a = [],[]
                for _i in range(info_a["num_UE"]+1):
                    if _i < info_a["num_UE"]:
                        prompts_a.append(info_a[f"UE{_i}_des"]) 
                        possible_actions_a.append([" ".join([x,y]) for x in Action_candidates for y in ue_signal_candidate ])
                    else:
                        prompts_a.extend(info_a["BS_des"])
                        possible_actions_a.extend([z for z in bs_signal_candidate]for _ in range(info_a["num_UE"]))

                output = lm_server.custom_module_fns(['score', 'value'],
                                                    contexts=prompts_a,
                                                    candidates=possible_actions_a)
                scores = scores_stacking([_o['score'] for _o in output])
                proba_dist = torch.distributions.Categorical(logits=scores)
                # values = torch.stack([_o["value"][0] for _o in output])
                value_a = (sum([_o["value"][0] for _o in output])/(info_a["num_UE"]*2))
                sampled_actions = proba_dist.sample()
                # log_probs = proba_dist.log_prob(sampled_actions)
                log_probs_a = torch.mean(proba_dist.log_prob(sampled_actions)).unsqueeze(0)
                actions_id_a = sampled_actions.cpu().numpy()
                UE_action,uplink_message,downlink_message = [],[],[]
                for j in range(len(actions_id_a)):
                    if j < info_a["num_UE"]:
                        command = possible_actions_a[j][int(actions_id_a[j])].split(' ')
                        UE_action.append(Action_candidates.index(command[0]))
                        uplink_message.append(ue_signal_candidate.index(command[-1]))
                    else:
                        downlink_message.append(int(actions_id_a[j]))
                values.append(value_a)
                log_probs.append(log_probs_a)
                actions_id.append(actions_id_a)
                env_act.append([UE_action,uplink_message,downlink_message])
                possible_actions.append(possible_actions_a)
                prompts.append(prompts_a)

            o, rew, d, infos = envs.step(env_act)
            epoch_ended = (t+1)*config_args.rl_script_args.number_envs == config_args.rl_script_args.steps_per_epoch
            bootstrap_dict = {
                "ids": [],
                "contexts": []
            }
            for i in range(config_args.rl_script_args.number_envs):
                buffers[i].store(prompts[i], possible_actions[i], actions_id[i], rew[i], values[i], log_probs[i])
                ep_ret[i] += rew[i]
                ep_len[i] += 1
                timeout = ep_len[i] == config_args.rl_script_args.max_ep_len
                terminal = d[i] or timeout
                if terminal or epoch_ended:
                    if not terminal:
                        bootstrap_dict["ids"].append(i)
                        # bootstrap_dict["contexts"].append(generate_prompt(o[i], infos[i]))
                        bootstrap_dict["contexts"].append(infos[i])
                    else:
                        buffers[i].finish_path(0)
                        ep_len[i], ep_ret[i] = 0, 0
                    (o, infos) = envs.reset()  

            if len(bootstrap_dict["ids"]) > 0:
                # print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                
                for _id,info_a in zip(bootstrap_dict["ids"],bootstrap_dict["contexts"]):
                    prompts_a = []
                    for _i in range(info_a["num_UE"]+1):
                        if _i < info_a["num_UE"]:
                            prompts_a.append(info_a[f"UE{_i}_des"]) 
                        else:
                            prompts_a.extend(info_a["BS_des"])
                    output = lm_server.custom_module_fns(
                        module_function_keys=['value'],
                        contexts=prompts_a,
                        candidates=[[""] for _ in range(len(prompts_a))]
                    )
                    val_a = (sum([_o["value"][0] for _o in output])/(info_a["num_UE"]*2))
                    buffers[_id].finish_path(val_a)

        # Perform PPO update!
        print(f"PPO update number {epoch + 1}")
        is_test_model = (epoch % config_args.rl_script_args.test_freq == 0 or
                         epoch == config_args.rl_script_args.epochs - 1) and epoch != 0
        if is_test_model:
            rwd,test_goput,colls,ariv = test_Env(config_args, lm_server)
            if config_args.rl_script_args.wandb_init:
                for env_id in range(config_args.rl_script_args.number_envs):
                    wandb.log({
                        f"test_Reward/env{env_id}": rwd[env_id],
                        f"test_Goodput/env{env_id}": test_goput[env_id],
                        f"test_CollisionRate/env{env_id}": colls[env_id],
                        f"test_ArrivalRate/env{env_id}": ariv[env_id]
                    })

        save_model_and_history = (epoch % config_args.rl_script_args.save_freq == 0 or
                                  epoch == config_args.rl_script_args.epochs - 1) and epoch != 0
        start_epoch = epoch - config_args.rl_script_args.save_freq
        saving_path = f"{config_args.rl_script_args.output_dir}/continued_epochs_{start_epoch}-{epoch}"
        if save_model_and_history:
            os.makedirs(saving_path, exist_ok=True)
        loading_path = config_args.rl_script_args.loading_path \
            if config_args.rl_script_args.loading_path is not None else ""
        # Stack trajectories for all envs
        # TODO: Randomize and mix up environments' trajectories
        # trajectories = [buf.get() for buf in buffers]
        # collected_trajectories = {
        #     k: torch.cat([traj[k] for traj in trajectories]) if isinstance(trajectories[0][k], torch.Tensor)
        #     else list(f.reduce(add, [traj[k] for traj in trajectories]))
        #     for k, _ in trajectories[0].items()
        # }
        for env_id, buf in enumerate(buffers):
            trajectories = [buf.get()]
            collected_trajectories = {
            k: torch.cat([traj[k] for traj in trajectories]) if isinstance(trajectories[0][k], torch.Tensor)
            else list(f.reduce(add, [traj[k] for traj in trajectories]))
            for k, _ in trajectories[0].items()
            }

            update_results = lm_server.update(collected_trajectories['obs'],
                                            collected_trajectories['possible_act'],
                                            actions=collected_trajectories['act'],
                                            returns=collected_trajectories['ret'],
                                            advantages=collected_trajectories['adv'],
                                            logprobs=collected_trajectories['logp'],
                                            values=collected_trajectories['val'],
                                            lr=config_args.rl_script_args.lr,
                                            clip_eps=config_args.rl_script_args.clip_eps,
                                            entropy_coef=config_args.rl_script_args.entropy_coef,
                                            value_loss_coef=config_args.rl_script_args.value_loss_coef,
                                            max_grad_norm=config_args.rl_script_args.max_grad_norm,
                                            ppo_epochs=config_args.rl_script_args.ppo_epochs,
                                            save_after_update=save_model_and_history,
                                            output_dir=saving_path,
                                            loading_path=loading_path,
                                            num_UE = config_args.rl_script_args.ue_num[env_id],
                                            )

            avg_loss = np.mean([_r['loss'] for _r in update_results])
            avg_policy_loss = np.mean([_r['policy_loss'] for _r in update_results])
            avg_value_loss = np.mean([_r['value_loss'] for _r in update_results])
            print(f"Update loss for env {env_id}: {avg_loss}")

            if config_args.rl_script_args.wandb_init:
                wandb.log({
                    f"train_policy_loss/env{env_id}": avg_policy_loss,
                    f"train_value_loss/env{env_id}": avg_value_loss,
                    })


def test_Env(config_args,llm_md = None):
    seed = config_args.rl_script_args_test.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment
    envs = UDTSEnv(config_args.rl_script_args_test)

    # 初始化测试历史记录
    test_history = {
    f"env{i}": {
        "ep_ret": [],  # 每个 episode 的累积奖励
        "ep_len": [],  # 每个 episode 的长度
        "goodput": [],  # 每个 episode 的平均吞吐量
        "colli_num": [],  # 每个 episode 的平均碰撞率
        "arri_num": []  # 每个 episode 的平均到达率
    }
    for i in range(config_args.rl_script_args_test.number_envs)
}

    # 准备与环境交互
    (o, infos), ep_ret, ep_len = envs.reset(), \
        [0 for _ in range(config_args.rl_script_args_test.number_envs)], \
        [0 for _ in range(config_args.rl_script_args_test.number_envs)]
    Action_candidates = ['nothing', 'transmit', 'delete']
    ue_signal_candidate = ['null', 'schedule_request']
    bs_signal_candidate = ['null', 'schedule_grant', 'acknowledge']
    # ue_signal_candidate = ['Msg_#0', 'Msg_#1', 'Msg_#2', 'Msg_#3']
    # bs_signal_candidate = ['Msg_#0', 'Msg_#1', 'Msg_#2', 'Msg_#3', 'Msg_#4', 'Msg_#5']
    if llm_md is not None:
        lm_server = llm_md
    else:
        lm_server = Caller(config_args.lamorel_args,
                    custom_updater=PPOUpdater(config_args.lamorel_args.llm_args.model_type,
                                                config_args.rl_script_args.minibatch_size,
                                                config_args.rl_script_args.gradient_batch_size),
                    custom_model_initializer=WeightsLoaderInitializer(config_args.rl_script_args.loading_path),
                    custom_module_functions={
                        'score': LogScoringModuleFn(config_args.lamorel_args.llm_args.model_type,
                                                    config_args.lamorel_args.llm_args.pre_encode_inputs),
                        'value': ValueHeadModuleFn(config_args.lamorel_args.llm_args.model_type,
                                                    config_args.lamorel_args.llm_args.pre_encode_inputs)
                    })
    for _ in tqdm(range(config_args.rl_script_args_test.test_num), ascii=" " * 9 + ">", ncols=100, desc="Test"):
        while True:
            possible_actions_all,prompts_all,env_act,actions_id_all = [],[],[],[]
            for info_a in infos:
                possible_actions, prompts = [], []
                for _i in range(info_a["num_UE"] + 1):
                    if _i < info_a["num_UE"]:
                        prompts.append(info_a[f"UE{_i}_des"])
                        possible_actions.append([" ".join([x, y]) for x in Action_candidates for y in ue_signal_candidate])
                    else:
                        prompts.extend(info_a["BS_des"])
                        possible_actions.extend([z for z in bs_signal_candidate] for _ in range(info_a["num_UE"]))

                # 获取模型的输出
                output = lm_server.custom_module_fns(['score'],
                                                    contexts=prompts,
                                                    candidates=possible_actions)
                scores = scores_stacking([_o['score'] for _o in output])
                sampled_actions = torch.argmax(scores, dim=-1)  # 选择得分最高的动作
                # sampled_actions = torch.distributions.Categorical(logits=scores).sample()
                actions_id = sampled_actions.cpu().numpy()

                # 执行动作
                UE_action, uplink_message, downlink_message = [], [], []
                for j in range(len(actions_id)):
                    if j < info_a["num_UE"]:
                        command = possible_actions[j][int(actions_id[j])].split(' ')
                        UE_action.append(Action_candidates.index(command[0]))
                        uplink_message.append(ue_signal_candidate.index(command[-1]))
                    else:
                        downlink_message.append(int(actions_id[j]))
                env_act.append([UE_action,uplink_message,downlink_message])
            
            o, r, d, infos = envs.step(env_act)

            for i in range(config_args.rl_script_args.number_envs):
                # 更新 episode 的累积奖励和长度
                ep_ret[i] += r[i]
                ep_len[i] += 1

            # 如果 episode 结束，记录结果并重置环境
            if ep_len[0] == 24 or d[0]:
                for i in range(config_args.rl_script_args.number_envs):
                    test_history[f"env{i}"]["ep_ret"].append(ep_ret[i])
                    test_history[f"env{i}"]["ep_len"].append(ep_len[i])
                    test_history[f"env{i}"]["goodput"].append(infos[i]["goodput"])
                    test_history[f"env{i}"]["colli_num"].append(infos[i]["colli_num"])
                    test_history[f"env{i}"]["arri_num"].append(infos[i]["arri_num"])
                (o, infos), ep_ret, ep_len = envs.reset(), \
                    [0 for _ in range(config_args.rl_script_args_test.number_envs)], \
                    [0 for _ in range(config_args.rl_script_args_test.number_envs)]
                break
    
    # 计算测试结果
    avg_ep_ret = [np.mean(test_history[f"env{i}"]["ep_ret"]) for i in range(config_args.rl_script_args_test.number_envs)]
    avg_Goodput = [np.mean(test_history[f"env{i}"]["goodput"])/24 for i in range(config_args.rl_script_args_test.number_envs)]
    avg_Collisions = [np.mean(test_history[f"env{i}"]["colli_num"])/24 for i in range(config_args.rl_script_args_test.number_envs)]
    avg_Arrive_rate = [np.mean(test_history[f"env{i}"]["arri_num"])/24 for i in range(config_args.rl_script_args_test.number_envs)]
    
    print(f"Average episode return: {avg_ep_ret}")
    print(f"Average Goodput: {avg_Goodput}")
    print(f"Average Collisions: {avg_Collisions}")
    print(f"Average Arrive rate: {avg_Arrive_rate}")
    return avg_ep_ret, avg_Goodput, avg_Collisions, avg_Arrive_rate
if __name__ == '__main__':
    main()