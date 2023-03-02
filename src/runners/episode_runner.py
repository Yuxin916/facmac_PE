from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import torch as th
import numpy as np
import copy
import time
import random
import matplotlib.pyplot as plt


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1, "This is a episode runner. 1 environment to run in parallel"

        if 'sc2' in self.args.env:
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        if 'PE' in self.args.env:
            self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)
        else:
            raise NotImplementedError

        self.episode_limit = self.env.episode_limit

        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        # EpisodeBatch is in episode_buffer.py. It is a class
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,  # TODO why +1?
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.env.reset()
        self.batch = self.new_batch()
        self.t = 0

    def run(self, test_mode=False, **kwargs):
        self.reset()
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        fig, ax = plt.subplots(3)  # Comment here
        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_stats()],
                # "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                  test_mode=test_mode,
                                                  explore=(not test_mode))
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
                # (1,num_agents, action_dim)
            if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
                actions = th.argmax(actions, dim=-1).long()

            if self.args.env in ["particle"]:
                cpu_actions = copy.deepcopy(actions).to("cpu").numpy()
                reward, terminated, env_info = self.env.step(cpu_actions[0])
                if isinstance(reward, (list, tuple)):
                    assert (reward[1:] == reward[:-1]), "reward has to be cooperative!"
                    reward = reward[0]
                episode_return += reward

                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }
            elif self.args.env in ["PE"]:
                cpu_actions = copy.deepcopy(actions).to("cpu").numpy()
                _, reward, done_n, truncate_n, env_info = self.env.step(cpu_actions[0])

                for i, o in enumerate(_):  # Comment this paragraph
                    ax[i].cla()
                    img = o[:21 * 21].reshape((21, 21))
                    ax[i].imshow(img)
                    plt.sca(ax[i])
                    plt.pause(0.01)
                self.env.render(mode="human")

                # for key in env_info.keys():
                #     env_info[key] = any(env_info[key])
                env_info['Collision'] = any(env_info['Collision'])
                env_info['Caught'] = any(env_info['Caught'])
                env_info['Time_limit_reached'] = any(env_info['Time_limit_reached'])

                terminated = any(done_n) or any(truncate_n)
                if isinstance(reward, (list, tuple, np.ndarray)):
                    assert (reward[1:].all() == reward[:-1].all()), "reward has to be cooperative!"
                    reward = reward[0]
                episode_return += reward
                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "Terminated": [(terminated,)],
                    "Collision": [(env_info.get("Collision"),)],
                    "Both_Catch": [(env_info.get("Both_Catch"),)],
                    "Time_limit_reached": [(env_info.get("Time_limit_reached"),)]
                }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        print(episode_return)
        last_data = {
            "state": [self.env.get_stats()],
            # "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode,
                                              explore=(not test_mode))
        else:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

        if getattr(self.args, "action_selector", "epsilon_greedy") == "gumbel":
            actions = th.argmax(actions, dim=-1).long()

        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else "train_"
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if self.args.action_selector is not None and hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
