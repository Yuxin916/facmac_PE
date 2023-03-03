import random
import yaml
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import os
from gym import spaces
from gym.utils import seeding

import hydra
from omegaconf import DictConfig

from src.envs.PE.agv_model import AGVAgent, RobotStatusIdx
from src.envs.PE.pursuit_env_base import PursuitEnvBase
from src.envs.PE.utils import plot_arrow


class PursuitMAEnv(PursuitEnvBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        cfg = cfg['env_args']

        self.distance_reset_max = cfg['constraints']['distance_reset_max']

        self.n_pursuer = cfg['agent']['pursuer']['n']
        self.n_evader = cfg['agent']['evader']['n']
        # Init pursuer and evader models
        assert self.n_pursuer >= 1, "n_pursuer must be greater than 1"
        assert self.n_evader >= 1, "n_evader must be greater than 1"

        self.shared_reward = True

        # Number of pursuer
        self.num_agents = cfg['agent']['pursuer']['n']

        # Init pursuer and evader models
        self.pursuer_agents = [AGVAgent(i, self.max_v, self.min_v, self.max_w, self.max_v_acc, self.max_w_acc,  # Limit
                                        self.init_x, self.init_y, self.init_yaw, 0.0, 0.0, self.pursuer_radius,  # Init
                                        True, self.laser_angle_max, self.laser_angle_min, self.laser_angle_step,
                                        self.laser_range)  # Laser
                               for i in range(self.n_pursuer)]

        self.evader_model = AGVAgent(0, self.max_v, self.min_v, self.max_w, self.max_v_acc, self.max_w_acc,  # Limit
                                     self.target_init_x, self.target_init_y, self.target_init_yaw, 0.0, 0.0,
                                     self.evader_radius,  # Init
                                     laser_on=False)  # Laser

        # Spaces # No agents index
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0  # TODO： ？

        for agents in self.pursuer_agents:
            self.action_space.append(spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                                high=np.array([1.0, 1.0], dtype=np.float32)))
            self.obs_num = agents.laser_map_size + 4  # 4 for reference angle (1) and distance (1) and Execute linear velocity (1) + Execute angular velocity (1)
            share_obs_dim += self.obs_num
            self.observation_space.append(spaces.Box(-np.inf, np.inf, shape=(self.obs_num,), dtype=float))
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)] * self.num_agents

        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.writer = None
        self.episode_limit = cfg['train']['episode_limit']

        self.is_done = True

    def seed(self, seed=None):
        self.np_RNG, seed_ = seeding.np_random(seed)
        print('The seed: ', seed_)
        return [seed_]

    def _get_info(self, agent):
        info = {}
        info["Collision"], info["Caught"], info["Time_limit_reached"] = False, False, False

        if agent.collide:
            info["Collision"] = True
        elif agent.catch:
            info["Caught"] = True
        elif agent.truncate:
            info["Time_limit_reached"] = True
        return info

    def step(self, action_n):

        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        truncate_n = []
        info_n = []

        # set action for each agent
        for i, agent in enumerate(self.pursuer_agents):
            self._set_action(action_n[i, :], agent)
        # Update laser map after update motion
        for i, agent in enumerate(self.pursuer_agents):
            agent_ob_list = []
            for j, agent_ in enumerate(self.pursuer_agents):
                if not i == j:
                    agent_ob_list.append([agent_.state[RobotStatusIdx.XCoordinateID.value],
                                          agent_.state[RobotStatusIdx.YCoordinateID.value]])

                # agent_ob_list.append([self.evader_model.state[RobotStatusIdx.XCoordinateID.value],
                #                 self.evader_model.state[RobotStatusIdx.YCoordinateID.value]])
                #
            if agent.laser_on:
                agent.laser.laser_points_update(np.array(self.ob_list + agent_ob_list),
                                                self.ob_radius,
                                                agent.get_transform(),
                                                self.line_ob_list)

        # record observation for each agent
        for agent in self.pursuer_agents:
            obs_n.append(self.get_obs_agent(agent))
            # TODO: move env check into world step and refine _get_obs for redundant
            agent.collide = self.obstacle_collision_check(agent) or self.boundary_collision_check(agent)
            agent.catch = self.catch_check(agent)
            if agent.catch:
                agent.caught = True
            agent.truncate = self.current_step >= self.limit_step
            # goal_distances.append(self._get_reward(agent)[0])
            reward_n.append([self._get_reward(agent)])
            # info = {'individual_reward': self._get_reward(agent), 'Done': self._get_info(agent)}
            # print(info)
            # info_n.append(info)
            # if info['Done']:
            #     print(str(agent.idx), info)
            # pass
            done_n.append(self._get_done(agent))
            truncate_n.append(agent.truncate)
            info = self._get_info(agent)
            info_n.append(info)

        def merge_dicts_by_key(list_of_dicts):
            merged_dict = {}
            for key in list_of_dicts[0].keys():
                for dictionary in list_of_dicts:
                    if key not in merged_dict:
                        merged_dict[key] = []
                    merged_dict[key].append(dictionary[key])
            return merged_dict

        info_n = merge_dicts_by_key(info_n)
        both_catch = self._get_task_complete()
        if both_catch:
            reward_n = [[reward[0] + 250.0] for reward in reward_n]
            info_n['Both_Catch'] = True
        else:
            info_n['Both_Catch'] = False

        done_n = [done or both_catch for done in done_n]

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        reward = np.sum(reward_n)
        info_n["team reward"] = reward
        if self.shared_reward:
            reward_n = [reward] * self.num_agents

        self.is_done = any(done_n) or any(truncate_n)

        return np.array(obs_n), np.array(reward_n), done_n, truncate_n, info_n

    def reset(self, **kwargs):
        if self.is_done:
            self.is_done = False

            self.reward_list = []

            # obstacle coordinates
            self.ob_list = self.cfg['obstacle']['o_coordinates']

            # initial setting for pursuer
            if self.pursuer_fixed:
                self.init_x = self.cfg['agent']['pursuer']['x']
                self.init_y = self.cfg['agent']['pursuer']['y']
                self.init_yaw = self.cfg['agent']['pursuer']['yaw']
                for agent in self.pursuer_agents:
                    agent.set_init_state(self.init_x, self.init_y, self.init_yaw)
            else:
                for agent in self.pursuer_agents:
                    init_x, init_y = self._random_spawn(agent)
                    init_yaw = random.random() * pi * random.choice([1, -1])
                    agent.set_init_state(init_x, init_y, init_yaw)

            self.pursuer_radius = self.cfg['agent']['pursuer']['radius']
            # initial condition for evader
            self.target_init_x = self.cfg['agent']['evader']['x']
            self.target_init_y = self.cfg['agent']['evader']['y']
            self.target_init_yaw = self.cfg['agent']['evader']['yaw']
            self.ob_radius = self.cfg['agent']['evader']['radius']

            self.evader_model.set_init_state(self.target_init_x, self.target_init_y, self.target_init_yaw)

            self.current_step = 0

            obs_n = []
            for i, agent in enumerate(self.pursuer_agents):
                # self._set_action([-1.0, 0.0], agent)
                obs_n.append(self.get_obs_agent(agent))

        pass

        # return np.array(obs_n), {}

    def render(self, mode='human'):
        plt.sca(self.ax)
        plt.cla()

        # executed linear&angular velocity + set linear&angular velocity
        # plt.text(0, 4, "Exe: " + str(pursuer_state[-4:-2]), fontsize=10)
        # plt.text(0, 4.5, "Des: " + str(pursuer_state[-2:]), fontsize=10)

        # draw pursuer
        for i, agent in enumerate(self.pursuer_agents):
            pursuer_state = agent.state
            plt.text(0, 4 - 0.6 * i, "The no.{} Action: ".format(i + 1) +
                     str([pursuer_state[RobotStatusIdx.LinearVelocityDes.value],
                          pursuer_state[RobotStatusIdx.AngularVelocityDes.value]]), fontsize=10)
            circle_pursuer = plt.Circle(
                (pursuer_state[RobotStatusIdx.XCoordinateID.value], pursuer_state[RobotStatusIdx.YCoordinateID.value]),
                agent.robot_radius, color="r")
            self.ax.add_patch(circle_pursuer)


        # draw obstacles
        if self.ob_list is not None:
            for o in self.ob_list:
                self.ax.add_patch(plt.Circle(o, self.ob_radius, color="black"))
                # draw boundary wall
                rect_wall = plt.Rectangle((self.boundary_xy[0], self.boundary_xy[1]), self.boundary_wh[0],
                                          -self.boundary_wh[1],
                                          fill=False, color="red", linewidth=2)
                self.ax.add_patch(rect_wall)

        # draw evader
        evader_state = self.evader_model.state
        circle_evader = plt.Circle(
            (evader_state[RobotStatusIdx.XCoordinateID.value], evader_state[RobotStatusIdx.YCoordinateID.value]),
            self.evader_radius, color="blue")
        self.ax.add_patch(circle_evader)

        # draw robot movement
        for agent in self.pursuer_agents:
            pursuer_state = agent.state
            plot_arrow(pursuer_state[0], pursuer_state[1], pursuer_state[2])
            # Render laser
            agent.laser.render(self.ax)

        plt.xlim([self.boundary_wall[0] - 1, self.boundary_wall[2] + 1])
        plt.ylim([self.boundary_wall[3] - 1, self.boundary_wall[1] + 1])
        plt.grid(True)

        if self.writer is not None:
            self.writer.grab_frame()
        plt.pause(0.05)

    def _get_reward(self, agent):
        goal_distance = np.linalg.norm(agent.state[:RobotStatusIdx.YawAngleID.value] -
                                       self.evader_model.state[:RobotStatusIdx.YawAngleID.value])

        reward = 1 - goal_distance * 0.1  # TODO

        if agent.collide:
            reward += -20
        # if agent.catch and not agent.caught:  # Only reward once
        #     reward += 20
        if agent.truncate:
            reward += -20

        # return goal_distance, reward - 0.5  # time
        return reward - 0.5  # time

    def _get_done(self, agent):
        if self.current_step >= self.limit_step or agent.collide:
            return True
        else:
            return False

    def _get_task_complete(self):
        for agent in self.pursuer_agents:
            if not agent.catch:
                return False
        return True

    def set_writer(self, writer):
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.gca()
        self.writer = writer

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        if all([isinstance(act_space, spaces.Discrete) for act_space in self.action_space]):
            return max([x.n for x in self.action_space])
        elif all([isinstance(act_space, spaces.Box) for act_space in self.action_space]):
            # if self.scenario_name == "simple_speaker_listener":
            #     return self.action_space[0].shape[0] + self.action_space[1].shape[0]
            # else:
            return max([x.shape[0] for x in self.action_space])
        elif all([isinstance(act_space, spaces.Tuple) for act_space in self.action_space]):
            return max([x.spaces[0].shape[0] + x.spaces[1].shape[0] for x in self.action_space])
        else:
            raise Exception("not implemented for this scenario!")

    def get_avail_actions(self):
        return np.ones((self.num_agents, self.get_total_actions()))

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs_n = []
        for _, agent in enumerate(self.pursuer_agents):
            obs = self.get_obs_agent(agent)
            obs_n.append(obs)
        return obs_n

    def get_stats(self):
        states = np.concatenate(self.get_obs())
        return states

    def get_state_size(self):
        """ Returns the shape of the state"""
        state_size = len(self.get_stats())
        return state_size

    def get_env_info(self):
        action_spaces = self.action_space

        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.observation_space[0].shape[0],
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.num_agents,
                    "episode_limit": self.episode_limit,
                    "action_spaces": action_spaces,
                    "actions_dtype": np.float32,
                    "normalise_actions": False}
        return env_info

    def _random_spawn(self, agent, gap=0.0):
        valid = False
        while not valid:
            if agent.idx == 0:
                init_x = self.boundary_xy[0] + random.random() * self.boundary_wh[0]
                init_y = self.boundary_xy[1] - random.random() * self.boundary_wh[1]
            else:
                init_x = self.pursuer_agents[0].state[
                             RobotStatusIdx.XCoordinateID.value] + random.random() * self.distance_reset_max * random.choice(
                    (-1, 1))
                init_y = self.pursuer_agents[0].state[
                             RobotStatusIdx.YCoordinateID.value] + random.random() * self.distance_reset_max * random.choice(
                    (-1, 1))
                if init_x < self.boundary_wall[0] or init_x > self.boundary_wall[1] or init_y > self.boundary_wall[
                    2] or init_y < self.boundary_wall[3]:
                    pass
            for ob in self.ob_list:
                if np.linalg.norm(np.array(
                        [init_x, init_y]) - ob) <= agent.robot_radius + self.ob_radius + gap:
                    continue
            valid = True
        return init_x, init_y


def fixed_action_env_test(env):
    # show plot
    fig, ax = plt.subplots(3)
    for _ in range(100):
        env.reset()
        a = 0
        while a < 100:
            action = np.array([[1.0, 0.0], [0.0, 1.0], [-1, -1]])
            obs, reward, done, truncate, info = env.step(action)
            for i, o in enumerate(obs):
                ax[i].cla()
                img = o[:21*21].reshape((21, 21))
                ax[i].imshow(img)
                plt.sca(ax[i])
                plt.pause(0.01)
            env.render(mode="human")
            print(reward)
            done_flag = False
            for ele in done:
                if ele:
                    done_flag = True
                    break
            if done_flag:
                break
            for ele in truncate:
                if ele:
                    done_flag = True
                    break
            if done_flag:
                break
            # print("The new reward is {}".format(reward))
            a += 1


@hydra.main(version_base=None, config_path="./", config_name="pursuit_ma_env")
def main(cfg: DictConfig):
    '''call env'''
    env = PursuitMAEnv(cfg)

    '''Test Env'''
    fixed_action_env_test(env)


def main_yaml():
    file_name = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config')), "envs",
                             "{}.yaml"
                             .format('PE'))
    with open(file_name) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    '''call env'''
    env = PursuitMAEnv(cfg)

    '''Test Env'''
    fixed_action_env_test(env)


if __name__ == '__main__':
    # main()
    main_yaml()
