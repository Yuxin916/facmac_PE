from functools import partial
import os
import yaml
from src.envs.PE.pursuit_ma_env import PursuitMAEnv

from .multiagentenv import MultiAgentEnv
# from .matrix_game.cts_matrix_game import Matrixgame as CtsMatrix
from .particle import Particle
# from .mamujoco import ManyAgentAntEnv, ManyAgentSwimmerEnv, MujocoMulti
# from smac.env import MultiAgentEnv, StarCraft2Env


def env_fn(env, **kwargs) -> MultiAgentEnv:
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)

REGISTRY = {}
# REGISTRY["cts_matrix_game"] = partial(env_fn, env=CtsMatrix)
REGISTRY["particle"] = partial(env_fn, env=Particle)
# REGISTRY["mujoco_multi"] = partial(env_fn, env=MujocoMulti)
# REGISTRY["manyagent_swimmer"] = partial(env_fn, env=ManyAgentSwimmerEnv)
# REGISTRY["manyagent_ant"] = partial(env_fn, env=ManyAgentAntEnv)
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

file_name = os.path.join(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'config')),"envs", "{}.yaml"
        .format('PE'))
with open(file_name,'r') as f:
    try:
        config_dict = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        assert False, "{}.yaml error: {}".format('PE', exc)

env = PursuitMAEnv(config_dict)
REGISTRY["PE"] = env