from os import path

import gym
import minerl
import numpy as np
from minerl.data import BufferedBatchIter

from algorithms.common import Experience

gym.logger.set_level(40)


def optional_observation_map(env, inner_obs):
    """
    If the env implements the `observation` function (i.e. if one of the
    wrappers is an ObservationWrapper), call that `observation` transformation
    on the observation produced by the inner environment
    """
    if hasattr(env, 'observation'):
        return env.observation(inner_obs)
    else:
        return inner_obs


def optional_action_map(env, inner_action):
    """
    This is doing something slightly tricky that is explained in the documentation for
    RecursiveActionWrapper (which TODO should eventually be in MineRL)
    Basically, it needs to apply `reverse_action` transformations from the inside out
    when converting the actions stored and used in a dataset

    """
    if hasattr(env, 'wrap_action'):
        return env.wrap_action(inner_action)
    else:
        return inner_action


def recursive_squeeze(dictlike):
    """
    Take a possibly-nested dictionary-like object of which all leaf elements are numpy ar
    """
    out = {}
    for k, v in dictlike.items():
        if isinstance(v, dict):
            out[k] = recursive_squeeze(v)
        else:
            out[k] = np.squeeze(v)
    return out


def create_data_iterator(
        wrapped_env: gym.Env,
        save_path: str,
        num_batches: int,
        remove_no_ops: bool = True
) -> Experience:
    env_id = wrapped_env.unwrapped.spec.id

    # Download data if needed
    if not path.exists(path.join(save_path, env_id)):
        minerl.data.download(
            directory=save_path,
            environment=env_id,
            update_environment_variables=False,
            disable_cache=True
        )

    data_pipeline = minerl.data.make(env_id, save_path)
    buffered_iterator = BufferedBatchIter(data_pipeline)
    if num_batches is None:
        num_epochs=1
    else:
        num_epochs=None
    for current_obs, action, reward, next_obs, done in buffered_iterator.buffered_batch_iter(batch_size=1,
                                                                                             num_epochs=num_epochs,
                                                                                             num_batches=num_batches):
        wrapped_obs = optional_observation_map(wrapped_env,
                                               recursive_squeeze(current_obs))
        wrapped_next_obs = optional_observation_map(wrapped_env,
                                                    recursive_squeeze(next_obs))
        wrapped_action = optional_action_map(wrapped_env,
                                             recursive_squeeze(action))

        if remove_no_ops and wrapped_action < 7:
            # This definitely makes assumptions about the action space, namely that all-zeros corresponds to a no-op
            yield Experience(wrapped_obs, wrapped_action, reward, wrapped_next_obs, done.squeeze())


class ExtractPOVAndTranspose(gym.ObservationWrapper):
    """
    Basically what it says on the tin. Extracts only the POV observation out of the `obs` dict,
    and transposes those observations to be in the (C, H, W) format used by stable_baselines and imitation
    """
    def __init__(self, env):
        super().__init__(env)
        non_transposed_shape = self.env.observation_space['pov'].shape
        self.high = np.max(self.env.observation_space['pov'].high)
        transposed_shape = (non_transposed_shape[2],
                            non_transposed_shape[0],
                            non_transposed_shape[1])
        # Note: this assumes the Box is of the form where low/high values are vector but need to be scalar
        transposed_obs_space = gym.spaces.Box(low=np.min(self.env.observation_space['pov'].low),
                                              high=np.max(self.env.observation_space['pov'].high),
                                              shape=transposed_shape,
                                              dtype=np.uint8)
        self.observation_space = transposed_obs_space

    def observation(self, observation):
        # Minecraft returns shapes in NHWC by default
        return np.swapaxes(observation['pov'], -1, -3)


class ReversibleActionWrapper(gym.ActionWrapper):
    """
    The goal of this wrapper is to add a layer of functionality on top of the normal ActionWrapper,
    and specifically to implement a way to start:
    (1) Construct a wrapped environment, and
    (2) Take in actions in whatever action schema is dictated by the innermost env, and then apply all action
    transformations/restructurings in the order they would be applied during live environment steps:
    from the inside out

    This functionality is primarily intended for converting a dataset of actions stored in the action
    schema of the internal env into a dataset of actions stored in the schema produced by the applied set of
    ActionWrappers, so that you can train an imitation model on such a dataset and easily transfer to the action
    schema of the wrapped environment for rollouts, RL, etc.

    Mechanically, this is done by assuming that all ActionWrappers have a `reverse_action` implemented
    and recursively constructing a method to call all of the `reverse_action` methods from inside out.

    As an example:
        > wrapped_env = C(B(A(env)))
    If I assume all of (A, B, and C) are action wrappers, and I pass an action to wrapped_env.step(),
    that's equivalent to calling all of the `action` transformations from outside in:
        > env.step(A.action(B.action(C.action(act)))

    In the case covered by this wrapper, we want to perform the reverse operation, so we want to return:
        > C.reverse_action(B.reverse_action(A.reverse_action(inner_action)))

    To do this, the `wrap_action` method searches recursively for the base case where there are no more
    `ReversibleActionWrappers` (meaning we've either reached the base env, or all of the wrappers between us and the
    base env are not ReversibleActionWrappers) by checking whether `wrap_action` is implemented. Once we reach the base
    case, we return self.reverse_action(inner_action), and then call all of the self.reverse_action() methods on the way
    out of the recursion

    """

    def wrap_action(self, inner_action):
        """
        :param inner_action: An action in the format of the innermost env's action_space
        :return: An action in the format of the action space of the fully wrapped env
        """
        if hasattr(self.env, 'wrap_action'):
            return self.reverse_action(self.env.wrap_action(inner_action))
        else:
            return self.reverse_action(inner_action)

    def reverse_action(self, action):
        raise NotImplementedError(
            "In order to use a ReversibleActionWrapper, you need to implement a `reverse_action` function"
            "that is the inverse of the transformation performed on an action that comes into the wrapper")


class ActionShaping(ReversibleActionWrapper):
    def __init__(
            self,
            env: gym.Env,
            camera_angle: int = 10,
            always_attack: bool = False,
            camera_margin: int = 5,
    ):
        """
        Arguments:
            env: The env to wrap.
            camera_angle: Discretized actions will tilt the camera by this number of
                degrees.
            always_attack: If True, then always send attack=1 to the wrapped environment.
            camera_margin: Used by self.wrap_action. If the continuous camera angle change
                in a dataset action is at least `camera_margin`, then the dataset action
                is discretized as a camera-change action.
        """
        super().__init__(env)

        self.camera_angle = camera_angle
        self.camera_margin = camera_margin
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)],
            [('forward', 1)],
            [('forward', 1), ('jump', 1)],
            [('camera', [-self.camera_angle, 0])],
            [('camera', [self.camera_angle, 0])],
            [('camera', [0, self.camera_angle])],
            [('camera', [0, -self.camera_angle])],
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)

        self.action_space = gym.spaces.Discrete(len(self.actions) + 1)

    def action(self, action):
        if action == 7:
            return self.env.action_space.noop()
        else:
            return self.actions[action]

    def reverse_action(self, action: dict) -> int:
        # Moving camera is most important (horizontal first)
        if action["camera"][0] < -self.camera_margin:
            return 3
        elif action["camera"][0] > self.camera_margin:
            return 4
        elif action["camera"][1] > self.camera_margin:
            return 5
        elif action["camera"][1] < -self.camera_margin:
            return 6
        elif action["forward"] == 1:
            if action["jump"] == 1:
                return 2
            else:
                return 1
        elif action["attack"] == 1:
            return 0
        else:
            # No reasonable mapping (would be no-op)
            return 7
