from collections import OrderedDict
from copy import deepcopy

import gym
import numpy as np
import torch


class ActionManager:
    def __init__(self, camera_angle=10, camera_movement_margin=5, always_attack=False):
        self.camera_angle = camera_angle
        self.camera_movement_margin = camera_movement_margin
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)],
            [('forward', 1)],
            #             [('back', 1)],
            #             [('left', 1)],
            #             [('right', 1)],
            #             [('jump', 1)],
            #             [('forward', 1), ('attack', 1)],
            [('forward', 1), ('jump', 1)],
            [('camera', [-self.camera_angle, 0])],
            [('camera', [self.camera_angle, 0])],
            [('camera', [0, self.camera_angle])],
            [('camera', [0, -self.camera_angle])],
        ]

        zero_action = OrderedDict([
            ('attack', 0),
            ('back', 0),
            ('camera', np.array([0., 0.])),
            ('forward', 0),
            ('jump', 0),
            ('left', 0),
            ('right', 0),
            ('sneak', 0),
            ('sprint', 0)
        ])

        self.actions = []
        for actions in self._actions:
            act = deepcopy(zero_action)
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)

        # Check left invertibility
        for i in range(len(self.actions)):
            assert self.action_id(self.action(i)) == i

        # Check right invertibility
        for action in self.actions:
            assert self.action(self.action_id(action)) == action

        # ActionWrapper is reversible function

    def action(self, action_id):
        return self.actions[action_id]

    def action_id(self, action):
        # Moving camera is most important (horizontal first)
        if action["camera"][0] < -self.camera_movement_margin:
            return 3
        elif action["camera"][0] > self.camera_movement_margin:
            return 4
        elif action["camera"][1] > self.camera_movement_margin:
            return 5
        elif action["camera"][1] < -self.camera_movement_margin:
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
            return -1


class ExtractPOV(gym.ObservationWrapper):
    """

    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['pov']

    def observation(self, observation):
        # Minecraft returns shapes in NHWC by default
        return observation['pov']


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
        raise NotImplementedError("In order to use a ReversibleActionWrapper, you need to implement a `reverse_action` function"
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

  def reverse_action(self, action: dict) -> np.ndarray:
        camera_actions = action["camera"].squeeze()
        attack_actions = action["attack"].squeeze()
        forward_actions = action["forward"].squeeze()
        jump_actions = action["jump"].squeeze()
        batch_size = len(camera_actions)
        actions = np.zeros((batch_size,), dtype=int)

        for i in range(len(camera_actions)):
            # Moving camera is most important (horizontal first)
            if camera_actions[i][0] < -self.camera_margin:
                actions[i] = 3
            elif camera_actions[i][0] > self.camera_margin:
                actions[i] = 4
            elif camera_actions[i][1] > self.camera_margin:
                actions[i] = 5
            elif camera_actions[i][1] < -self.camera_margin:
                actions[i] = 6
            elif forward_actions[i] == 1:
                if jump_actions[i] == 1:
                    actions[i] = 2
                else:
                    actions[i] = 1
            elif attack_actions[i] == 1:
                actions[i] = 0
            else:
                # No reasonable mapping (would be no-op)
                actions[i] = 7

        return actions
