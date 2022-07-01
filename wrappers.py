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


class ObservationShaping(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return torch.tensor(observation['pov'].copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0


class ActionShaping(gym.ActionWrapper):
    def __init__(self, env: gym.Env, action_manager: ActionManager):
        super().__init__(env)
        self.action_manager = action_manager

        self.action_space = gym.spaces.Discrete(len(action_manager.actions))

    def action(self, action):
        return self.action_manager.action(action)

    def reverse_action(self, action):
        return self.action_manager.action_id(action)
