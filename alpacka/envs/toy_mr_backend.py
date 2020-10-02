# pylint: skip-file

import os
import re
from collections import defaultdict
from copy import deepcopy
from itertools import product

import gym
import numpy as np
import pygame
from gym import Wrapper
from gym.spaces import Box
from gym.spaces.discrete import Discrete
from matplotlib.colors import to_rgb
from matplotlib.pyplot import cm

from alpacka.data import nested_map
from alpacka.utils.hashing import HashableNdarray
from alpacka.utils.images_logging import fig2rgb
from alpacka.utils.images_logging import images2grid

GRID_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)

AGENT_COLOR = (100, 100, 100)
WALL_COLOR = (0, 0, 0)
KEY_COLOR = (218, 165, 32)
DOOR_COLOR = (50, 50, 255)
TRAP_COLOR = (255, 0, 0)
LIVES_COLOR = (0, 255, 0)

# ACTIONS
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

# Cell Code
WALL_CODE = 1
KEY_CODE = 2
DOOR_CODE = 3
TRAP_CODE = 4
AGENT_CODE = 5
LIVES_CODE = 5

# Orders in which keys are taken and doors are open.
# one_room_shifted.txt and hall_way_shifted.txt are modifications of original
# levels with all rooms coordinates increased by one (to avoid zero coordinates)
KEYS_ORDER = {
    "one_room_shifted.txt": [((1, 1), (1, 8))],
    "four_rooms.txt": [((2, 2), (8, 8))],
    "hall_way_shifted.txt": [((1, 1), (8, 1)), ((2, 1), (1, 8)),
                             ((3, 1), (8, 8))],
    "full_mr_map.txt": None,  # there are multiple possible orders
}

DOORS_ORDER = {
    "one_room_shifted.txt": [((1, 1), (8, 9))],
    "four_rooms.txt": [((1, 1), (3, 9))],
    "hall_way_shifted.txt": [((1, 1), (9, 4)), ((2, 1), (9, 4)),
                             ((3, 1), (9, 4))],
    "full_mr_map.txt": None,  # there are multiple possible orders
}

# used for display only
ROOM_ORDER = {
    "hall_way_shifted.txt": [(1, 1), (2, 1), (3, 1)],
    "four_rooms.txt": [(1, 1), (2, 1), (2, 2)],
}


class StaticRoom:
    """ Room data which does NOT change during episode (no keys or doors)"""

    def __init__(self, loc, room_size):
        self.loc = loc
        self.size = room_size
        self.map = np.zeros(room_size, dtype=np.uint8)
        self.walls = set()
        # self.keys = set()
        # self.doors = set()
        self.traps = set()
        self.list_generated = False

    def generate_lists(self):
        if self.list_generated:
            raise ValueError("This is supposed to be called only once.")
        self.walls = set()
        # self.keys = set()
        # self.doors = set()
        self.traps = set()

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if self.map[x, y] != 0:
                    if self.map[x, y] == WALL_CODE:
                        self.walls.add((x, y))
                    # elif self.map[x, y] == KEY_CODE:
                    #     self.keys.add((x, y))
                    # elif self.map[x, y] == DOOR_CODE:
                    #     self.doors.add((x, y))
                    elif self.map[x, y] == TRAP_CODE:
                        self.traps.add((x, y))


tile_size = 16
level_tile_size = 8
hud_height = 16
RENDERING_MODES = ['rgb_array', 'one_hot', "codes"]


class ToyMR(gym.Env):

    def __init__(self, map_file, max_lives=1, absolute_coordinates=False,
                 doors_keys_scale=1, save_enter_cell=True, trap_reward=0.):
        """
        Based on implementation provided here
        https://github.com/chrisgrimm/deep_abstract_q_network/blob/master/toy_mr.py
        Args:
            absolute_coordinates: If use absolute coordinates in observed state.
                E.g. for room = (2, 3), agent = (1, 7), room size = 10, absolute
                coordinates are (2.1, 3.7)
            save_enter_cell: if state should consist enter_cell. Even if set to
                False if max_lives > 1 enter_cell would be encoded into state.
        """
        self.map_file = map_file
        self.max_lives = max_lives

        self.rooms, self.starting_room, self.starting_cell, self.goal_room, self.keys, self.doors = \
            self.parse_map_file(map_file)
        self.room = self.starting_room
        self.room_first_visit = {loc: None for loc in self.rooms.keys()}
        self.t = 0
        self.room_first_visit[self.room.loc] = self.t
        self.agent = self.starting_cell
        self.num_keys = 0

        self.lives = max_lives
        self.enter_cell = self.agent
        self.previous_action = 0
        self.terminal = False
        # self.max_num_actions = max_num_actions
        # self.discovered_rooms = set()
        self.key_neighbor_locs = []
        self.door_neighbor_locs = []
        # self.action_ticker = 0
        self.action_space = Discrete(4)
        # fix order of doors and keys used when cloning / restoring state
        filename = os.path.basename(map_file)
        if KEYS_ORDER.get(filename) is None:
            self.doors_order = tuple(sorted(self.doors.keys()))
            self.keys_order = tuple(sorted(self.keys.keys()))
        else:
            self.doors_order = DOORS_ORDER[filename]
            self.keys_order = KEYS_ORDER[filename]
        assert all(
            [(key_loc in self.keys) for key_loc in self.keys_order])
        assert all(
            [(door_loc in self.doors) for door_loc in self.doors_order])

        self.absolute_coordinates = absolute_coordinates
        if self.absolute_coordinates:
            self.room_size = self.room.size[0]
            for room in self.rooms.values():
                assert room.size == (self.room_size, self.room_size)
        self.doors_keys_scale = doors_keys_scale
        self.save_enter_cell = save_enter_cell
        self.trap_reward = trap_reward
        np_state = self.get_np_state()
        self.observation_space = Box(low=0, high=255, shape=np_state.shape,
                                     dtype=np.uint8)
        self.state_space = self.observation_space  # state == observation

    @staticmethod
    def obs2state(observation, copy=True):
        if copy:
            observation = deepcopy(observation)
        return HashableNdarray(observation)

    @staticmethod
    def state2obs(state):
        return state.array

    @property
    def init_kwargs(self):
        return {
            attr: getattr(self, attr)
            for attr in (
                'map_file', 'max_lives', 'absolute_coordinates',
                'doors_keys_scale', 'save_enter_cell', 'trap_reward'
            )
        }

    @staticmethod
    def flood(y, x, symbol, unchecked_sections, whole_room):
        height = len(whole_room)
        width = len(whole_room[0])
        flood_area = {(y, x)}
        to_flood = {(y, x)}
        while to_flood:
            (y, x) = next(iter(to_flood))
            unchecked_sections.remove((y, x))
            to_flood.remove((y, x))
            neighbors = [(y, x) for (y, x) in
                         [(y + 1, x), (y - 1, x), (y, x - 1), (y, x + 1)]
                         if 0 <= x < width and 0 <= y < height and (
                         y, x) in unchecked_sections
                         and whole_room[y][x] == symbol]
            for n in neighbors:
                to_flood.add(n)
                flood_area.add(n)
        return flood_area

    def check_room_abstraction_consistency(self, whole_room, room_number):
        height = len(whole_room)
        width = len(whole_room[0])
        unchecked_sections = set(
            [(y, x) for x in range(width) for y in range(height)
             if whole_room[y][x] != '|'])
        symbol_area_mapping = dict()
        while unchecked_sections:
            y, x = next(iter(unchecked_sections))
            symbol = whole_room[y][x]
            flood_area = self.flood(y, x, symbol, unchecked_sections,
                                    whole_room)
            if symbol in symbol_area_mapping:
                raise Exception(
                    'Improper Abstraction in Room %s with symbol %s' % (
                    room_number, symbol))
            else:
                symbol_area_mapping[symbol] = flood_area

    @property
    def mode(self):
        return "one_hot"

    def get_state_named_tuple(self):
        attrs = dict()

        if not self.absolute_coordinates:
            attrs["agent"] = self.agent
            attrs['.loc'] = self.room.loc
        else:
            attrs["abs_position"] = tuple(
                np.array(self.room.loc) * self.room_size +
                np.array(self.agent)
            )
        attrs['.num_keys'] = (self.num_keys,)
        if self.max_lives > 1 or self.save_enter_cell:
            attrs["enter_cell"] = self.enter_cell

        for i, key_position in enumerate(self.keys_order):
            attrs['key_%s' % i] = (
                self.keys[key_position] * self.doors_keys_scale,
            )

        for i, doors_possition in enumerate(self.doors_order):
            attrs['door_%s' % i] = (
                self.doors[doors_possition] * self.doors_keys_scale,
            )

        attrs["lives"] = (self.lives,)

        attrs["terminal"] = (self.terminal,)

        return tuple(sorted(attrs.items()))

    def room_and_agent_coordinate(self, abs_coordinate: float):
        room_coord = abs_coordinate // self.room_size
        agent_coord = abs_coordinate % self.room_size
        return room_coord, agent_coord

    def room_and_agent_coordinates(self, abs_coordinates: tuple):
        room_0, agent_0 = self.room_and_agent_coordinate(abs_coordinates[0])
        room_1, agent_1 = self.room_and_agent_coordinate(abs_coordinates[1])
        return (room_0, room_1), (agent_0, agent_1)

    def restore_full_state_from_np_array_version(self, state_np, quick=False):
        del quick
        assert state_np.shape == self.observation_space.shape, f"{state_np.shape} {self.observation_space.shape}"
        # We will use this for structure and names, and ignore values
        assert state_np.shape == self.observation_space.shape
        # state_np = state_np[:, 0, 0]  # remove unused dimensions
        state_tuple_template = self.get_state_named_tuple()
        self.agent = None

        ix = 0
        for name, value in state_tuple_template:
            atrr_size = len(value)
            value = state_np[ix: ix + atrr_size]
            ix += atrr_size
            if name == "agent":
                self.agent = tuple(value)
            elif name == ".loc":
                # Return goal room if .loc is invalid
                self.room = self.rooms.get(tuple(value), self.goal_room)
            elif name == "abs_position":
                assert self.absolute_coordinates
                room_coor, agent_coor = \
                    self.room_and_agent_coordinates(tuple(value))
                self.agent = agent_coor
                self.room = self.rooms[room_coor]
            elif name == "enter_cell":
                assert self.max_lives > 1 or self.save_enter_cell, \
                    "enter_cell is meant to be used in state only when " \
                    "max_lives > 1 or when self.save_enter_cell == True"
                self.enter_cell = tuple(value)
            elif name == ".num_keys":
                self.num_keys = value[0]
            elif name.startswith("key"):
                key_number = int(re.match(
                    'key_(?P<key_number>\d*)', name
                ).group("key_number"))
                key_position = self.keys_order[key_number]
                self.keys[key_position] = bool(
                    value[0] // self.doors_keys_scale)
            elif name.startswith("door"):
                door_number = int(re.match(
                    'door_(?P<door_number>\d*)', name
                ).group("door_number"))
                door_position = self.doors_order[door_number]
                self.doors[door_position] = bool(
                    value[0] // self.doors_keys_scale)
            elif name == "lives":
                self.lives = value[0]
            elif name == "terminal":
                self.terminal = value[0]
            else:
                raise NotImplementedError(
                    f"get_state_named_tuple() is not compatible with this "
                    f"method, please update this method, to handle '{name}'"
                )
        assert ix == state_np.size, f"Data missmatch, loaded {ix} numbers, " \
                                    f"got np_state of size {state_np.size}"

    def get_np_state(self):
        state_tuple = self.get_state_named_tuple()
        state = list()
        for name, val in state_tuple:
            assert isinstance(val, tuple)
            for elem in val:
                assert np.isscalar(elem), f"{elem} {type(elem)}"
                state.append(elem)
        state = np.array(state)
        assert (state >= 0).all()
        assert (state < 256).all()
        state = state.astype(np.float32)
        return state

    def clone_full_state(self):
        state = self.get_np_state()
        return HashableNdarray(state)

    def restore_full_state(self, state):
        assert isinstance(state, HashableNdarray)
        state_np = state.array
        self.restore_full_state_from_np_array_version(state_np)

    @staticmethod
    def parse_map_file(map_file):
        rooms = {}
        keys = {}
        doors = {}

        r = -1
        starting_room, starting_cell, goal_room = None, None, None
        with open(map_file) as f:
            for line in f.read().splitlines():
                if r == -1:
                    room_x, room_y, room_w, room_h = map(int, line.split(' '))
                    room = StaticRoom((room_x, room_y), (room_w, room_h))
                    r = 0
                else:
                    if len(line) == 0:
                        room.generate_lists()
                        rooms[room.loc] = room
                        r = -1
                    elif line == 'G':
                        goal_room = room
                    else:
                        for c, char in enumerate(line):
                            if char == '1':
                                room.map[c, r] = '1'
                            elif char == 'K':
                                # room.map[c, r] = KEY_CODE
                                keys[(room.loc, (c, r))] = True
                            elif char == 'D':
                                # room.map[c, r] = DOOR_CODE
                                doors[(room.loc, (c, r))] = True
                            elif char == 'T':
                                room.map[c, r] = TRAP_CODE
                            elif char == 'S':
                                starting_room = room
                                starting_cell = (c, r)
                        r += 1
        if r >= 0:
            room.generate_lists()
            rooms[room.loc] = room

        if starting_room is None or starting_cell is None:
            raise Exception(
                'You must specify a starting location and goal room')
        return rooms, starting_room, starting_cell, goal_room, keys, doors

    @staticmethod
    def _get_delta(action):
        dx = 0
        dy = 0
        if action == NORTH:
            dy = -1
        elif action == SOUTH:
            dy = 1
        elif action == EAST:
            dx = 1
        elif action == WEST:
            dx = -1
        return dx, dy

    def _move_agent(self, action):
        dx, dy = self._get_delta(action)
        return self.agent[0] + dx, self.agent[1] + dy

    def step(self, action):
        new_agent = self._move_agent(action)
        reward = 0
        self.t += 1

        # room transition checks
        if (new_agent[0] < 0 or new_agent[0] >= self.room.size[0] or
                new_agent[1] < 0 or new_agent[1] >= self.room.size[1]):
            room_dx = 0
            room_dy = 0

            if new_agent[0] < 0:
                room_dx = -1
                new_agent = (self.room.size[0] - 1, new_agent[1])
            elif new_agent[0] >= self.room.size[0]:
                room_dx = 1
                new_agent = (0, new_agent[1])
            elif new_agent[1] < 0:
                room_dy = -1
                new_agent = (new_agent[0], self.room.size[1] - 1)
            elif new_agent[1] >= self.room.size[1]:
                room_dy = 1
                new_agent = (new_agent[0], 0)

            new_room = self.rooms[
                (self.room.loc[0] + room_dx, self.room.loc[1] + room_dy)]
            if self.room_first_visit[new_room.loc] is None:
                self.room_first_visit[new_room.loc] = self.t
            # check intersecting with adjacent door
            if self.doors.get((new_room.loc, new_agent), False):
                if self.num_keys > 0:
                    # new_room.doors.remove(new_agent)
                    self.num_keys -= 1
                    self.doors[(new_room.loc, new_agent)] = False

                    self.room = new_room
                    self.agent = new_agent
                    self.enter_cell = new_agent
            else:
                self.room = new_room
                self.agent = new_agent
                self.enter_cell = new_agent

            if self.room == self.goal_room:
                reward = 1
                self.terminal = True
        else:
            # collision checks
            if self.keys.get((self.room.loc, new_agent), False):
                cell_type = KEY_CODE
            elif self.doors.get((self.room.loc, new_agent), False):
                cell_type = DOOR_CODE
            elif new_agent in self.room.walls:
                cell_type = WALL_CODE
            elif new_agent in self.room.traps:
                cell_type = TRAP_CODE
            else:
                cell_type = 0

            if cell_type == 0:
                self.agent = new_agent
            elif cell_type == KEY_CODE:
                # if self.keys[(self.room.loc, new_agent)]:
                # assert new_agent in self.room.keys
                # self.room.keys.remove(new_agent)
                self.num_keys += 1
                assert (self.room.loc, new_agent) in self.keys
                self.keys[(self.room.loc, new_agent)] = False
                self.agent = new_agent
            elif cell_type == DOOR_CODE:
                if self.num_keys > 0:
                    # assert new_agent in self.room.doors
                    # self.room.doors.remove(new_agent)
                    self.num_keys -= 1
                    self.agent = new_agent
                    assert (self.room.loc, new_agent) in self.doors
                    self.doors[(self.room.loc, new_agent)] = False
            elif cell_type == TRAP_CODE:
                self.lives -= 1
                reward = self.trap_reward
                if self.lives == 0:
                    self.terminal = True
                else:
                    self.agent = self.enter_cell

        # self.action_ticker += 1

        # self.discovered_rooms.add(self.room.loc)
        # return self._get_encoded_room(), reward, self.is_current_state_terminal(), {}
        obs = self.get_np_state()
        done = self.is_current_state_terminal()
        info = {
            "solved": self.room == self.goal_room,
            "room_first_visit": self.room_first_visit,
        }
        return obs, reward, done, info

    def reset(self):
        self.room = self.starting_room
        self.agent = self.starting_cell
        self.num_keys = 0
        self.room_first_visit = {loc: None for loc in self.rooms.keys()}
        self.t = 0
        self.room_first_visit[self.room.loc] = self.t
        self.terminal = False
        # self.action_ticker = 0
        self.lives = self.max_lives
        self.enter_cell = self.agent

        # for room in self.rooms.values():
        #     room.reset()

        for key, val in self.keys.items():
            self.keys[key] = True

        for key, val in self.doors.items():
            self.doors[key] = True

        return self.get_np_state()

    def is_current_state_terminal(self):
        return self.terminal  # or self.action_ticker > self.max_num_actions

    def is_action_safe(self, action):
        new_agent = self._move_agent(action)
        if new_agent in self.room.traps:
            return False
        return True

    def _get_encoded_room(self, room_loc, agent_pos):
        room = self.rooms[room_loc]
        encoded_room = np.zeros((room.size[0], room.size[1]), dtype=np.uint8)
        # This is hack if model put agent out of bound.
        if agent_pos is not None:
            try:
                encoded_room[int(agent_pos[0]), int(agent_pos[1])] = AGENT_CODE
            except IndexError:
                pass
        for coord in room.walls:
            encoded_room[coord] = WALL_CODE

        for (room_coord, key_coord), present in self.keys.items():
            if room_coord == room.loc and present:
                encoded_room[key_coord] = KEY_CODE

        for (room_coord, door_coord), present in self.doors.items():
            if room_coord == room.loc and present:
                encoded_room[door_coord] = DOOR_CODE

        for coord in room.traps:
            encoded_room[coord] = TRAP_CODE

        return encoded_room

    def render(self, mode="one_hot"):
        if mode == 'codes':
            return self._get_encoded_room(self.room.loc, self.agent)
        if mode == 'rgb_array':
            encoded_room = self._get_encoded_room(self.room.loc, self.agent)
            return render_screen(
                encoded_room, self.num_keys, self.lives, self.max_lives
            )
        if mode == "one_hot":
            return self.get_np_state()

    def calculate_position_visit_freq(
            self,
            state_visit_count
    ):
        obs_labels = self._get_dense_obs_labels()
        room_loc_idxs = obs_labels.index('.loc_y'), obs_labels.index('.loc_x')
        agent_pos_idxs = (obs_labels.index('agent_y'),
                          obs_labels.index('agent_x'))
        keys_idxs = tuple(
            idx for idx, label in enumerate(obs_labels) if 'key_' in label
        )
        nb_keys_total = len(keys_idxs)
        max_visits = 0
        assert nb_keys_total == len(self.keys)

        visits_counts = {room_loc: {} for room_loc in self.rooms}

        for state, state_freq in state_visit_count.items():
            obs = self.state2obs(state)
            room_loc = tuple(np.take(obs, room_loc_idxs))
            agent_pos = tuple(np.take(obs, agent_pos_idxs))
            keys_taken = np.sum(1 - np.take(obs, keys_idxs)).astype(int)

            if agent_pos not in visits_counts[room_loc]:
                visits_counts[room_loc][agent_pos] = np.zeros(
                    (1 + nb_keys_total,)
                )

            visits_counts[room_loc][agent_pos][keys_taken] += state_freq
            if visits_counts[room_loc][agent_pos][keys_taken] > max_visits:
                max_visits = visits_counts[room_loc][agent_pos][keys_taken]

        visits_frequencies = nested_map(lambda x: x / max_visits, visits_counts)
        return visits_frequencies

    def separate_position_visit_freqs_by_key(self, position_visit_freqs):
        nb_keys_total = len(self.keys)
        freqs_by_key = {
            keys_taken: {room_loc: {} for room_loc in self.rooms}
            for keys_taken in range(1 + nb_keys_total)
        }
        for nb_keys_taken in range(nb_keys_total):
            for room_loc, room_visit_freqs in position_visit_freqs.items():
                for agent_pos, agent_pos_freq in room_visit_freqs.items():
                    freq = agent_pos_freq[nb_keys_taken]
                    if freq > 0:
                        freqs_by_key_r = freqs_by_key[nb_keys_taken][room_loc]
                        freqs_by_key_r[agent_pos] = np.zeros(
                            (1 + nb_keys_total,)
                        )
                        freqs_by_key_r_a = freqs_by_key_r[agent_pos]
                        freqs_by_key_r_a[nb_keys_taken] = freq

        return freqs_by_key

    def get_highlighted_transitions_pos_with_colors(
            self,
            highlighted_transitions,
            filter_by_nb_keys=None
    ):
        obs_labels = self._get_dense_obs_labels()
        keys_idxs = tuple(
            idx for idx, label in enumerate(obs_labels) if 'key_' in label
        )
        transitions_targets_unique = set()

        for source_state, target_states in highlighted_transitions.items():
            source_obs = self.state2obs(source_state)
            nb_keys_taken = np.sum(1 - np.take(source_obs, keys_idxs))
            if (filter_by_nb_keys is not None and
                    nb_keys_taken != filter_by_nb_keys):
                continue
            for target_state in target_states.values():
                transitions_targets_unique.add(target_state)

        nb_unique_target_states = len(transitions_targets_unique)
        colors = {
            transition_target: (255 * np.array(to_rgb(color))).astype(int)
            for transition_target, color in zip(
                transitions_targets_unique,
                cm.hsv(np.linspace(0, 1, 1 + nb_unique_target_states))
            )
        }

        invalid_transition_sources = {room_loc: [] for room_loc in self.rooms}
        invalid_transition_targets = {room_loc: [] for room_loc in self.rooms}

        room_loc_idxs = obs_labels.index('.loc_y'), obs_labels.index('.loc_x')
        agent_pos_idxs = (obs_labels.index('agent_y'),
                          obs_labels.index('agent_x'))

        for state, highlighted_transitions in highlighted_transitions.items():
            obs = self.state2obs(state)
            room_loc = tuple(np.take(obs, room_loc_idxs))
            agent_pos = tuple(np.take(obs, agent_pos_idxs))
            keys_taken = np.sum(1 - np.take(obs, keys_idxs)).astype(int)

            if (filter_by_nb_keys is not None and
                    keys_taken != filter_by_nb_keys):
                continue

            for action, target_state in highlighted_transitions.items():
                target_obs = self.state2obs(target_state)
                next_room_loc = tuple(np.take(target_obs, room_loc_idxs))
                next_agent_pos = tuple(np.take(target_obs, agent_pos_idxs))

                color = colors[target_state]

                invalid_transition_sources[room_loc].append(
                    (agent_pos, action, color)
                )
                if (next_agent_pos[0] < 0 or
                        next_agent_pos[0] >= self.room.size[0] or
                        next_agent_pos[1] < 0 or
                        next_agent_pos[1] >= self.room.size[1] or
                        next_room_loc not in self.rooms):
                    continue

                invalid_transition_targets[next_room_loc].append(
                    (next_agent_pos, color)
                )

        return invalid_transition_sources, invalid_transition_targets

    def render_single_heat_map(self, pos_visit_freqs, invalid_transitions):
        heat_maps = {}
        invalid_transition_srcs, invalid_transition_tgts = invalid_transitions

        for room_loc in self.rooms:
            encoded_room = self._get_encoded_room(room_loc, agent_pos=None)
            heat_map = render_screen(
                encoded_room, self.num_keys, self.lives, self.max_lives,
                render_agent=False, frequencies=pos_visit_freqs[room_loc],
                invalid_transition=(
                    invalid_transition_srcs[room_loc],
                    invalid_transition_tgts[room_loc],
                )
            )
            heat_maps[room_loc] = heat_map

        return images2grid(heat_maps)

    def render_visit_heat_map(
            self,
            state_visit_count,
            highlighted_transitions,
            separate_by_keys=False
    ):
        pos_visit_frequencies = self.calculate_position_visit_freq(
            state_visit_count
        )

        if separate_by_keys:
            pos_visit_freqs_by_key = self.separate_position_visit_freqs_by_key(
                pos_visit_frequencies
            )

            heat_maps = {}
            nb_keys_total = len(self.keys)
            for nb_keys_taken in range(1 + nb_keys_total):
                transitions = self.get_highlighted_transitions_pos_with_colors(
                    highlighted_transitions,
                    filter_by_nb_keys=nb_keys_taken
                )
                heat_map = self.render_single_heat_map(
                   pos_visit_freqs_by_key[nb_keys_taken], transitions
                )

                n_rows_in_grid = np.ceil(np.sqrt(1 + nb_keys_total)).astype(int)
                col = 1 + (nb_keys_taken // n_rows_in_grid)
                row = 1 + (nb_keys_taken % n_rows_in_grid)
                heat_maps[(col, row)] = heat_map
            return images2grid(heat_maps)
        else:
            transitions = self.get_highlighted_transitions_pos_with_colors(
                highlighted_transitions
            )
            return self.render_single_heat_map(
                pos_visit_frequencies, transitions
            )

    def render_heat_maps(self, state_visit_freq, network):
        map_file_name = os.path.basename(self.map_file)
        ordered_keys = KEYS_ORDER[map_file_name]
        ordered_doors = DOORS_ORDER[map_file_name]

        ordered_rooms = ROOM_ORDER[map_file_name]
        level_mean_by_progress = list()
        level_std_by_progress = list()
        level_freq_by_progress = list()
        level_mask_by_progress = list()
        n_door_key_combinations = len(self.keys) + len(self.doors)

        room_shape = self.starting_room.size
        state_freqs = {}
        for progress in range(n_door_key_combinations):
            n_keys_taken = progress // 2 + progress % 2
            n_doors_open = progress // 2
            state_freqs[(n_keys_taken, n_doors_open)] = {}
            for room_loc in ordered_rooms:
                state_freqs[(n_keys_taken, n_doors_open)][room_loc] = np.zeros(
                    room_shape
                )
        for state, state_freq in state_visit_freq.items():
            obs = state.array
            n_keys = obs[2]
            n_doors = 3 - sum(obs[5:8])
            room_loc = tuple(obs[:2])
            y, x = obs[3:5].astype(int)
            state_freqs[(n_keys, n_doors)][room_loc][y, x] = state_freq

        max_freq = max(state_visit_freq.values())
        state_freqs_scaled = nested_map(lambda x: x / max_freq, state_freqs)
        self.reset()
        start_state = self.clone_full_state()
        for progress in range(n_door_key_combinations):
            n_keys_taken = progress // 2 + progress % 2
            n_doors_open = progress // 2
            rooms_mean = list()
            rooms_std = list()
            rooms_masks = list()
            rooms_freq = list()
            freqs_progress = state_freqs_scaled[(n_keys_taken, n_doors_open)]
            # iterate over rooms
            for room_loc in ordered_rooms:
                room = self.rooms[room_loc]
                mean_img = np.zeros(room_shape)
                std_img = np.zeros(room_shape)
                freq_img = np.zeros(room_shape)
                freqs_room = freqs_progress[room_loc]
                for i, j in product(range(room_shape[0]), range(room_shape[1])):
                    if room.map[i, j] != 0:
                        # traps or walls
                        continue
                    self.restore_full_state(start_state)
                    self.room = room
                    self.agent = (i, j)

                    # set open doors and taken keys
                    for key_coord in ordered_keys[:n_keys_taken]:
                        self.keys[key_coord] = False
                        self.num_keys += 1

                    for door_coord in ordered_doors[:n_doors_open]:
                        self.doors[door_coord] = False
                        self.num_keys -= 1
                    assert self.num_keys in [0, 1]

                    obs = self.render(mode='one_hot')
                    # this will not work properly if _value returns policy
                    ensemble_values = network.predict(
                        np.expand_dims(obs, 0)  # expand "batch" dimension
                    )
                    ensemble_mean = np.mean(ensemble_values)
                    ensemble_std = np.std(ensemble_values)
                    mean_img[i, j] = ensemble_mean
                    std_img[i, j] = ensemble_std
                    freq_img[i, j] = freqs_room[i, j]

                rooms_masks.append(room.map != 0)
                rooms_mean.append(mean_img)
                rooms_std.append(std_img)
                rooms_freq.append(freq_img)

            level_mean = np.concatenate(rooms_mean, axis=0)
            level_std = np.concatenate(rooms_std, axis=0)
            level_freq = np.concatenate(rooms_freq, axis=0)
            level_mask = np.concatenate(rooms_masks, axis=0)

            level_mean_by_progress.append(level_mean)
            level_std_by_progress.append(level_std)
            level_freq_by_progress.append(level_freq)
            level_mask_by_progress.append(level_mask)

        whole_img_mean = np.concatenate(level_mean_by_progress, axis=1)
        whole_img_std = np.concatenate(level_std_by_progress, axis=1)
        whole_img_freq = np.concatenate(level_freq_by_progress, axis=1)
        whole_img_mask = np.concatenate(level_mask_by_progress, axis=1)

        results = []
        for name, stats_img, cmap, info_range in [
            ("std", whole_img_std, 'Blues', (0, 1.)),
            ("mean", whole_img_mean, 'coolwarm', (-1., 1.)),
            ("freq", whole_img_freq, 'Greens', (0., 1.))
        ]:
            img_array = self.visualize_board_info(
                stats_img.transpose(),  # toy_mr coordinates are transposed
                info_range, cmap=cmap,
                shape=(room_shape[1] * 50 * len(ordered_rooms),
                       room_shape[0] * 50 * n_door_key_combinations),
                # transpose
                mask=whole_img_mask.transpose()  # transpose
            )
            results.append(img_array)

        return results

    def visualize_board_info(self, info, info_range, cmap, shape, mask=None,
                             fmt=".0E",
                             annot=False):
        # INFO: mostly copied from validation.py
        # visualize_board_info()
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        (min_info, max_info) = info_range
        import seaborn as sns
        sns.heatmap(info, annot=annot, ax=ax, cmap=cmap,
                    vmin=min_info, fmt=fmt, vmax=max_info,
                    cbar=True, mask=mask,
                    )
        plt.axis('off')
        return fig2rgb(fig, dpi=600)

    def save_map(self, file_name, tile_size=level_tile_size):
        pygame.init()
        map_h = max(coord[1] for coord in self.rooms)
        map_w = max(coord[0] for coord in self.rooms)

        map_ = pygame.Surface((tile_size * self.room.size[0] * map_w,
                               tile_size * self.room.size[1] * map_h))
        map_.fill(BACKGROUND_COLOR)

        for room_loc in self.rooms:
            print(f"loc {room_loc}, size {self.rooms[room_loc].size}")
            room = self.rooms[room_loc]

            room_x, room_y = room_loc
            room_x = (room_x - 1) * tile_size * room.size[0]
            room_y = (room_y - 1) * tile_size * room.size[1]

            if room == self.goal_room:
                continue
                rect = (room_x, room_y, tile_size * room.size[0],
                        tile_size * room.size[1])
                pygame.draw.rect(map_, (0, 255, 255), rect)

                myfont = pygame.font.SysFont('Helvetica', 8 * tile_size)

                # render text
                label = myfont.render("G", True, (0, 0, 0))
                label_rect = label.get_rect(
                    center=(room_x + (tile_size * room.size[0]) / 2,
                            room_y + (tile_size * room.size[1]) / 2))
                map_.blit(label, label_rect)
                continue

            # loop through each row
            for row in range(self.room.size[1] + 1):
                pygame.draw.line(map_, GRID_COLOR,
                                 (room_x, row * tile_size + room_y),
                                 (room.size[1] * tile_size + room_x,
                                  row * tile_size + room_y))
            for column in range(self.room.size[0] + 1):
                pygame.draw.line(map_, GRID_COLOR,
                                 (column * tile_size + room_x, room_y),
                                 (column * tile_size + room_x,
                                  room.size[0] * tile_size + room_y))

            # draw walls
            for coord in room.walls:
                rect = (
                coord[0] * tile_size + room_x, coord[1] * tile_size + room_y,
                tile_size, tile_size)
                pygame.draw.rect(map_, WALL_COLOR, rect)

            # draw key
            for room_coord, key_coord in self.keys.keys():
                if room_coord == room.loc:
                    rect = (key_coord[0] * tile_size + room_x,
                            key_coord[1] * tile_size + room_y, tile_size,
                            tile_size)
                    pygame.draw.rect(map_, KEY_COLOR, rect)

            # # draw doors
            for room_coord, door_coord in self.doors.keys():
                if room_coord == room.loc:
                    rect = (door_coord[0] * tile_size + room_x,
                            door_coord[1] * tile_size + room_y, tile_size,
                            tile_size)
                    pygame.draw.rect(map_, DOOR_COLOR, rect)

            # draw traps
            for coord in room.traps:
                rect = (
                coord[0] * tile_size + room_x, coord[1] * tile_size + room_y,
                tile_size, tile_size)
                pygame.draw.rect(map_, TRAP_COLOR, rect)

        pygame.image.save(map_, file_name + '.png')

    def _get_dense_obs_labels(self):
        sample_observation = self.get_state_named_tuple()
        observation_labels = []
        for attr_name, attr_val in sample_observation:
            if len(attr_val) == 1:
                observation_labels.append(attr_name)
            else:
                observation_labels.append(f'{attr_name}_y')
                observation_labels.append(f'{attr_name}_x')

        return observation_labels


def draw_circle(screen, coord, color, scale=1., pos_inside_tile=None, border=0):
    if pos_inside_tile is None:
        padding = tile_size * (1 - scale) / 2
        padding_x = padding
        padding_y = padding
    else:
        pos_inside_tile_x, pos_inside_tile_y = pos_inside_tile
        padding_x = pos_inside_tile_x * (tile_size / 3)
        padding_y = pos_inside_tile_y * (tile_size / 3)
    rect = (
        coord[0] * tile_size + padding_x,
        coord[1] * tile_size + padding_y + hud_height,
        tile_size * scale,
        tile_size * scale
    )
    if border > 0:
        rect_border = (
            rect[0] - border,
            rect[1] - border,
            rect[2] + 2 * border,
            rect[3] + 2 * border
        )
        pygame.draw.ellipse(screen, (0, 0, 0), rect_border)
    pygame.draw.ellipse(screen, color, rect)


def draw_rect(screen, coord, color, scale=1., pos_inside_tile=None, border=0):
    if pos_inside_tile is None:
        padding = tile_size * (1 - scale) / 2
        padding_x = padding
        padding_y = padding
    else:
        pos_inside_tile_x, pos_inside_tile_y = pos_inside_tile
        padding_x = pos_inside_tile_x * (tile_size / 3)
        padding_y = pos_inside_tile_y * (tile_size / 3)
    rect = (
        coord[0] * tile_size + padding_x,
        coord[1] * tile_size + padding_y + hud_height,
        tile_size * scale,
        tile_size * scale
    )
    if border > 0:
        rect_border = (rect[0] - 1, rect[1] - 1, rect[2] + 2, rect[3] + 2)
        pygame.draw.rect(screen, (0, 0, 0), rect_border)
    pygame.draw.rect(screen, color, rect)


def get_color_shade(nb_keys_taken):
    colors_scheme = {
        0: (np.array((0, 50, 0)), np.array((0, 200, 0))),
        1: (np.array((0, 0, 50)), np.array((0, 0, 200))),
        2: (np.array((50, 0, 0)), np.array((200, 0, 0))),
        3: (np.array((0, 50, 0)), np.array((0, 200, 0))),
        4: (np.array((0, 0, 50)), np.array((0, 0, 200))),
    }
    return colors_scheme[nb_keys_taken]


def get_cell_color(state_key_freqs):
    color_delta = np.array((0., 0., 0.))
    for nb_keys_taken, freq in enumerate(state_key_freqs):
        min_visit_shade, max_visit_shade = get_color_shade(nb_keys_taken)
        color_delta += (
                max_visit_shade * freq +
                min_visit_shade * (1 - freq)
        )

    if max(color_delta) > 255:
        color_delta *= 255 / max(color_delta)

    cell_color = color_delta + (255 - max(color_delta))
    return cell_color


def render_screen(code_state, num_keys, lives, max_lives, render_agent=True,
                  frequencies=None, invalid_transition=None):
    room_size = code_state.shape
    screen = pygame.Surface(
        (room_size[0] * tile_size, room_size[1] * tile_size + hud_height))
    screen.fill(BACKGROUND_COLOR)

    if frequencies is not None:
        for agent_pos, state_key_freqs in frequencies.items():
            state_color = get_cell_color(state_key_freqs)
            draw_rect(screen, agent_pos, state_color)

    # loop through each row
    for row in range(room_size[1] + 1):
        pygame.draw.line(screen, GRID_COLOR, (0, row * tile_size + hud_height),
                         (room_size[1] * tile_size,
                          row * tile_size + hud_height))
    for column in range(room_size[0] + 1):
        pygame.draw.line(screen, GRID_COLOR, (column * tile_size, hud_height),
                         (column * tile_size,
                          room_size[0] * tile_size + hud_height))

    for index_, x in np.ndenumerate(code_state):
        if frequencies is not None and index_ in frequencies:
            continue
        if x == AGENT_CODE and render_agent:
            draw_circle(screen, index_, AGENT_COLOR)

        if x == WALL_CODE:
            draw_rect(screen, index_, WALL_COLOR)

        if x == DOOR_CODE:
            draw_rect(screen, index_, DOOR_COLOR)

        if x == TRAP_CODE:
            draw_rect(screen, index_, TRAP_COLOR)

        if x == KEY_CODE:
            draw_rect(screen, index_, KEY_COLOR)

    if invalid_transition is not None:
        invalid_transition_targets = invalid_transition[1]
        for pos, color in invalid_transition_targets:
            draw_rect(
                screen, pos, color, scale=0.33, pos_inside_tile=(1, 1),
                border=1
            )
        invalid_transition_sources = invalid_transition[0]
        for pos, action, color in invalid_transition_sources:
            if action == NORTH:
                pos_inside_tile = (1, 0)
            elif action == EAST:
                pos_inside_tile = (2, 1)
            elif action == SOUTH:
                pos_inside_tile = (1, 2)
            elif action == WEST:
                pos_inside_tile = (0, 1)
            else:
                raise TypeError(f'Unrecognized action {action}.')

            draw_circle(
                screen, pos, color, scale=0.33, pos_inside_tile=pos_inside_tile,
                border=1
            )

    for i in range(int(num_keys)):
        draw_rect(screen, (i, -1), KEY_COLOR)
    if max_lives > 1:
        for i in range(int(lives)):
            draw_rect(screen, (room_size[0] - 1 - i, -1), LIVES_COLOR)

    image = pygame.surfarray.array3d(screen)
    return image.transpose([1, 0, 2])


class DebugCloneRestoreWrapper(Wrapper):
    """"Performs clone restore operations during step."""

    def __init__(self, env: ToyMR):
        super(DebugCloneRestoreWrapper, self).__init__(env)
        self.second_env = ToyMR(**env.init_kwargs)
        self.t = 0

    def step(self, action):
        if action < 5:  # and random.random() > 0.5:
            state = self.env.clone_full_state()
            self.second_env.restore_full_state(state)
            assert (
                        state.array == self.second_env.clone_full_state().array).all()
            assert state == self.second_env.clone_full_state()
            # swap envs
            self.env, self.second_env = self.second_env, self.env
            # print(f"enter cell {self.env.enter_cell}")
            print(f'step = {self.t}')
            arr_str = ', '.join([str(int(v)) for v in state.array])
            print(f'state = np.array([{arr_str}])')
            print(self.env._get_dense_obs_labels())
            self.t += 1
        return self.env.step(action)


if __name__ == "__main__":
    import random
    from PIL import Image

    map_file_ = 'mr_maps/full_mr_map.txt'  # 'mr_maps/hall_way_shifted.txt' 'mr_maps/four_rooms.txt' 'mr_maps/full_mr_map.txt' 'mr_maps/one_room_shifted.txt'

    env_kwargs = dict(map_file=map_file_, max_lives=5,
                      absolute_coordinates=False, save_enter_cell=False)
    # env_kwargs = dict(map_file=map_file_, max_lives=1, save_enter_cell=False,
    #                   absolute_coordinates=True)
    env = ToyMR(**env_kwargs)
    restore_env = ToyMR(**env_kwargs)
    env.save_map("/tmp/map")

    env.reset()
    for i in range(20):
        if i % 3 == 0:
            state = env.clone_full_state()
            restore_env.restore_full_state(state)
            assert (state == restore_env.clone_full_state())
        a = random.randint(0, 3)
        env.step(a)

        ob = env.render(mode="rgb_array")
        im = Image.fromarray(ob)
        im.save(f"/tmp/mr{i}.png")

    env = DebugCloneRestoreWrapper(ToyMR(**env_kwargs))
    # env = ToyMR(map_file_, max_lives=5)
    from gym.utils.play import play

    keys_to_action = {
        (ord('d'),): EAST,
        (ord('a'),): WEST,
        (ord('w'),): NORTH,
        (ord('s'),): SOUTH}


    def callback(obs_t, obs_tp1, action, rew, done, info):
        if done or rew > 0:
            print(f"done {done}, reward {rew}, info {info}")


    # You need to change this line in gym.play to use it nicely:
    # action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
    # change 0 -> 999  (or some other integer out of ToyMR action space)
    # Without this change gym.play will choose action 0 when no key is pressed.
    play(env, keys_to_action=keys_to_action, fps=10, callback=callback)
