"""Module with utilities for Hanoi Tower env."""
import numpy as np
import pygame

BACKGROUND_COLOR = (255, 255, 255)
JUST_VISITED_COLOR = (0, 200, 50)
VISITED_OFTEN_COLOR = (0, 255, 0)
STATE_BORDER_COLOR = (0, 0, 0)
VERTEX_COLOR = (0, 0, 0)

SCALE = 25
A = np.array([0, 0])
B = np.array([1, 1])
C = np.array([-1, 1])
D = np.array([2, 0])

class HanoiHelper:
    """Helping function for Hanoi Tower env."""
    def __init__(self, n_disks):
        self.n_disks = n_disks
        self._all_states = self.generate_all_states()
        self._state_positions = {state: self.find_state_position(state) for
                                 state in self._all_states}
        self._position_to_states = {self.find_state_position(state): state for
                                 state in self._all_states}
        self._all_transitions = []
        self._all_transitions_pos = []
        self.generate_all_transitions()

    def to_base_3(self, num, n_disks):
        num_base_3 = ''
        while num > 0:
            num_base_3 = str(num % 3) + num_base_3
            num = num // 3
        if len(num_base_3) < n_disks:
            num_base_3 = '0' * (n_disks - len(num_base_3)) + num_base_3
        return num_base_3

    def generate_all_states(self):
        list_of_all_states = []
        for num in range(3 ** self.n_disks):
            state_str = self.to_base_3(num, self.n_disks)
            list_of_all_states.append(tuple([int(p) for p in state_str]))
        return list_of_all_states

    def generate_all_transitions(self):
        """Generates all possible transitions."""

        neighbours_vec = [B, -B, C, -C, D, -D]
        for state in self._all_states:
            for vec in neighbours_vec:
                state_pos = np.array(self._state_positions[state])
                new_pos = tuple((state_pos + vec * SCALE))
                if new_pos in list(self._state_positions.values()):
                    self._all_transitions_pos.append((state_pos, new_pos))
                    new_state = self._position_to_states[new_pos]
                    self._all_transitions.append((state, new_state))
        return self._all_transitions

    def find_state_position(self, state):
        """Calculates the position of the states on the states map."""
        pos = np.array([0, 0], dtype=np.float64)

        def change(fixed, other):
            if other == fixed:
                return other
            else:
                for i in range(3):
                    if i not in (fixed, other):
                        return i

        top_lvl = {0: A, 1: B, 2: C}
        curr_lvl = top_lvl
        for i, x in enumerate(reversed(state)):
            pos += curr_lvl[x] * 2 ** (self.n_disks - i - 1)
            new_curr_lvl = {i : curr_lvl[change(x, i)] for i in range(3)}
            curr_lvl = new_curr_lvl

        return tuple(pos * SCALE)

    def shift_state_position(self, state, center):
        return (np.array(self._state_positions[state]) + center).astype(int)

    def shift_point(self, point, center):
        return tuple((point + center).astype(int))

    def render_heatmap(self, visited_freq, resolution=400):
        """Renders hanoi heat map."""
        pygame.init() # pylint: disable=no-member

        resize_factor = 2 ** (self.n_disks-3)
        map_ = pygame.Surface((resolution * resize_factor, # pylint: disable=too-many-function-args
                               resolution * resize_factor * 0.6))
        map_.fill(BACKGROUND_COLOR)

        center = np.array([resolution * resize_factor // 2, SCALE])

        for edge in self._all_transitions_pos:
            pygame.draw.line(map_, STATE_BORDER_COLOR,
                             self.shift_point(edge[0], center),
                             self.shift_point(edge[1], center),
                             1)

        for state, _ in self._state_positions.items():
            mean_visit = np.mean(list(visited_freq.values()))
            alpha = min(0.5 * visited_freq[state] / mean_visit, 1)
            state_color = STATE_BORDER_COLOR
            if alpha > 0:
                state_color = tuple((1- alpha) * np.array(JUST_VISITED_COLOR) +
                                    alpha * np.array(VISITED_OFTEN_COLOR))
            pygame.draw.circle(map_, state_color,
                               self.shift_state_position(state, center),
                               7)

            myfont = pygame.font.SysFont('Helvetica', 8)
            def label_inverse(s):
                inv_label = ''
                for l in str(s):
                    inv_label = l + inv_label
                return  '(' + inv_label[1:-1] + ')'

            label = myfont.render(label_inverse(state), True, (255, 0, 0))
            label_rect = label.get_rect(
                center=self.shift_state_position(state, center
                                                 + np.array([0, 0])))
            map_.blit(label, label_rect)

        return pygame.surfarray.array3d(map_)
