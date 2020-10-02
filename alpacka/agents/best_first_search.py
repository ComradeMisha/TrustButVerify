"""Greedy agent for deterministic environments."""

import asyncio
import random
from collections import deque
from heapq import heappush, heappop

import gin
import numpy as np

from alpacka import data
from alpacka.agents import base
from alpacka.agents.deterministic_mcts import ScalarValueTraits
from alpacka.agents.deterministic_mcts import ScalarValueAccumulator
from alpacka.data import Request
from alpacka.data import RequestType
from alpacka.envs import TrainableModelEnv
from alpacka.utils import space as space_utils
from alpacka.utils.transformations import discount_cumsum


@gin.configurable
def neg_distance(node):
    return -node.distance


@gin.configurable
def index_without_auxiliary_loss(node):
    return node.value_acc.index_without_auxiliary_loss()


@gin.configurable
def bonus(node):
    return max([0] + node.bonus)


@gin.configurable
def value_plus_bonus(node):
    return np.mean(node.value_acc.get()) + max([0] + node.bonus)


@gin.configurable
def index(node):
    return node.value_acc.index()


@gin.configurable
def solved_then_auxiliary_then_bonus(node):
    return (
        node.solved, node.value_acc.auxiliary_loss, max([0] + node.bonus)
    )


@gin.configurable
def solved_then_index(node):
    return node.solved, node.value_acc.index()


class Fringe:
    """Priority queue of node-candidates for expansion."""
    def __init__(self, heuristic_fn):
        """Initializes Fringe

        Args:
            heuristic_fn: function mapping node -> utility.
        """
        self._heuristic_fn = heuristic_fn
        self._heap_queue = []
        self._nodes = set()

    def add(self, node):
        if node not in self._nodes:
            self._nodes.add(node)
            value = self._heuristic_fn(node)
            heappush(self._heap_queue, (-value, node))

    def get(self):
        _, node = heappop(self._heap_queue)
        self._nodes.remove(node)
        return node

    def empty(self):
        return len(self._heap_queue) == 0

    def __iter__(self):
        return iter(self._nodes)

    def __len__(self):
        return len(self._nodes)


class BestFirstSearchAgent(base.OnlineAgent):
    """Monte Carlo Tree Search for deterministic environments.

    Implements transposition tables (sharing value estimates between multiple
    tree nodes corresponding to the same environment state) and loop avoidance.
    """

    class _GraphNode:
        """Graph node, corresponding 1-1 to an environment state."""

        def __init__(
            self,
            value_acc,
            state=None,
            terminal=False,
            solved=False,
            distance=np.inf,
        ):
            self.value_acc = value_acc
            self.rewards = {}
            self.bonus = list()
            self.edges = {}  # {valid_action: GraphNode}
            self.state = state
            self.terminal = terminal
            self.solved = solved
            self.distance = distance  # upper bound
            self.shortest_path_predecessor = None
            self.shortest_path_preceding_action = None

        def __lt__(self, other):
            return False

        def __gt__(self, other):
            return False

    def __init__(
        self,
        gamma=0.99,
        n_nodes_per_step=10,
        expand_heuristic=neg_distance,
        top_level_heuristic=solved_then_index,
        top_level_epsilon=0.,
        bonus_quantile_threshold=None,
        value_traits_class=ScalarValueTraits,
        value_accumulator_class=ScalarValueAccumulator,
        model_class=None,
        ensemble_size=None,
        ensemble_mask_size=None,
        model_ensemble_size=None,
        model_ensemble_mask_size=None,
        render_rollout=False,
        bonus_queue_length=200,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._gamma = gamma
        self._n_nodes_per_step = n_nodes_per_step
        self._expand_heuristic = expand_heuristic
        self._top_level_heuristic = top_level_heuristic
        self._value_traits = value_traits_class()
        self._value_acc_class = value_accumulator_class
        self._bonus_queue_length = bonus_queue_length
        self._bonuses = deque([], maxlen=self._bonus_queue_length)
        self._state2node = {}
        self._fringe = self._get_empty_fringe()
        self._reachable_nodes = set()

        # If model_class is None, we use true environment as model.
        self._use_trainable_env = model_class is not None
        # We assume that model_class represents imperfect model.
        # However, it should work with perfect model as well.
        self._model_class = model_class
        self._model = None

        self._current_node = None
        self._render_rollout = render_rollout

        if bool(ensemble_size) != bool(ensemble_mask_size):
            raise ValueError('Invalid value function ensemble parameters.')
        self._use_ensembles = ensemble_size is not None
        self._ensemble_size = ensemble_size
        self._ensemble_mask_size = ensemble_mask_size

        if bool(model_ensemble_size) != bool(model_ensemble_mask_size):
            raise ValueError('Invalid model ensemble parameters.')
        self._use_model_ensembles = model_ensemble_size is not None
        self._model_ensemble_size = model_ensemble_size
        self._model_ensemble_mask_size = model_ensemble_mask_size

        self._top_level_epsilon = top_level_epsilon
        self._bonus_quantile_threshold = bonus_quantile_threshold

    def _initialize_graph_node(self, initial_value, state, done, solved):
        value_acc = self._value_acc_class(initial_value)
        new_node = self._GraphNode(
            value_acc,
            state=state,
            terminal=done,
            solved=solved,
        )
        # store newly initialized node in _state2node
        self._state2node[state] = new_node
        return new_node

    def _expand_graph_node(self, node):
        assert bool(node.rewards) == bool(node.edges)
        if (
            len(node.edges) > 0 or  # graph node is expanded already
            node.solved or
            node.terminal
        ):
            return

        # neighbours are ordered in the order of actions:
        # 0, 1, ..., _model.num_actions
        observations, rewards, dones, infos, states = \
            yield from self._model.predict_steps(
                node.state,
                list(space_utils.element_iter(self._model.action_space))
            )
        # solved = [info.get('solved', False) for info in infos]
        assert all([reward in (0, 1) for reward in rewards]), \
            'We assume that env is deterministic, and there are goal states ' \
            'obtaining which gives you reward=1 and ends episode. All other ' \
            'actions should give reward=0'
        solved = [reward == 1 for reward in rewards]
        node.bonus = [info.get('bonus', 0.) for info in infos]

        node.value_acc.add_bonus(max(node.bonus))
        self._bonuses.append(max(node.bonus))

        value_batch = yield Request(
            RequestType.AGENT_PREDICTION, np.array(observations)
        )

        for idx, action in enumerate(
            space_utils.element_iter(self._action_space)
        ):
            node.rewards[action] = rewards[idx]
            new_node = self._state2node.get(states[idx], None)
            if new_node is None:
                if dones[idx]:
                    child_value = self._value_traits.zero
                else:
                    [child_value] = value_batch[idx]
                new_node = self._initialize_graph_node(
                    child_value, states[idx], dones[idx], solved=solved[idx]
                )
            node.edges[action] = new_node
        self._update_from_node(node)

    def reset(self, env, observation):
        yield from super().reset(env, observation)

        if not self._use_trainable_env:
            self._model = TrainableModelEnv.wrap_perfect_env(env)
        else:
            if self._model is None:
                # Deferred construction. The model will be reused in all
                # subsequent calls to reset().
                self._model = self._model_class(modeled_env=env)
            if self._use_model_ensembles:
                self._model.set_global_index_mask(
                    self._model_ensemble_size, self._model_ensemble_mask_size
                )

        # 'reset' mcts internal variables: _state2node and _model
        self._bonuses = deque([], maxlen=self._bonus_queue_length)
        self._state2node = {}
        self._fringe = self._get_empty_fringe()
        self._reachable_nodes = set()
        if not self._use_trainable_env:
            state = self._model.clone_state()
        else:
            state = self._model.obs2state(observation)
        [[value]] = yield Request(
            RequestType.AGENT_PREDICTION, np.array([observation])
        )

        if self._use_ensembles:
            self._value_acc_class.set_global_index_mask(
                self._ensemble_size, self._ensemble_mask_size
            )

        # Initialize root.
        graph_node = self._initialize_graph_node(
            initial_value=value, state=state, done=False, solved=False
        )
        self._current_node = graph_node

    def _choose_action_based_on_novelty(self, node):
        bonus_threshold = np.quantile(
            self._bonuses, self._bonus_quantile_threshold
        )
        action, bonus = np.argmax(node.bonus), np.max(node.bonus)
        if bonus > bonus_threshold and random.random() > 0.5:
            return action
        else:
            return None

    def _select_action_top_level(self):
        action = None
        if self._bonus_quantile_threshold is not None:
            action = self._choose_action_based_on_novelty(self._current_node)
        if action is None:
            if len(self._reachable_nodes) == 1:
                # Corner case: all actions lead to current_node
                action = self._model.action_space.sample()
            elif random.random() < self._top_level_epsilon:
                self._current_node.value_acc.add_auxiliary(
                    -self._value_traits.avoid_history_coeff
                )
                action = self._model.action_space.sample()
            else:
                node = max(
                    self._reachable_nodes - {self._current_node},
                    key=self._top_level_heuristic
                )
                while node != self._current_node:
                    action = node.shortest_path_preceding_action
                    node = node.shortest_path_predecessor
                    assert isinstance(node, self._GraphNode)
        return action

    def _test_graph(self):
        for node in self._reachable_nodes:
            assert node.distance < np.inf
            if node != self._current_node:
                assert node.shortest_path_predecessor is not None
                assert node.shortest_path_preceding_action is not None

        for node in self._fringe:
            assert node in self._reachable_nodes

    def _update_from_node(self, node):
        # Iterate over nodes with incorrect distances, add appropriate nodes
        # to fringe and reachable nodes.
        assert node.distance < np.inf
        nodes_queue = deque([node])
        while nodes_queue:
            node = nodes_queue.pop()
            self._reachable_nodes.add(node)
            for action, neighbour in node.edges.items():
                if neighbour.distance > node.distance + 1:
                    neighbour.distance = node.distance + 1
                    neighbour.shortest_path_predecessor = node
                    neighbour.shortest_path_preceding_action = action
                    nodes_queue.appendleft(neighbour)
            if not node.edges:
                self._fringe.add(node)

    def _update_graph(self):
        for node in self._state2node.values():
            node.shortest_path_predecessor = None
            node.shortest_path_preceding_action = None
            node.distance = np.inf
        self._current_node.distance = 0
        self._fringe = self._get_empty_fringe()
        self._reachable_nodes = set()

        self._update_from_node(self._current_node)

    def _expand_graph(self):
        for _ in range(self._n_nodes_per_step):
            if self._fringe.empty():
                break
            node = self._fringe.get()
            yield from self._expand_graph_node(node)

    def act(self, observation):
        del observation
        self._current_node.value_acc.add_auxiliary(
            self._value_traits.avoid_history_coeff
        )
        self._update_graph()
        yield from self._expand_graph()
        action = self._select_action_top_level()
        info = self._get_node_info()
        self._current_node = self._current_node.edges[action]
        return action, info

    @asyncio.coroutine
    def _handle_env_feedback(self, agent_info, action, next_observation, reward,
                             done, env_info):
        """Handles model's mispredictions."""

        if not self._use_trainable_env:
            # We use perfect model, so there aren't any mispredictions
            # to handle.
            return
        root_parent = agent_info['node']
        true_state = self._model.obs2state(next_observation)
        solved = env_info.get('solved', False)

        # Correct mispredicted reward.
        root_parent.rewards[action] = reward

        if self._current_node.state != true_state:
            # self._model predicted wrong state, initialize new tree from
            # the true state
            new_node = self._state2node.get(true_state, None)
            if new_node is None:
                # True next state was not visited previously.
                # Initialize new GraphNode.
                if done:
                    value = self._value_traits.zero
                else:
                    # Batch stepper requires all requests submitted at the same
                    # time to have equal shape. The only other place, which
                    # sends requests, is self._expand_leaf() method, where
                    # `n_actions` observations are sent - so we do the same
                    # here.
                    n_actions = space_utils.max_size(self._model.action_space)
                    response = yield Request(
                        RequestType.AGENT_PREDICTION,
                        np.array([next_observation] * n_actions)
                    )
                    [value] = response[0]  # we ignore all other responses

                new_node = self._initialize_graph_node(
                    value, true_state, done, solved
                )
            # Correct mispredicted state in GraphNode, so we won't make
            # the same mistake again.
            root_parent.edges[action] = new_node
            self._current_node = new_node

        self._current_node.terminal = done

        self._current_node.solved = solved

    def postprocess_transitions(self, transitions):
        rewards = [transition.reward for transition in transitions]
        discounted_returns = discount_cumsum(rewards, self._gamma)

        for transition, discounted_return in zip(
                transitions, discounted_returns
        ):
            transition.agent_info.pop('node')
            transition.agent_info['discounted_return'] = discounted_return
        return transitions

    def _get_empty_fringe(self):
        return Fringe(self._expand_heuristic)

    @staticmethod
    def compute_metrics(episodes):
        def episode_info(key):
            for episode in episodes:
                yield from episode.transition_batch.agent_info[key]

        def count_distinct_observations(episode):
            return np.unique(
                (
                    list(episode.transition_batch.observation) +
                    [episode.transition_batch.next_observation[-1]]
                ),
                axis=0
            ).shape[0]

        distinct_observations = [
            count_distinct_observations(episode) for episode in episodes
        ]

        final_fringe_sizes = [
            episode.transition_batch.agent_info['fringe_size'][-1]
            for episode in episodes
        ]

        return {
            'graph_size_max': np.max(list(episode_info('graph_size'))),

            'reachable_nodes_max': np.max(
                list(episode_info('reachable_nodes'))),
            'reachable_nodes_mean': np.mean(
                list(episode_info('reachable_nodes'))),

            'fringe_size_max': np.max(
                list(episode_info('fringe_size'))),
            'fringe_size_mean': np.mean(
                list(episode_info('fringe_size'))),

            'final_fringe_size_mean': np.mean(final_fringe_sizes),
            'final_fringe_size_min': np.min(final_fringe_sizes),

            'distinct_observations_mean': np.mean(distinct_observations),
            'distinct_observations_max': np.max(distinct_observations),

            'one_step_bonus_mean': np.mean(
                list(episode_info('one_step_bonus'))
            ),
            'value_mean': np.mean(list(episode_info('value'))),
        }

    def _compute_trainable_env_info(self):
        transitions = self._model.interesting_transitions()
        info = {}
        for t_name, (obs, action) in transitions.items():
            next_obs, _, _ = yield from self._model.predict_step(obs, action)
            info[t_name] = next_obs

        return info

    def _compute_graph_metrics(self):
        return {
            'graph_size': len(self._state2node),
            'reachable_nodes': len(self._reachable_nodes),
            'fringe_size': len(self._fringe),
        }

    def _compute_node_info(self, node):
        value = node.value_acc.target()

        targets = np.array([
            child.value_acc.target() for child in node.edges.values()
        ])
        bonus = np.array(node.bonus)

        node_info = {
            'value': value,
            # 'initial_vf': network_q_values,
            'accumulator_target': targets,
            # 'index': tree_q_values,
            'one_step_bonus': bonus,
        }

        if self._use_ensembles:
            statistics = np.array([
                child.value_acc.ensemble_stats()
                for child in node.edges.values()
            ])
            values_mean, values_std, aux_loss = statistics.T

            node_info.update({
                'vf_ens_mean': values_mean,
                'vf_ens_std': values_std,
                'avoid_history': aux_loss,
            })

        return node_info

    def _render_rollout_info(self, node):
        """Updates agent info for trainable model logging purposes."""
        info = dict()

        info['children_observations'] = np.stack([
            self._model.state2obs(child.state)
            for child in node.edges.values()
        ])
        info['children_rewards'] = np.fromiter(
            node.rewards.values(), dtype=np.float
        )
        info['children_dones'] = np.array(
            [child.terminal for child in node.edges.values()],
            dtype=float
        )

        return info

    @staticmethod
    def network_signature(observation_space, action_space):
        del action_space
        return data.NetworkSignature(
            input=space_utils.signature(observation_space),
            output=data.TensorSignature(shape=(1,)),
        )

    def _get_node_info(self):
        info = {'node': self._current_node}
        info.update(self._compute_node_info(self._current_node))
        info.update(self._compute_graph_metrics())
        if self._render_rollout:
            info.update(self._render_rollout_info(self._current_node))
        return info


def td_backup(node, action, value, gamma):
    if action is None:
        return value
    return node.rewards[action] + gamma * value
