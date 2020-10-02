"""Monte Carlo Tree Search for deterministic environments."""

import asyncio
import random
from collections import deque

import gin
import numpy as np

from alpacka import data
from alpacka.agents import base
from alpacka.data import Request
from alpacka.data import RequestType
from alpacka.envs import TrainableModelEnv
from alpacka.utils import space as space_utils
from alpacka.utils.transformations import discount_cumsum


class ValueTraits:
    """Value traits base class.

    Defines constants for abstract value types.
    """

    zero = None
    dead_end = None


@gin.configurable
class ScalarValueTraits(ValueTraits):
    """Scalar value traits.

    Defines constants for the most basic case of scalar values.
    """

    zero = 0.0

    def __init__(self, dead_end_value=-2.0, avoid_history_coeff=0.0):
        self.dead_end = dead_end_value
        self.avoid_history_coeff = avoid_history_coeff


class ValueAccumulator:
    """Value accumulator base class.

    Accumulates abstract values for a given node across multiple MCTS passes.
    """

    def __init__(self, value):
        # Creates and initializes with typical add
        self._initial_value = value
        self.add(value)

    def add(self, value):
        """Adds an abstract value to the accumulator.

        Args:
            value: Abstract value to add.
        """
        raise NotImplementedError

    def get(self):
        """Returns the accumulated abstract value for backpropagation.

        May be non-deterministic.
        """
        raise NotImplementedError

    def index(self):
        """Returns an index for selecting the best node."""
        raise NotImplementedError

    def target(self):
        """Returns a target for value function training."""
        raise NotImplementedError

    def count(self):
        """Returns the number of accumulated values."""
        raise NotImplementedError

    def initial_value(self):
        return self._initial_value


@gin.configurable
class ScalarValueAccumulator(ValueAccumulator):
    """Scalar value accumulator.

    Calculates a mean over accumulated values and returns it as the
    backpropagated value, node index and target for value network training.
    """

    def __init__(self, value, alpha=1., beta=0.):
        self._alpha = alpha
        self._beta = beta
        self._sum = 0.0
        self._bonus_sum = 0.0
        self._count = 0
        self._bonus_count = 0
        self.auxiliary_loss = 0
        super().__init__(value)

    def add(self, value):
        self._sum += value
        self._count += 1

    def add_auxiliary(self, value):
        self.auxiliary_loss += value

    def add_bonus(self, value):
        self._bonus_sum += value
        self._bonus_count += 1

    def get(self):
        return self._alpha * self._sum / self._count

    def get_bonus(self):
        if self._bonus_count == 0:
            return 0
        else:
            return self._beta * self._bonus_sum / self._bonus_count

    def index_without_auxiliary_loss(self):
        return self.get() + self.get_bonus()

    def index(self):
        return self.index_without_auxiliary_loss() + self.auxiliary_loss

    def target(self):
        return self.get()

    def count(self):
        return self._count


@gin.configurable
class EnsembleValueAccumulator(ScalarValueAccumulator):
    """Ensemble value accumulator

    Stores one-dimensional numpy array of values. It is compatible with
    EnsembleValueTraits.

    In some initial cases the stored array is of length 1 - like the length
    of EnsembleValueTraits. That's the reason for using _safe_mask function.
    But at the point, when we .add() a properly sized array of values,
    the stored array is broadcasted to the number of ensembles.
    """

    _index_mask = None
    _ensemble_size = 1

    @classmethod
    def set_global_index_mask(cls, ensemble_size, n_ensembles_per_episode):
        """Specifies indices of ensembles used by all accumulators globally."""
        cls._index_mask = np.random.choice(
            ensemble_size, n_ensembles_per_episode, replace=False
        )
        cls._ensemble_size = ensemble_size

    def __init__(self, value, alpha=1., beta=0., kappa=3):
        if np.isscalar(value):
            value = np.full((self._ensemble_size,), value)
        assert value.shape == (self._ensemble_size,)
        super().__init__(value, alpha, beta)
        self._kappa = kappa

    def index_without_auxiliary_loss(self):
        vals = np.take(self.get(), self._index_mask)
        vf_bonus = self._kappa * np.std(vals)
        return np.mean(vals) + vf_bonus + self.get_bonus()

    def ensemble_stats(self):
        vals = np.take(super().target(), self._index_mask)
        return np.mean(vals), np.std(vals), self.auxiliary_loss

    def target(self):
        vals = np.take(super().target(), self._index_mask)
        return np.mean(vals)

    def initial_value(self):
        vals = np.take(super().initial_value(), self._index_mask)
        return np.mean(vals)


class GraphNode:
    """Graph node, corresponding 1-1 to an environment state.

    Accumulates value across multiple passes through the same environment state.
    """

    def __init__(
        self,
        value_acc,
        state=None,
        terminal=False,
        solved=False,
    ):
        self.value_acc = value_acc
        self.rewards = {}
        self.bonus = {}
        self.edges = {}  # {valid_action: GraphNode}
        self.state = state
        self.terminal = terminal
        self.solved = solved


class TreeNode:
    """Node in the search tree, corresponding many-1 to GraphNode.

    Stores children, and so defines the structure of the search tree. Many
    TreeNodes can point to the same GraphNode, because multiple paths from the
    root of the search tree can lead to the same environment state.
    """

    def __init__(self, node):
        self.node = node
        self.children = {}  # {valid_action: TreeNode}
        self.visit_count = 0

    @property
    def rewards(self):
        return self.node.rewards

    @property
    def bonus(self):
        return self.node.bonus

    @property
    def value_acc(self):
        return self.node.value_acc

    @property
    def state(self):
        return self.node.state

    @state.setter
    def state(self, state):
        self.node.state = state

    def expanded(self):
        return bool(self.children)

    def is_leaf(self):
        return not self.expanded()

    @property
    def terminal(self):
        return self.node.terminal

    @property
    def solved(self):
        return self.node.solved

    @terminal.setter
    def terminal(self, terminal):
        self.node.terminal = terminal


class DeterministicMCTSAgent(base.OnlineAgent):
    """Monte Carlo Tree Search for deterministic environments.

    Implements transposition tables (sharing value estimates between multiple
    tree nodes corresponding to the same environment state) and loop avoidance.
    """

    def __init__(
        self,
        gamma=0.99,
        n_passes=10,
        avoid_loops=True,
        value_traits_class=ScalarValueTraits,
        value_accumulator_class=ScalarValueAccumulator,
        model_class=None,
        ensemble_size=None,
        ensemble_mask_size=None,
        model_ensemble_size=None,
        model_ensemble_mask_size=None,
        render_rollout=False,
        top_level_epsilon=0.,
        avoid_termination=False,
        bonus_quantile_threshold=None,
        bonus_queue_length=2000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._gamma = gamma
        self._n_passes = n_passes
        self._avoid_loops = avoid_loops
        self._value_traits = value_traits_class()
        self._value_acc_class = value_accumulator_class
        self._bonus_queue_length = bonus_queue_length
        self._bonuses = deque([], maxlen=self._bonus_queue_length)
        self._state2node = {}

        # If model_class is None, we use true environment as model.
        self._use_trainable_env = model_class is not None
        # We assume that model_class represents imperfect model.
        # However, it should work with perfect model as well.
        self._model_class = model_class
        self._model = None

        self._root = None
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
        self._avoid_termination = avoid_termination
        self._bonus_quantile_threshold = bonus_quantile_threshold

    def run_mcts_pass(self):
        # search_path = list of tuples (node, action)
        # leaf does not belong to search_path (important for not double counting
        # its value)
        leaf, search_path = self._traverse()
        value, bonus = yield from self._expand_leaf(leaf)
        self._backpropagate(search_path, value, bonus)

    def _traverse(self):
        node = self._root
        seen_states = set()
        search_path = []
        # new_node is None iff node has no unseen children, i.e. it is Dead
        # End
        while node is not None and node.expanded():
            seen_states.add(node.state)
            # INFO: if node Dead End, (new_node, action) = (None, None)
            # INFO: _select_child can SAMPLE an action (to break tie)
            states_to_avoid = seen_states if self._avoid_loops else set()
            new_node, action = self._select_child(
                node, states_to_avoid, strict=True
            )
            search_path.append((node, action))
            node = new_node
        # at this point node represents a leaf in the tree (and is None for Dead
        # End). node does not belong to search_path.
        return node, search_path

    def _backpropagate(self, search_path, value, bonus):
        # Note that a pair
        # (node, action) can have the following form:
        # (Terminal node, None),
        # (Dead End node, None),
        # (TreeNode, action)
        for node, action in reversed(search_path):
            # returns value if action is None
            value = td_backup(node, action, value, self._gamma)
            bonus = self._gamma * bonus
            node.visit_count += 1
            node.value_acc.add(value)
            node.value_acc.add_bonus(bonus)

    def _initialize_graph_node(self, initial_value, state, done, solved):
        value_acc = self._value_acc_class(initial_value)
        new_node = GraphNode(
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
        if node.edges:
            return  # graph node is expanded already

        # neighbours are ordered in the order of actions:
        # 0, 1, ..., _model.num_actions
        observations, rewards, dones, infos, states = \
            yield from self._model.predict_steps(
                node.state,
                list(space_utils.element_iter(self._model.action_space))
            )
        solved = [info.get('solved', False) for info in infos]
        node.bonus = [info.get('bonus', 0.) for info in infos]
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

    def _expand_leaf(self, leaf):
        if leaf is None:  # Dead End
            return self._value_traits.dead_end, self._value_traits.zero

        leaf.visit_count += 1

        if leaf.terminal:  # Terminal state
            return self._value_traits.zero, self._value_traits.zero

        yield from self._expand_graph_node(leaf.node)
        leaf.children = {
            action: TreeNode(neighbor)
            for action, neighbor in leaf.node.edges.items()
        }
        return leaf.value_acc.get(), max(leaf.node.bonus)

    def _child_index(self, parent, action, select_method):
        accumulator = parent.children[action].value_acc
        if select_method == 'tree_greedy':
            value = accumulator.index()
        elif select_method == 'default_policy':
            value = accumulator.initial_value()
            return td_backup(parent, action, value, self._gamma)
        else:
            raise ValueError(
                f'Unrecognized child index evaluation method: '
                f'{select_method}'
            )

        return td_backup(parent, action, value, self._gamma)

    def _allowed_actions(self, node, states_to_avoid, avoid_terminal):
        def is_allowed(action, child):
            if child.state in states_to_avoid:
                return False
            elif (
                avoid_terminal and
                node.rewards[action] <= 0. and
                child.terminal
            ):
                return False
            else:
                return True

        return [
            action for action, child in node.children.items()
            if is_allowed(action, child)
        ]

    def _rate_children(
        self, node, states_to_avoid, avoid_terminal, select_method='tree_greedy'
    ):
        assert self._avoid_loops or len(states_to_avoid) == 0
        allowed_actions = self._allowed_actions(
            node, states_to_avoid, avoid_terminal=avoid_terminal)
        return [
            (self._child_index(node, action, select_method), action)
            for action in allowed_actions
        ]

    # Select the child with the highest score
    def _select_child(self, node, states_to_avoid, strict=True):
        values_and_actions = self._rate_children(
            node, states_to_avoid, avoid_terminal=self._avoid_termination)
        if not values_and_actions:
            if strict:
                return None, None
            values_and_actions = self._rate_children(
                node, {}, avoid_terminal=False)
            assert len(values_and_actions) > 0
        (max_value, _) = max(values_and_actions)
        argmax = [
            action for value, action in values_and_actions if value == max_value
        ]
        # INFO: here can be sampling
        if len(argmax) > 1:
            action = np.random.choice(argmax)
        else:
            action = argmax[0]
        return node.children[action], action

    def _choose_action_based_on_novelty(self, node):
        bonus_threshold = np.quantile(
            self._bonuses, self._bonus_quantile_threshold
        )
        action, bonus = np.argmax(node.bonus), np.max(node.bonus)
        if bonus > bonus_threshold and random.random() > 0.5:
            return action
        else:
            return None

    def _select_child_top_level(self, node, states_to_avoid):
        action = None
        select_child_info = {
            'random_action': False,
            'novelty_based_action': False,
            'bonus_based_action' : False,
        }
        if self._bonus_quantile_threshold is not None:
            action = self._choose_action_based_on_novelty(node)
            if action is not None:
                select_child_info['novelty_based_action'] = True

        if action is None:
            if random.random() > self._top_level_epsilon:
                _, action = self._select_child(
                    node, states_to_avoid, strict=False
                )
            else:
                select_child_info['random_action'] = True
                self._root.value_acc.add_auxiliary(
                    -self._value_traits.avoid_history_coeff
                )
                action = np.random.choice(list(node.children.keys()))
        new_root = node.children[action]
        return new_root, action, select_child_info

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
        self._root = TreeNode(graph_node)

    def act(self, observation):
        # perform MCTS passes.
        # each pass = tree traversal + leaf evaluation + backprop
        self._root.value_acc.add_auxiliary(
            self._value_traits.avoid_history_coeff
        )
        for _ in range(self._n_passes):
            yield from self.run_mcts_pass()
        info = {'node': self._root}
        info.update(self._compute_node_info(self._root))
        info.update(self._compute_tree_metrics(self._root))
        if self._render_rollout:
            info.update(self._render_rollout_info(self._root))

        states_to_avoid = set()

        # INFO: possible sampling for exploration
        self._root, action, select_child_info = self._select_child_top_level(
            self._root, states_to_avoid
        )
        info.update(self._compute_new_node_info(self._root))
        info.update(select_child_info)
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
        if root_parent.rewards[action] != reward:
            root_parent.node.rewards[action] = reward

        if self._root.state != true_state:
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
            root_parent.node.edges[action] = new_node
            self._root = TreeNode(new_node)

        if self._root.terminal != done:
            self._root.node.terminal = done

        if self._root.solved != solved:
            self._root.node.solved = solved

    def postprocess_transitions(self, transitions):
        rewards = [transition.reward for transition in transitions]
        discounted_returns = discount_cumsum(rewards, self._gamma)

        for transition, discounted_return in zip(
                transitions, discounted_returns
        ):
            transition.agent_info.pop('node')
            transition.agent_info['discounted_return'] = discounted_return
        return transitions

    @staticmethod
    def compute_metrics(episodes):
        def episode_info(key):
            for episode in episodes:
                yield from episode.transition_batch.agent_info[key]

        def entropy(probs):
            def plogp(p):
                # If out this case to avoid log(0).
                return p * np.log(p) if p else 0

            return -np.sum([plogp(p) for p in probs])

        return {
            'one_step_bonus_mean': np.mean(
                list(episode_info('one_step_bonus'))
            ),
            'value_mean': np.mean(list(episode_info('value'))),
            'depth_mean': np.mean(list(episode_info('depth_mean'))),
            'depth_max': max(episode_info('depth_max')),
            'entropy_mean': np.mean(
                list(map(entropy, episode_info('visits')))
            ),
        }

    def _compute_trainable_env_info(self):
        transitions = self._model.interesting_transitions()
        info = {}
        for t_name, (obs, action) in transitions.items():
            next_obs, _, _ = yield from self._model.predict_step(obs, action)
            info[t_name] = next_obs

        return info

    def _compute_tree_metrics(self, root):
        def generate_leaf_depths(node, depth):
            if node.is_leaf:
                yield depth
            for child in node.children.values():
                yield from generate_leaf_depths(child, depth + 1)

        depths = list(generate_leaf_depths(root, 0))
        return {
            'depth_mean': sum(depths) / len(depths),
            'depth_max': max(depths),
        }

    def _compute_node_info(self, node):
        value = node.value_acc.target()
        action_counts = np.array([
            child.visit_count for child in node.children.values()
        ])
        if np.sum(action_counts) == 0:
            # This is rare case when our agent thinks it is stuck in the state,
            # i.e. no matter what action agent takes, it will move to the same
            # state. In this case, if avoid_loops is set to True, agent will not
            # visit any state from current one.
            action_histogram = action_counts
        else:
            action_histogram = action_counts / np.sum(action_counts)
        tree_q_values = np.array(
            self._rate_children(
                node, {}, avoid_terminal=False, select_method='tree_greedy'
            )
        )[:, 0]
        # Method _rate_children returns list of [[value, action]].
        # Only values are needed here.
        network_q_values = np.array(
            self._rate_children(
                node, {}, avoid_terminal=False, select_method='default_policy'
            )
        )[:, 0]
        targets = np.array([
            child.value_acc.target() for child in node.children.values()
        ])
        bonus = np.array(node.bonus)

        node_info = {
            'value': value,
            'visits': action_histogram,
            'initial_vf': network_q_values,
            'target/get': targets,
            'index': tree_q_values,
            'one_step_bonus': bonus,
        }

        if self._use_ensembles:
            statistics = np.array([
                child.value_acc.ensemble_stats()
                for child in node.children.values()
            ])
            values_mean, values_std, aux_loss = statistics.T

            node_info.update({
                'vf_ens_mean': values_mean,
                'vf_ens_std': values_std,
                'avoid_history': aux_loss,
            })

        return node_info

    @staticmethod
    def _compute_new_node_info(node):
        return {'mcts_visits': node.visit_count}

    def _render_rollout_info(self, node):
        """Updates agent info for trainable model logging purposes."""
        info = dict()

        info['children_observations'] = np.stack([
            self._model.state2obs(child.state)
            for child in node.children.values()
        ])
        info['children_rewards'] = np.fromiter(
            node.rewards.values(), dtype=np.float
        )
        info['children_dones'] = np.array(
            [child.terminal for child in node.children.values()],
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


def td_backup(node, action, value, gamma):
    if action is None:
        return value
    return node.rewards[action] + gamma * value
