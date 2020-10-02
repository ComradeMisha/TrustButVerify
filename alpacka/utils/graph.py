"""Utilities for graph creation and calculation of shortest distances."""

from collections import defaultdict, deque


class Graph:
    """Simple graph structure for calculation of distances between nodes."""
    def __init__(self):
        self.edges = defaultdict(list)
        self.edges_transposed = defaultdict(list)

    def add_edge(self, node1, node2):
        self.edges[node1].append(node2)
        self.edges_transposed[node2].append(node1)

    @staticmethod
    def _calc_distances(node, edges):
        distances = {node: 0}
        if node not in edges:
            return distances
        nodes_queue = deque([node])
        while nodes_queue:
            node = nodes_queue.pop()
            for neighbour in edges[node]:
                if neighbour not in distances:
                    distances[neighbour] = distances[node] + 1
                    nodes_queue.appendleft(neighbour)
        return distances

    def calc_distances_from(self, start_node):
        """Calculate distances from node.

        Args:
            start_node - graph node
        Returns:
            dict(graph node: distance from start_node)
        """
        return self._calc_distances(start_node, self.edges)

    def calc_distances_to(self, goal_node):
        """Calculate distances to node.

        Args:
            node - graph node
        Returns:
            dict(graph node: distance to goal_node)
        """
        return self._calc_distances(goal_node, self.edges_transposed)


def calc_distance_to_closest_goal(goal_nodes, graph):
    """Calculate distance to closest goal

    For each node in graph calculate shortest distance to one of the goal
    states.

    Args:
        goal_nodes: nodes from graph.
        graph (Graph): graph in which to search.
    Returns:
        dict(graph node: distance to closest goal)
    """
    distances = dict()
    for goal in goal_nodes:
        distances_to_goal = graph.calc_distances_to(goal)
        for state, distance in distances_to_goal.items():
            if (
                state not in distances or
                distances[state] > distance
            ):
                distances[state] = distance
    return distances


def generate_env_state_space_graph_and_goal_states(env):
    """Generate state-space graph and goal states."""
    env.reset()
    graph = Graph()
    state = env.clone_state()
    state_queue = deque([state])
    goal_states = set()
    while state_queue:
        state = state_queue.pop()
        for action in range(env.action_space.n):
            env.restore_state(state)
            _, reward, done, _ = env.step(action)
            neighbour = env.clone_state()
            graph.add_edge(state, neighbour)
            if (
                neighbour not in graph.edges and
                not done
            ):
                state_queue.append(neighbour)
            if reward != 0:
                assert reward == 1
                goal_states.add(neighbour)
    env.reset()
    return graph, goal_states


if __name__ == '__main__':
    # run with
    # python -m alpacka.utils.graph
    from alpacka.envs import ToyMR
    from alpacka.envs.toy_mr import ToyMRMaps

    # Test on simple graph
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 2)
    print('distances to')
    for i in range(5):
        print(i, g.calc_distances_to(i))

    print('distances from')
    for i in range(5):
        print(i, g.calc_distances_from(i))

    print('shortest distance to one of the goals (goals: 1 and 4)',
          calc_distance_to_closest_goal([1, 4], g)
          )

    # How to use it with env
    # env = Hanoi(n_disks=4)
    env = ToyMR(map_file=ToyMRMaps.FULL_MAP)
    graph, goal_states = generate_env_state_space_graph_and_goal_states(env)
    print('Graph size', len(graph.edges))
    print('Nuber of goal states', len(goal_states))
    # assert len(goal_states) == 1
    # goal_state = list(goal_states)[0]
    env.reset()
    start_state = env.clone_state()
    distance_from_start = graph.calc_distances_from(start_state)
    distance_to_goal = calc_distance_to_closest_goal(goal_states, graph)
    print(distance_from_start.values())
