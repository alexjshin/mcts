import random
import time
import math

class Node:
    def __init__(self, state, state_dict, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 1
        self.untried_actions = set(state.get_actions())
        self.isterminal = state.is_terminal()
        self.actor = state.actor()
        self.offset = 1 if self.actor == 0 else -1
        self.state_dict = state_dict

    def add_child(self, action, state):
        child_node = Node(state, self.state_dict, parent=self, action=action)
        self.children.append(child_node)
        self.untried_actions.remove(action)
        return child_node

    def best_child(self, c_param=math.sqrt(2)):
        choices_weights = [
            self.offset * (child.wins/child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def rollout_policy(self, possible_moves):
        return random.choice(possible_moves)

    def update(self, result):
        self.visits += 1
        self.wins += result

def select_node(node):
    while not node.isterminal:
        if node.untried_actions:
            return expand_node(node)
        else:
            node = node.best_child()
    return node

def expand_node(node):
    action = random.choice(list(node.untried_actions))
    next_state = node.state.successor(action)
    return node.add_child(action, next_state)

def simulate_random_game(node):
    current_state = node.state
    while not current_state.is_terminal():
        possible_moves = current_state.get_actions()
        action = node.rollout_policy(possible_moves)
        current_state = current_state.successor(action)
    return current_state.payoff()

def backpropagate(node, result):
    while node is not None:
        node.update(result)
        node = node.parent

def mcts(root_state, time_limit, state_dict):
    if root_state in state_dict:
        root_node = state_dict[root_state]
    else:
        root_node = Node(root_state, state_dict)
        state_dict[root_state] = root_node
    end_time = time.time() + time_limit

    while time.time() < end_time:
        selected_node = select_node(root_node)
        result = simulate_random_game(selected_node)
        backpropagate(selected_node, result)

    return root_node.best_child(c_param=0).action

def mcts_policy(time_limit):
    state_dict = dict()
    def policy(state):
        return mcts(state, time_limit, state_dict)
    return policy
