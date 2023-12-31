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
        self.actions = state.get_actions()
        self.leaf = True
        self.state_dict = state_dict
        self.isterminal = state.is_terminal()
        self.actor = state.actor()
        self.offset = 1 if self.actor == 0 else -1

    def add_child(self, action, state):
        child_node = Node(state, self.state_dict, parent=self, action=action)
        self.children.append(child_node)
        return child_node
    
    def best_child(self, c_param=math.sqrt(2)):
        choices_weights = [
            self.offset * (child.wins/child.visits) + c_param * math.sqrt((2*math.log(self.visits)/child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index((max(choices_weights)))]
    
    def rollout_policy(self, possible_moves):
        return random.choice(possible_moves)
    
    def update(self, result):
        self.visits += 1
        self.wins += result

def select_node(node, visited_nodes):
    while not node.isterminal:
        if node.leaf:
            temp = expand_node(node)
            visited_nodes.append(node)
            return temp
        else:
            node = node.best_child()
        visited_nodes.append(node)
    return node

def expand_node(node):
    for action in node.actions:
        next_state = node.state.successor(action)
        node.add_child(action, next_state)
        
    node.leaf = False
    return random.choice(node.children)

def simulate_random_game(node):
    current_state = node.state
    while not current_state.is_terminal():
        possible_moves = current_state.get_actions()
        action = node.rollout_policy(possible_moves)
        current_state = current_state.successor(action)
    return current_state.payoff()

def backpropagate(result, visited_nodes):
    for node in visited_nodes:
        node.update(result)

def mcts(state_dict, root_state, time_limit):
    if(root_state in state_dict):
        root_node = state_dict[root_state]
    else: 
        root_node = Node(root_state, state_dict)
        state_dict[root_state] = root_node
    end_time = time.time() + time_limit

    while time.time() < end_time:
        visited_nodes = []
        selected_node = select_node(root_node, visited_nodes)
        result = simulate_random_game(selected_node)
        backpropagate(result, visited_nodes)
    
    maxPayoff = -1 * float("inf")
    maxAction = None

    for i in range(len(root_node.children)):
        child = root_node.children[i]
        payoff = root_node.offset * (child.wins/child.visits)
        if payoff > maxPayoff:
            maxPayoff = payoff
            maxAction = child.action

    return maxAction

def mcts_policy(time_limit):
    state_dict = dict()

    def policy(state):
        return mcts(state_dict, state, time_limit)
    return policy