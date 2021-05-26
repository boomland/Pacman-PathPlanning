# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

'''
It seems that we do not need this class. We can work only with MultiAgentSearchAgent.
You can check how works ReflexAgent for example. But in principle we neew only change
2 classes: MonteCarloTreeSearchAgent, ReinforcementLearningAgent
'''

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MCTSInterface(Agent, object):
    class Node(object):
        _visited_states = set()

        def __init__(self, game_state, parent=None, action=None):
            if parent is None:
                self._visited_states = set()
            self._visited_states.add(game_state.getPacmanPosition())

            self._game_state = game_state
            self._parent = parent
            # Store an action to get to this node from the parent
            self._action = action

            self._n_backpropagations = 0
            self._total_reward = 0


            self._children = []
            self._possible_actions = []
            if self._game_state.isLose():
                print(self._game_state.getLegalActions())
            for u in self._game_state.getLegalActions():
                self._possible_actions.append(u)

        # Derived class should implement this method
        def select(self):
            util.raiseNotDefined()

        # Derived class should implement this method
        def expand(self):
            util.raiseNotDefined()

        def backpropagate(self, reward):
            self._n_backpropagations += 1
            self._total_reward += reward
            if self._parent is not None:
                self._parent.backpropagate(reward)

        def get_game_state(self):
            return self._game_state.deepCopy()

        # Useful helper
        @property
        def _average_reward(self):
            if self._n_backpropagations == 0:
                raise Exception('Trying to calculate average reward without any backpropagations.')
            return self._total_reward / self._n_backpropagations

        @property
        def _best_child(self):
            best_childs = self._children[0]
            best_score = None
            for curr_child in self._children:
                if best_score is None or curr_child._average_reward > best_score:
                    best_score = curr_child._average_reward
                    best_childs = [curr_child]
                elif curr_child._average_reward == best_score:
                    best_childs.append(curr_child)
            return random.choice(best_childs)

        # Useful helper
        def choose_best_action(self):
            return self._best_child._action

    def __init__(self):
        self._max_mcts_iterations = 50
        self._max_simulation_iterations = 5

    def getAction(self, gameState):
        root = self.create_tree(gameState)
        for _ in range(self._max_mcts_iterations):
            selected_node = root.select()
            expanded_node = selected_node.expand()
            reward = self.simulate(expanded_node.get_game_state())
            expanded_node.backpropagate(reward)
        action = root.choose_best_action()
        # if action == 'Stop':
        #     for child in root._children:
        #         print "action '{}', reward = {}, n = {}".format(
        #                 child._action, child._average_reward, child._n_backpropagations)
        return action

    # Implement this in derived class
    def create_tree(self, gameState):
        util.raiseNotDefined()

    # Reimplement this in derived class if needed
    def simulate(self, gameState):
        # Do not simulate anything, just give the current score
        return gameState.getScore()


# Actually, this algorithm works bad, as in case of stopping
class MCTSAsInHW(MCTSInterface):
    """
      A MCTS agent chooses an action using Monte-Carlo Tree Search.
    """
    class Node(MCTSInterface.Node):
        def __init__(self, game_state, parent=None, action=None):
            super(self.__class__, self).__init__(game_state, parent, action)

        def select(self):
            if len(self._possible_actions) != 0 or len(self._children) == 0:
                return self
            return self._best_child.select()

        def expand(self):
            if len(self._children) != 0 and len(self._possible_actions) == 0:
                raise Exception("Couldn't select a node without possible children, bit with existing.")
            while len(self._possible_actions) != 0:
                action = self._possible_actions[0]
                self._possible_actions.pop(0)
                next_game_state = self._game_state.generatePacmanSuccessor(action)
                if next_game_state.getPacmanPosition() in self._visited_states:
                    continue
                expanded_child = self.__class__(
                    game_state=self._game_state.generatePacmanSuccessor(action),
                    parent=self,
                    action=action,
                )
                self._children.append(expanded_child)
                return expanded_child
            return self

    def __init__(self):
        super(self.__class__, self).__init__()

    def create_tree(self, game_state):
        return self.Node(game_state=game_state)

    def simulate(self, game_state):
        def dist_l1(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        def find_nearest_food(game_state):
            curr_pos = game_state.getPacmanPosition()
            food_grid = game_state.getFood()
            nearest = None
            for x in range(food_grid.width):
                for y in range(food_grid.height):
                    if not food_grid[x][y]:
                        continue
                    if nearest is None or \
                            dist_l1(curr_pos, (x, y)) < dist_l1(curr_pos, nearest):
                        nearest = (x, y)
            return nearest

        nearest_food = None
        n_iter = 0
        while not game_state.isWin() and not game_state.isLose() and \
              n_iter < self._max_simulation_iterations:
            if nearest_food is None:
                nearest_food = find_nearest_food(game_state)
            best_dist = None
            best_next_state = None
            for action in game_state.getLegalActions():
                next_state = game_state.generatePacmanSuccessor(action)
                curr_dist = dist_l1(next_state.getPacmanPosition(), nearest_food)
                if best_dist is None or curr_dist < best_dist:
                    best_dist = curr_dist
                    best_next_state = next_state
            game_state = best_next_state
            if game_state.getPacmanPosition() == nearest_food:
                nearest_food = None
            n_iter += 1
        return game_state.getScore()


class ReinforcementLearningAgent(MultiAgentSearchAgent):
    '''
    Agent base on Reinforcement Learning
    '''
    def getAction(self, gameState):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

# class MinimaxAgent(MultiAgentSearchAgent):
#     """
#       Your minimax agent (question 2)
#     """

#     def getAction(self, gameState):
#         """
#           Returns the minimax action from the current gameState using self.depth
#           and self.evaluationFunction.

#           Here are some method calls that might be useful when implementing minimax.

#           gameState.getLegalActions(agentIndex):
#             Returns a list of legal actions for an agent
#             agentIndex=0 means Pacman, ghosts are >= 1

#           gameState.generateSuccessor(agentIndex, action):
#             Returns the successor game state after an agent takes an action

#           gameState.getNumAgents():
#             Returns the total number of agents in the game
#         """
#         "*** YOUR CODE HERE ***"
#         util.raiseNotDefined()

# class AlphaBetaAgent(MultiAgentSearchAgent):
#     """
#       Your minimax agent with alpha-beta pruning (question 3)
#     """

#     def getAction(self, gameState):
#         """
#           Returns the minimax action using self.depth and self.evaluationFunction
#         """
#         "*** YOUR CODE HERE ***"
#         util.raiseNotDefined()

# class ExpectimaxAgent(MultiAgentSearchAgent):
#     """
#       Your expectimax agent (question 4)
#     """

#     def getAction(self, gameState):
#         """
#           Returns the expectimax action using self.depth and self.evaluationFunction

#           All ghosts should be modeled as choosing uniformly at random from their
#           legal moves.
#         """
#         "*** YOUR CODE HERE ***"
#         util.raiseNotDefined()

# def betterEvaluationFunction(currentGameState):
#     """
#       Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
#       evaluation function (question 5).

#       DESCRIPTION: <write something here so we know what you did>
#     """
#     "*** YOUR CODE HERE ***"
#     util.raiseNotDefined()

# # Abbreviation
# better = betterEvaluationFunction

