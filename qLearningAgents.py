from environment.game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from collections import defaultdict

import random, math
import environment.util as util


class QLearningAgent(ReinforcementAgent):
    """
        Q-Learning Agent
        Functions you should fill in:
            - computeValueFromQValues
            - computeActionFromQValues
            - getQValue
            - getAction
            - update
        Instance variables you have access to
            - self.epsilon (exploration prob)
            - self.alpha (learning rate)
            - self.discount (discount rate)
        Functions you should use
            - self.getLegalActions(state)
                which returns legal actions for a state
    """

    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.q_values = defaultdict(float)  # (state, action) -> q_value

    def is_legal(self, legal_actions):
        if len(legal_actions) <= 1:
            return False
        else:
            return True

    def getQValue(self, state, action):
        """
            Returns Q(state,action)
            Should return 0.0 if we have never seen a state
            or the Q node value otherwise
        """
        return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
            Returns max_action Q(state,action)
            where the max is over legal actions.
        """
        legal_actions = self.getLegalActions(state)
        q_val = 0.0
        if legal_actions:
            q_val = max([self.getQValue(state, action) for action in legal_actions])
        return q_val

    def computeActionFromQValues(self, state):
        """
            Compute the best action to take in a state.
        """
        legal_actions = self.getLegalActions(state)
        best_action = None
        if legal_actions:
            best_q_value = self.computeValueFromQValues(state)
            best_value_actions = [action for action in legal_actions if \
                                  self.getQValue(state, action) == best_q_value]
            best_action = random.choice(best_value_actions)
        return best_action

    def getAction(self, state):
        """
            Compute the action to take in the current state.  With
            probability self.epsilon, we should take a random action and
            take the best policy action otherwise.
        """
        legal_actions = self.getLegalActions(state)
        action = None
        if legal_actions:
            if util.flipCoin(self.epsilon):
                action = random.choice(legal_actions)
            else:
                action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
            The parent class calls this to observe a
            state = action => nextState and reward transition.
        """
        sa = (state, action)
        next_best_action = self.getPolicy(nextState)
        sa_next = (nextState, next_best_action)
        old_q_value = self.getQValue(*sa)
        next_q_value = self.getQValue(*sa_next)
        self.q_values[sa] = old_q_value + self.alpha * (reward + self.discount * next_q_value - old_q_value)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.1, gamma=0.85, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
                python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
         ApproximateQLearningAgent
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
            Should return Q(state,action) = w * featureVector
            where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        q_value = sum([feat * self.weights[feat] for feat in features.values()])
        return q_value

    def computeActionFromQValues(self, state):
        """
            Compute the best action to take in a state.
        """
        legal_actions = self.getLegalActions(state)
        best_action = None
        if legal_actions:
            best_q_value = self.computeValueFromQValues(state)
            best_value_actions = [action for action in legal_actions if \
                                  self.getQValue(state, action) == best_q_value]
            best_action = random.choice(best_value_actions)
        return best_action

    def computeValueFromQValues(self, state):
        """
            Returns max_action Q(state,action)
            where the max is over legal actions.
        """
        legal_actions = self.getLegalActions(state)
        q_value = 0.0
        if legal_actions:
            for action in legal_actions:
                q_value = max(q_value, self.getQValue(state, action))
        return q_value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        features = self.featExtractor.getFeatures(state, action)
        for feat in features.keys():
            sa = (state, action)
            next_best_action = self.getPolicy(nextState)
            sa_next = (nextState, next_best_action)
            old_q_value = self.getQValue(*sa)
            next_q_value = self.getQValue(*sa_next)
            self.weights[feat] = self.weights[feat] + \
                                 self.alpha * (reward + self.discount * next_q_value - old_q_value) * \
                                 features[feat]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print(self.weights.keys()[10][0].generatePacmanSuccessor(self.weights.keys()[10][1]))
            pass