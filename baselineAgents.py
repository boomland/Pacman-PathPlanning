from environment.util import manhattanDistance
from environment.game import Directions, Agent
import environment.util as util
import random

import eval_functions


class ReflexAgent(Agent):
    def __init__(self, evalFn='eval_v3'):
        super().__init__()
        self.evaluationFunction = getattr(eval_functions, evalFn)

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
        scores = [self.evaluationFunction(gameState.generateSuccessor(0, action))
                  for action in legalMoves if action != 'Stop']
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]
