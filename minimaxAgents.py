from environment.util import manhattanDistance
from environment.game import Directions, Agent
import environment.util as util
import random
import eval_functions
from baselineAgents import ReflexAgent


import numpy as np
from types import SimpleNamespace


class MultiAgentSearchAgent(ReflexAgent):
    def __init__(self, evalFn='eval_v3', depth='4'):
        super().__init__(evalFn=evalFn)
        self.depth = int(depth)


def minimax_func(state, cur_depth, cur_agent, context):
  if (cur_depth == context.max_depth * context.n_agents) or state.isWin() or state.isLose():
    return context.evaluationFunction(state), None

  if cur_agent == 0:
    # Pacman turn. He is maximizing agent
    best_val, best_action = -np.inf, None
    for action in state.getLegalActions(cur_agent):
      if action != 'Stop':
        new_val, _ = minimax_func(
          state.generateSuccessor(cur_agent, action),
          cur_depth + 1,
          (cur_agent + 1) % context.n_agents,
          context
        )
        if new_val > best_val:
          best_val = new_val
          best_action = action
  else:
    # Ghosts make their turns. They are minimizing agents.
    best_val, best_action = np.inf, None
    for action in state.getLegalActions(cur_agent):
      if action != 'Stop':
        new_val, _ = minimax_func(
          state.generateSuccessor(cur_agent, action),
          cur_depth + 1,
          (cur_agent + 1) % context.n_agents,
          context
        )
        if new_val < best_val:
          best_val = new_val
          best_action = action

  return best_val, best_action


def minimax_alphabeta(state, cur_depth, cur_agent, alpha, beta, context):
  if (cur_depth == context.max_depth * context.n_agents) or state.isWin() or state.isLose():
    return context.evaluationFunction(state), None

  if cur_agent == 0:
    # Pacman turn. He is maximizing agent
    best_val, best_action = -np.inf, None
    for action in state.getLegalActions(cur_agent):
      if action != 'Stop':
        new_val, _ = minimax_alphabeta(
          state.generateSuccessor(cur_agent, action),
          cur_depth + 1,
          (cur_agent + 1) % context.n_agents,
          alpha, beta, context
        )
        if new_val > best_val:
          best_val = new_val
          best_action = action

        alpha = max(alpha, new_val)
        if beta < alpha:
          break
  else:
    # Ghosts make their turns. They are minimizing agents.
    best_val, best_action = np.inf, None
    for action in state.getLegalActions(cur_agent):
      if action != 'Stop':
        new_val, _ = minimax_alphabeta(
          state.generateSuccessor(cur_agent, action),
          cur_depth + 1,
          (cur_agent + 1) % context.n_agents,
          alpha, beta, context
        )
        if new_val < best_val:
          best_val = new_val
          best_action = action

        beta = min(beta, new_val)
        if beta < alpha:
          break

  return best_val, best_action  

import time

class TimingAgent(MultiAgentSearchAgent):
  def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.timings = []

  def stats(self):
    print('Mean step time: {:.2f} ms'.format(np.mean(self.timings) * 1000))
    print('Std time: {:.2f} ms'.format(np.std(self.timings) * 1000))
    print('Max step time: {:.2f} ms'.format(np.max(self.timings) * 1000))

class MinimaxAgent(TimingAgent):
    def getAction(self, gameState):
        s = time.time()
        res = minimax_func(gameState, 0, 0,
          SimpleNamespace(n_agents=gameState.getNumAgents(), max_depth=self.depth, evaluationFunction=self.evaluationFunction))[1]
        t = time.time()
        self.timings.append(t - s)
        new_state = gameState.generateSuccessor(0, res)
        if new_state.isWin() or new_state.isLose():
          self.stats()

        return res

class AlphaBetaAgent(TimingAgent):
    def getAction(self, gameState):
        s = time.time()
        res = minimax_alphabeta(gameState, 0,0, -np.inf, np.inf,
          SimpleNamespace(n_agents=gameState.getNumAgents(), max_depth=self.depth, evaluationFunction=self.evaluationFunction))[1]
        t = time.time()
        self.timings.append(t - s)
        new_state = gameState.generateSuccessor(0, res)
        if new_state.isWin() or new_state.isLose():
          self.stats()

        return res
