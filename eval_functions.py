from environment.util import manhattanDistance

from collections import deque
import numpy as np


def is_valid(pos, mat):
    return (pos[0] >= 0) and (pos[1] >= 0) and (pos[0] < mat.shape[0]) and (pos[1] < mat.shape[1])

def bfs(mat, src, dest):
    assert is_valid(src, mat) and is_valid(dest, mat), "Coordinates"
    visited = np.zeros_like(mat).astype(np.bool)
    visited[src] = True
    dist = np.ones_like(mat).astype(np.int32) * np.inf
    dist[src] = 0

    q = deque()
    s = (src, 0)
    q.append(s)

    while q:
        pt, dist = q.popleft()
        if pt == dest:
            return dist

        for dx, dy in zip([-1, 0, 0, 1], [0, -1, 1, 0]):
            new = (pt[0] + dx, pt[1] + dy)

            if is_valid(new, mat) and mat[new] and (not visited[new]):
                visited[new] = True
                q.append((new, dist + 1))
    return -1


def eval_v1(currentGameState):
    return currentGameState.getScore()


def eval_v2(currentGameState):
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    currentPosition = currentGameState.getPacmanPosition()
    if successorGameState.isWin():
        return 100000
    score = 0
    for state in newGhostStates:
        if state.getPosition() == currentPosition and state.scaredTimer == 0:
            return -100000
    scared_ghost = [manhattanDistance(state.getPosition(), currentPosition) for state in newGhostStates if state.scaredTimer != 0]
    if len(scared_ghost) > 0:
        score += float(3 / min(scared_ghost))
    #if action == 'Stop':
    #    score -= 1000
    foodDistance = [manhattanDistance(newPos, food) for food in newFood]
    nearestFood = min(foodDistance)
    score += float(1/nearestFood)
    if len(scared_ghost) == 0:
        currentGhostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in currentGameState.getGhostStates()]
        nearestCurrentGhost = min(currentGhostDistances)
        score -= float(1 / (nearestCurrentGhost + 1e-5))
    newGhostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    nearestNewGhost = min(newGhostDistances)
    return successorGameState.getScore() + score


def eval_v3(currentGameState):
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    foodPos = newFood.asList()
    foodCount = len(foodPos)

    # walls = currentGameState.getWalls()

    # mat = np.zeros((walls.width, walls.height), dtype=np.bool)
    # for i in range(walls.width):
    #  for j in range(walls.height):
    #    mat[i,j] = walls[i][j]

    score = currentGameState.getScore()

    closestDistance = 1e6
    for i in range(foodCount):
        distance = manhattanDistance(newPos, foodPos[i]) + foodCount * 100
        # distance = bfs(mat, newPos, foodPos[i]) + foodCount * 100
        if distance < closestDistance:
            closestDistance = distance
            closestFood = foodPos

    if foodCount == 0:
        closestDistance = 0

    score -= closestDistance

    for i in range(len(newGhostStates)):
        ghostPos = currentGameState.getGhostPosition(i + 1)
        if manhattanDistance(newPos, ghostPos) <= 1:
            score -= 1e6

    return score

