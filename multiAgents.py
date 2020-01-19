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
import math

from game import Agent

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        exFood = currentGameState.getFood()
        exFoodAsList = exFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        evalVal = currentGameState.getScore()
        newGhostPositions = list(map(lambda x: x.getPosition(), newGhostStates))
            
        if (newPos in newGhostPositions): # if pacman's position and ghost's position intersects
            if max(newScaredTimes) == 0: # Capsuls not eaten
                return evalVal + (-1000000000)
            else: # Capsuls eaten...
                return -evalVal + 1000000000
        else: # Focus on foods left...
            if len(exFoodAsList) != 0:
                pacmanDistanceToCurrentFood = list(map(lambda x: util.manhattanDistance(newPos, x), exFoodAsList))
                evalVal = evalVal - (min(pacmanDistanceToCurrentFood))
        
        return evalVal

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def value(state, agentIndex, currentDepth):
            tempAction = ""
            
            if agentIndex == state.getNumAgents():
                currentDepth = currentDepth + 1
                agentIndex = agentIndex % agentIndex
                
            if currentDepth == self.depth or state.isWin() or state.isLose():
                return (tempAction, self.evaluationFunction(state))
            else:
                if agentIndex == 0: # pacman(max agent)'s turn...
                    return maxVal(state, agentIndex, currentDepth)
                elif agentIndex >= 1: # ghosts(min agents)' turn...
                    return minVal(state, agentIndex, currentDepth)
        
        # In maxVal and minVal functions, return values are kept as tuple of action and its corresponding value
        # The reason is to be able to return just the right action among actions list while comparing their values...
        
        def maxVal(state, agentIndex, currentDepth):
            tempAction = ""
            v = float("-Inf")
            actionValuePair = (tempAction, v)
            maxActions = state.getLegalActions(agentIndex)
            
            for maxAction in maxActions:
                # Successor of a state after one action corresponds to a single successor state
                successor = state.generateSuccessor(agentIndex, maxAction)
                successorVal = value(successor, agentIndex + 1, currentDepth)
                if successorVal[1] > v:
                    actionValuePair = (maxAction, successorVal[1])
                    v = successorVal[1]
                        
            return actionValuePair
            
        def minVal(state, agentIndex, currentDepth):
            tempAction = ""
            v = float("Inf")
            actionValuePair = (tempAction, v)
            minActions = state.getLegalActions(agentIndex)
            
            for minAction in minActions:
                # Successor of a state after one action corresponds to a single successor state
                successor = state.generateSuccessor(agentIndex, minAction)
                successorVal = value(successor, agentIndex + 1, currentDepth)
                if successorVal[1] < v:
                    actionValuePair = (minAction, successorVal[1])
                    v = successorVal[1]
                        
            return actionValuePair
            
        return value(gameState, 0, 0)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(state, agentIndex, currentDepth, alpha, beta):
            tempAction = ""
            
            if agentIndex == state.getNumAgents():
                currentDepth = currentDepth + 1
                agentIndex = agentIndex % agentIndex
                
            if currentDepth == self.depth or state.isWin() or state.isLose():
                return (tempAction, self.evaluationFunction(state))
            else:
                if agentIndex == 0: # pacman(max agent)'s turn...
                    return maxVal(state, agentIndex, currentDepth, alpha, beta)
                elif agentIndex >= 1: # ghosts(min agents)' turn...
                    return minVal(state, agentIndex, currentDepth, alpha, beta)
                
        # In maxVal and minVal functions, return values are kept as tuple of action and its corresponding value
        # The reason is to be able to return just the right action among actions list while comparing their values...
        
        def maxVal(state, agentIndex, currentDepth, alpha, beta):
            tempAction = ""
            v = float("-Inf")
            actionValuePair = (tempAction, v)
            maxActions = state.getLegalActions(agentIndex)
            
            for maxAction in maxActions:
                # Successor of a state after one action corresponds to a single successor state
                successor = state.generateSuccessor(agentIndex, maxAction)
                successorVal = value(successor, agentIndex + 1, currentDepth, alpha, beta)
                if successorVal[1] > v:
                    actionValuePair = (maxAction, successorVal[1])
                    v = successorVal[1]
                if v > beta:
                    return (maxAction, v)
                alpha = max(alpha, v)
                        
            return actionValuePair
            
        def minVal(state, agentIndex, currentDepth, alpha, beta):
            tempAction = ""
            v = float("Inf")
            actionValuePair = (tempAction, v)
            minActions = state.getLegalActions(agentIndex)
            
            for minAction in minActions:
                # Successor of a state after one action corresponds to a single successor state
                successor = state.generateSuccessor(agentIndex, minAction)
                successorVal = value(successor, agentIndex + 1, currentDepth, alpha, beta)
                if successorVal[1] < v:
                    actionValuePair = (minAction, successorVal[1])
                    v = successorVal[1]
                if v < alpha:
                    return (minAction, v)
                beta = min(beta, v)
                        
            return actionValuePair
            
        return value(gameState, 0, 0, float("-Inf"), float("Inf"))[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def value(state, agentIndex, currentDepth):
            tempAction = ""
            
            if agentIndex == state.getNumAgents():
                currentDepth = currentDepth + 1
                agentIndex = agentIndex % agentIndex
                
            if currentDepth == self.depth or state.isWin() or state.isLose():
                return (tempAction, self.evaluationFunction(state))
            else:
                if agentIndex == 0: # pacman(max agent)'s turn...
                    return maxVal(state, agentIndex, currentDepth)
                elif agentIndex >= 1: # chance...
                    return expMinVal(state, agentIndex, currentDepth)
        
        # In maxVal and minVal functions, return values are kept as tuple of action and its corresponding value
        # The reason is to be able to return just the right action among actions list while comparing their values...
        
        def maxVal(state, agentIndex, currentDepth):
            tempAction = ""
            v = float("-Inf")
            actionValuePair = (tempAction, v)
            maxActions = state.getLegalActions(agentIndex)
            
            for maxAction in maxActions:
                # Successor of a state after one action corresponds to a single successor state
                successor = state.generateSuccessor(agentIndex, maxAction)
                successorVal = value(successor, agentIndex + 1, currentDepth)
                if successorVal[1] > v:
                    actionValuePair = (maxAction, successorVal[1])
                    v = successorVal[1]
                        
            return actionValuePair
            
        def expMinVal(state, agentIndex, currentDepth):
            tempAction = ""
            v = 0
            actionValuePair = (tempAction, v)
            expActions = state.getLegalActions(agentIndex)
            # Probability of randomly acting ghosts, denoted as p
            # Since probability is uniform...
            p = 1 / len(expActions)
            
            for expAction in expActions:
                # Successor of a state after one action corresponds to a single successor state
                successor = state.generateSuccessor(agentIndex, expAction)
                successorVal = value(successor, agentIndex + 1, currentDepth)
                v = v + (successorVal[1] * p)
                actionValuePair = (tempAction, v)
                        
            return actionValuePair
            
        return value(gameState, 0, 0)[0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    evalVal = currentGameState.getScore()
    currentFood = currentGameState.getFood()
    currentFoodAsList = currentFood.asList()
    pacmanPos = currentGameState.getPacmanPosition()
    capsules = currentGameState.getCapsules()
    ghostPos = currentGameState.getGhostPositions()
    
    if len(currentFoodAsList) != 0:
        pacmanDistanceToFoods = list(map(lambda x: util.manhattanDistance(pacmanPos, x), currentFoodAsList))
        evalVal = evalVal - min(pacmanDistanceToFoods)
    if len(capsules) != 0:
        pacmanDistanceToCapsules = list(map(lambda x: util.manhattanDistance(pacmanPos, x), capsules))
        evalVal = evalVal - min(pacmanDistanceToCapsules)
    else:
        pacmanDistanceToGhost = list(map(lambda x: util.manhattanDistance(pacmanPos, x), ghostPos))
        evalVal = evalVal - min(pacmanDistanceToGhost)
    
    return evalVal
# Abbreviation
better = betterEvaluationFunction
