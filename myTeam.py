import logging
import random
import time

from pacai.agents.capture.capture import CaptureAgent
from pacai.util import util
from pacai.util import reflection

def createTeam(firstIndex, secondIndex, isRed,
        dummy = 'pacai.agents.capture.dummy.DummyAgent',
        first = 'pacai.student.myTeam.OffensivePhishingAgent',
        second = 'pacai.student.myTeam.DefensivePhishingAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    # For now, modify firstAgent and keep secondAgent stupid
    firstAgent = reflection.qualifiedImport(first)
    secondAgent = reflection.qualifiedImport(second)

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]

"""
    CaptureAgent (capture.py)
    A base class for capture agents.
    This class has some helper methods that students may find useful.
    The recommended way of setting up a capture agent is just to extend this class
    and implement `CaptureAgent.chooseAction`.
"""

class PhishingAgent(CaptureAgent):
    """
    This will be a base class for our teams agents utilizing Minimax/Expectimax algorithms
    for determining the best action to take.
    """

    def __init__(self, index, evalFn = 'pacai.core.eval.score', depth = 2, **kwargs):
        super().__init__(index, **kwargs)
        print("initial index: ", self.index)

        self._evaluationFunction = reflection.qualifiedImport(evalFn)
        self._treeDepth = int(depth)
        self.agent = CaptureAgent

    def getEvaluationFunction(self):
        return self._evaluationFunction

    def getTreeDepth(self):
        return self._treeDepth

    # Everything above this sets up CaptureAgent like a MultiAgentSearchAgent
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `PhishingAgent.evaluate`.
        """

        '''
        Start of Theo's code from multiagents.py, AlphaBetaAgent (P2)
        Will need to modify this to take into account that there are 2 friendly agents
        The basic AlphaBetaAgent code does not communicate with team member for best
        wholistic plan. Not sure how to do that right now, going to try this as is.
        '''

        # Agent count has been reformatted to be count of opponents + 1 (our PhishingAgent, self)
        opponent_index = self.getOpponents(gameState)
        agent_count = len(opponent_index)+1
        max_depth = self.getTreeDepth() * agent_count
        print(opponent_index)

        def minValue(state, depth, alpha, beta):
            minimum = float("inf")
            # We need to determine the agent differently
            # If agent_index != 0, then we must refer to an opponent agent
            agent_index = depth % agent_count
            print("agent_index: ", agent_index)
            # With naive setup, our firstAgent will have agent_index = 0
            if agent_index != 0:
                print(opponent_index[agent_index - 1])
                agent = opponent_index[agent_index - 1]
                legal_actions = state.getLegalActions(agent)
                print(legal_actions)
            else:
                legal_actions = state.getLegalActions(self.index)

            if legal_actions is None or len(legal_actions) == 0:
                return self.evaluate(state)
            for action in legal_actions:
                if str(action) != "Stop":
                    if agent is not None:
                        print("Generating successor from + " + str(state.getAgentPosition(agent)) + " with action " + action)
                        # This is where it bricks
                        successorState = state.generateSuccessor(agent, action)
                    else:
                        successorState = state.generateSuccessor(self.index, action)
                    if agent_index == agent_count - 1:
                        # last ghost
                        successorValue = maxValue(successorState, depth + 1, alpha, beta)
                    else:
                        successorValue = minValue(successorState, depth + 1, alpha, beta)

                    if minimum > successorValue:
                        minimum = successorValue

                    if minimum <= alpha:
                        return minimum
                    beta = min(beta, minimum)

            return minimum

        def maxValue(state, depth, alpha, beta):
            '''
                If we're at the top-level call, we want to return the action
                associated with the max-value rather than the value itself
            '''
            maximum = -float("inf")
            max_action = None
            legal_actions = state.getLegalActions(self.index)
            print("index: ", self.index)
            print(state.getAgentPosition(self.index))
            print(legal_actions)

            if legal_actions is None or len(legal_actions) == 0 or depth == max_depth:
                return self.evaluate(state)

            for action in legal_actions:
                if str(action) != "Stop":
                    print("Generating successor from + " + str(state.getAgentPosition(0)) + " with action " + action)
                    successorState = state.generateSuccessor(self.index, action)
                    successorValue = minValue(successorState, depth + 1, alpha, beta)
                    if maximum < successorValue:
                        maximum = successorValue
                        max_action = action
                        print("Override max: " + str(maximum) + " going " + str(action))

                    if maximum >= beta and depth != 0:
                        return maximum
                    alpha = max(alpha, maximum)

            if depth == 0:
                return max_action
            else:
                return maximum

        return maxValue(gameState, 0, -float("inf"), float("inf"))

    def getSuccessor(self, gameState, action):
        """
        *** Unchanged from ReflexCaptureAgent ***
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState):
        """
        *** Unchanged AND UNUSED from ReflexCaptureAgent ***
        Computes a linear combination of features and feature weights.
        """
        '''
        features = {}
        features['stateScore'] = self.getScore(gameState)
        # Compute distance to the nearest food.
        foodList = self.getFood(gameState).asList()
        if (len(foodList) > 0):
            myPos = gameState.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        weights = {
            'stateScore': 100,
            'distanceToFood': -1
        }
        stateEval = sum(features[feature] * weights[feature] for feature in features)
        return stateEval
        '''
        features = self.getFeatures(gameState)
        weights = self.getWeights(gameState)
        capsules = self.getCapsules(gameState)
        stateEval = sum(features[feature] * weights[feature] for feature in features)

        return stateEval

    def getFeatures(self, gameState):
        """
        *** Unchanged AND UNUSED from ReflexCaptureAgent ***
        Returns a dict of features for the state.
        The keys match up with the return from `ReflexCaptureAgent.getWeights`.
        """

        return {
            'stateScore': self.getScore(gameState)
        }

    def getWeights(self, gameState):
        """
        *** Unchanged AND UNUSED from ReflexCaptureAgent ***
        Returns a dict of weights for the state.
        The keys match up with the return from `ReflexCaptureAgent.getFeatures`.
        """

        return {
            'stateScore': 1.0
        }

class OffensivePhishingAgent(PhishingAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState):
        features = {}
        features['stateScore'] = self.getScore(gameState)

        '''
        # Compute distance to nearest boundary.
        myPos = gameState.getAgentState(self.index).getPosition()
        min_boundary = 1000000
        for min_b in range(len(self.boundary)):
            max_boundary = self.getMazeDistance(myPos, self.boundary[min_b])
            if(max_boundary < min_boundary):
                min_boundary = max_boundary
        features['returned'] = min_boundary
        '''

        # Compute distance to the nearest food.
        foodList = self.getFood(gameState).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = gameState.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        # Compute distance to nearest capsule
        capsuleList = self.getCapsules(gameState)
        if len(capsuleList) > 0:
            minCapDist = 99999
            dist = min([self.getMazeDistance(myPos, caps) for caps in capsuleList])
            if dist < minCapDist:
                minCapDist = dist
            features['distanceToCapsule'] = minCapDist
        else:
            features['distanceToCapsule'] = 0

        return features

    def getWeights(self, gameState):
        return {
            'stateScore': 100,
            'distanceToFood': -1,
            'distanceToCapsule': -1
        }

class DefensivePhishingAgent(PhishingAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState):
        features = {}

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        '''
        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1
        '''

        return features

    def getWeights(self, gameState):
        '''
        These both rely on an associated action
        Removed: 'reverse': -2
                 'stop': -100
        '''
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
        }
