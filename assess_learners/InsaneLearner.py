"""Create several bagged learners"""
import numpy as np
import LinRegLearner as lrl
import BagLearner as bl

class InsaneLearner(object):
    def __init__(self, verbose):
        self.learners = []
        pass

    def author(self):
        return 'analwaya3'

    def addEvidence(self, x, y):
        for i in range(20):
            learner = bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False)
            learner.addEvidence(x, y)
            self.learners.append(learner)

    def query(self, points):
        y = []
        for learner in self.learners:
            y.append(learner.query(points))
        return np.mean(y, axis= 0)