"""Class to create a bag of learners"""
import numpy as np
import DTLearner as dt
import RTLearner as rt
import LinRegLearner as lrl

class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost, verbose):
        self.learner = learner
        self.bags = bags
        self.kwargs = kwargs
        self.learners = []
        for i in range(self.bags):
            self.learners.append(learner(**kwargs))
        pass

    def author(self):
        return 'analwaya3'

    def addEvidence(self, x, y):
        data = np.column_stack((x, y))
        self.bagging(data)

    def bagging(self, data):
        row_size = data.shape[0]
        n = int(1.0* data.shape[0])
        data_bag = np.empty(shape = (0, data.shape[1]))
        for i in range(self.bags):
            idx = np.random.randint(row_size, size=row_size)
            data_bag = data[idx, :]
            data_bagX = np.delete(data_bag, -1, axis = 1)
            data_bagY = data_bag[:, -1]
            self.learners[i].addEvidence(data_bagX, data_bagY)

    """Taking the average of predictions by bagged learners for making a final prediction"""
    def query(self, points):
        units = list()
        for i in range(self.bags):
            learned = self.learners[i].query(points)
            units.append(learned)
        z = np.mean(units, axis = 0)
        return z

