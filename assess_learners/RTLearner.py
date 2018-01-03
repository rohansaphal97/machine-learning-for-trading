"""Implementing a Regression Tree Learner"""
import numpy as np

class RTLearner(object):
    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        pass

    def author(self):
        return 'analwaya3'

    def addEvidence(self, x, y):
        self.model = np.array([]).reshape((0,4))
        data = np.column_stack((x, y))
        #Building the model using recursive calls
        self.model = np.concatenate([self.model, self.createDT(data)], axis=0)

    """Recursive function to build the decision tree"""
    def createDT(self, data):
        x = data[:, 0:-1]
        y = data[:, -1:]
        if (x.shape[0] <= self.leaf_size) or (np.all(y == y[0])):
            split_val = y.mean()
            return np.vstack((self.model, np.array([['leaf', split_val, np.nan, np.nan]])))
        else:
            max_index = self.get_index(x, y)
            split_val = np.median(data[:, max_index])
            righttree_shape = (data[data[:, max_index] > split_val]).shape[0]
            if (righttree_shape == 0):
                split_val = np.mean(data[:, max_index])
                leaf_val = y.mean()
                return np.vstack((self.model, np.array([['leaf', leaf_val, np.nan, np.nan]])))
            lefttree = self.createDT(data[data[:, max_index] <= split_val])
            righttree = self.createDT(data[data[:, max_index] > split_val])
            root = np.array([[max_index, split_val, 1, lefttree.shape[0] + 1]])
            return np.vstack((self.model, root, lefttree, righttree))

    """Function to query the decision tree"""
    def query(self, points):
        decision = np.empty([points.shape[0], 1], dtype = float)
        for index, row in enumerate(points):
            start_index = 0
            factor = self.model[start_index, 0]
            while (factor != 'leaf'):
                factor_int = int(float(factor))
                if (row[factor_int] <= float(self.model[start_index, 1])):
                    start_index += 1
                else:
                    start_index += int(float(self.model[start_index, 3]))
                factor = self.model[start_index, 0]

            decision[index, 0] = self.model[start_index, 1]

        return decision.flatten()

    """Function to get index of the column to perform the split on"""
    def get_index(self, x, y):
        idx = np.random.randint(x.shape[1], size=1)
        return idx[0]

    """Using correlation as the basis of defining the column for split"""
    def get_correlation(self, x, y):
        return np.corrcoef(x, y)
