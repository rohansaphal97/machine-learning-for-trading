import numpy as np
import math

class DTLearner(object):
    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        pass

    def author(self):
        return 'analwaya3'

    def addEvidence(self, x, y):
        self.model = np.array([]).reshape((0,4))
        data = np.column_stack((x, y))
        self.model = np.concatenate([self.model, self.createDT(data)], axis=0)

    def createDT(self, data):
        x = data[:, 0:-1]
        y = data[:, -1:]
        if (x.shape[0] <= self.leaf_size) or (np.all(y == y[0])):
            split_val = y.mean()
            return np.vstack((self.model, np.array([['leaf', split_val, np.nan, np.nan]])))
        else:
            max_index = self.get_index(x, y)
            split_val = np.median(data[:, max_index])
            lefttree_shape = (data[data[:, max_index] <= split_val]).shape[0]
            righttree_shape = (data[data[:, max_index] > split_val]).shape[0]
            if (lefttree_shape == data.shape[0]) or (righttree_shape == data.shape[0]):
                split_val = np.mean(data[:, max_index])
                leaf_val = y.mean()
                return np.vstack((self.model, np.array([['leaf', leaf_val, np.nan, np.nan]])))
            lefttree = self.createDT(data[data[:, max_index] <= split_val])
            righttree = self.createDT(data[data[:, max_index] > split_val])
            root = np.array([[max_index, split_val, 1, lefttree.shape[0] + 1]])
            return np.vstack((self.model, root, lefttree, righttree))


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

            decision[index, 0] = float(self.model[start_index, 1])

        return decision.flatten()


    def get_index(self, x, y):
        corrs = abs(np.array([self.get_correlation(x_attr, y.T) for x_attr in x.T]))
        corr_matrix = np.zeros((x.shape[1], 0))
        for index, corr in enumerate(corrs):
            if(not(math.isnan(corr[0, 1]))):
                corr_matrix = np.append(corr_matrix, np.array(corr[0,1]))
            else:
                corr_matrix = np.append(corr_matrix, np.array([0]))

        selected_attr = np.argmax(corr_matrix)

        return selected_attr

    def get_correlation(self, x, y):
        return np.corrcoef(x, y)
