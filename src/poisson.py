import numpy as np


class Poisson(object):

    __profile = [3,3,3,3,3,5,10,15,15,10,7,10,15,10,7,5,7,8,7,5,5,5,5,3]

    def generate(self, num=1, profile=None):
        if profile is None:
            profile = self.__profile
        result = []
        for i in range(num):
            result += [np.random.poisson(j) for j in self.__profile]
        return np.array(result).astype(np.float32)

