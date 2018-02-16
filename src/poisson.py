import numpy as np


class Poisson(object):

    __profile1 = [3,3,3,3,3,5,10,15,15,10,7,10,15,10,7,5,7,8,7,5,5,5,5,3]
    __profile2 = [3,2,1,3,1,3,16,20,30,15,10,20,10,5,4,2,3,5,3,3,2,2,1,2]

    def generate(self, num=1, profile=1):
        if profile == 1:
            profile = self.__profile1
        else:
            profile = self.__profile2
        result = []
        for i in range(num):
            result += [np.random.poisson(j) for j in profile]
        return np.array(result).astype(np.float32)

