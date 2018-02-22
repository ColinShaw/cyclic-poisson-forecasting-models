import numpy as np


class Poisson(object):

    __profile1 = [3,3,3,3,3,5,10,15,15,10,7,10,15,10,7,5,7,8,7,5,5,5,5,3]
    __profile2 = [3,2,1,3,1,3,16,20,30,15,10,20,10,5,4,2,3,5,3,3,2,2,1,2]

    def generate(self, num=1, profile=1):
        if profile == 1:
            profile = self.__profile1
        else:
            profile = self.__profile2
        result, normed = [], []
        for i in range(num):
            new = [np.random.poisson(j) for j in profile]
            result += new
            new = np.subtract(new, profile).tolist()
            normed += new
            
        return np.array(result).astype(np.float32), np.array(normed).astype(np.float32)

