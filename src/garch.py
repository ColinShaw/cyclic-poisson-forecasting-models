from arch import arch_model
import numpy


class GARCH(object):

    def __init__(self, pq=(1,1), look_back=72):
        self.__p, self.__q = pq
        self.__look_back = look_back

    def __create_garch_dataset(self, dataset):
        x = []
        for i in range(len(dataset)-self.__look_back-1):
            x.append(dataset[i:i+self.__look_back])
        return x
            
    def predict(self, data):
        data = self.__create_garch_dataset(data)
        result = []
        for d in data:
            model = arch_model(
                d, 
                vol  = 'Garch', 
                p    = self.__p, 
                o    = 0, 
                q    = self.__q, 
                dist = 'Normal'
            )
            res = model.fit(update_freq=5)
            forecast = res.forecast()
            result.append(forecast.mean.iloc[-1][0])
        return result
        
