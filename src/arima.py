from statsmodels.tsa.arima_model import ARIMA as A
import numpy


class ARIMA(object):

    def __init__(self, order=(2,1,0), look_back=72):
        self.__order = order
        self.__look_back = look_back

    def __create_arima_dataset(self, dataset):
        x = []
        for i in range(len(dataset)-self.__look_back-1):
            x.append(dataset[i:i+self.__look_back])
        return x
            
    def predict(self, data):
        data = self.__create_arima_dataset(data)
        result = []
        for d in data:
            model = A(d, self.__order)
            fit = model.fit(disp=0)
            result.append(fit.forecast()[0])
        return result
        
