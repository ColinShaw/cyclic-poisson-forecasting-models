import statsmodels.api as sm
import numpy


class SARIMAX(object):

    def __init__(self, order=(0,0,2), season=(1,1,2,24), look_back=72):
        self.__order = order
        self.__season = season
        self.__look_back = look_back

    def __create_sarimax_dataset(self, dataset):
        x = []
        for i in range(len(dataset)-self.__look_back-1):
            x.append(dataset[i:i+self.__look_back])
        return x
            
    def predict(self, data):
        data = self.__create_sarimax_dataset(data)
        result = []
        for d in data:
            model = sm.tsa.statespace.SARIMAX(
                d, 
                order                 = self.__order,
                seasonal_order        = self.__season,
                enforce_stationarity  = False,
                enforce_invertibility = False
            )
            fit = model.fit(disp=0)
            result.append(fit.forecast()[0])
        return result
        
