from statsmodels.tsa.arima_model import ARIMA as A


class ARIMA(object):

    def __init__(self, order=(1,0,0), look_back=25):
        self.__order = order
        self.__look_back = look_back

    def __create_arima_dataset(self, dataset):
        x, y = [], []
        for i in range(len(dataset)-self.__look_back-1):
            x.append(dataset[i:i+self.__look_back])
            y.append(dataset[i+self.__look_back])
        x = numpy.array(x)
        y = numpy.array(y)
        x = numpy.reshape(x, (x.shape[0],1,x.shape[1]))
        y = numpy.reshape(y, (y.shape[0],1,y.shape[1]))
        return x, y
            
    def predict(self, data):
        data, _ = self.__create_arima_dataset(data)
        result = []
        for d in data:
            model = A(d, self.__order)
            fit = model.fit(disp=0)
            result.append(fit.forecast[0])
        return result
        
