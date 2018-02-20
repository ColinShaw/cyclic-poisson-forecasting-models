import numpy 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM as L, Dense
from keras.optimizers import Adam


class LSTM(object):

    def __init__(self, units=400, look_back=72):
        self.__scaler = MinMaxScaler(feature_range=(0,1))
        self.__look_back = look_back
        self.__model = Sequential([
            L(
                units,  
                input_shape      = (1,look_back),
                return_sequences = True,
            ),
            Dense(1)
        ])
        self.__model.compile(
            loss      = 'mse', 
            optimizer = Adam(lr=1.0e-3)
        )

    def __create_lstm_dataset(self, dataset):
        x, y = [], []
        for i in range(len(dataset)-self.__look_back-1):
            x.append(dataset[i:i+self.__look_back])
            y.append(dataset[i+self.__look_back])
        x = numpy.array(x)
        y = numpy.array(y)
        x = numpy.reshape(x, (x.shape[0],1,x.shape[1]))
        y = numpy.reshape(y, (y.shape[0],1,y.shape[1]))
        return x, y
        
    def train(self, train_set, epochs=100, batch_size=32):
        train_set = [[x] for x in train_set]
        train_set = self.__scaler.fit_transform(train_set)
        train_x, train_y = self.__create_lstm_dataset(train_set)
        self.__model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2)

    def predict(self, data):
        data = [[x] for x in data]
        data = self.__scaler.transform(data)
        data, _ = self.__create_lstm_dataset(data)
        pred = self.__model.predict(data)
        pred = [x[0] for x in pred]
        pred = self.__scaler.inverse_transform(pred)
        return [x[0] for x in pred]

