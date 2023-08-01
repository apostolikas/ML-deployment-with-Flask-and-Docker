from keras.models import Sequential
from keras.layers import Dense, Dropout

class SimpleNet():
  @staticmethod
  def build():
    model=Sequential()
    model.add(Dense(1000, input_dim=4, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    return model
