import os
print(os.getcwd())
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
dataset = loadtxt("pima-indians-diabetes.csv",delimiter = ",")
X = dataset[:,:8]
y = dataset[:,8]
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X,y,epochs = 150,batch_size = 16,verbose = 0) #setting verbose to zero doesnt show epoch progress
_,accuracy = model.evaluate(X,y)
print("accuracy is ",accuracy * 100)
predictions = model.predict_classes(X)
for i in range(5):
    print("%s => %d expected(%d)"% (X[i].tolist(),predictions[i],y[i]))
for layer in model.layers:
    print(layer.get_weights())
