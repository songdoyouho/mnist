from keras.utils import np_utils  
from sklearn.model_selection import train_test_split
import numpy as np
import csv

def loadTrainData():  
    l=[]  
    with open('train.csv') as file:  
        lines=csv.reader(file)  
        for line in lines:  
            l.append(line) 
        l.remove(l[0]) 
        l = [[int(tmppp) for tmppp in tmp] for tmp in l]
        l=np.array(l) 
        label=l[:,0]  
        data=l[:,1:]  
    return data,label
  
def loadTestData():  
    l=[]  
    with open('test.csv') as file: 
        lines=csv.reader(file)  
        for line in lines:  
            l.append(line)  
        l.remove(l[0]) 
        l = [[int(tmppp) for tmppp in tmp] for tmp in l]
        data=np.array(l)  
    return data

random_seed = np.random.seed(10)  

# Read MNIST data  
X_Train, y_Train = loadTrainData()
X_Test = loadTestData() 
  
# resize the data  
X_Train4D = X_Train.reshape(X_Train.shape[0], 28, 28, 1).astype('float32')  
X_Test4D = X_Test.reshape(X_Test.shape[0], 28, 28, 1).astype('float32')

# Standardize feature data  
X_Train4D_norm = X_Train4D / 255  
X_Test4D_norm = X_Test4D /255  
  
# Label Onehot-encoding  
y_TrainOneHot = np_utils.to_categorical(y_Train)

# get validation data
X_Train4D_norm, X_Val, y_TrainOneHot, y_Val = train_test_split(X_Train4D_norm, y_TrainOneHot, test_size=0.1, random_seed=random_seed)

from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,Activation
from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.0000001, verbose=1)

model = Sequential() 

'''
model.add(Conv2D(filters=64,  
                 kernel_size=(5,5),  
                 padding='same',  
                 input_shape=(28,28,1),  
                 activation='relu',
                 name='conv2d_1')) 

model.add(Conv2D(filters=128,  
                 kernel_size=(5,5),  
                 padding='same',  
                 activation='relu',
                 name='conv2d_2')) 

model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_1'))

model.add(Conv2D(filters=256,
                kernel_size=(5,5),
                padding='same',
                activation='relu',
                name='conv2d_3'))

model.add(Conv2D(filters=256,
                kernel_size=(3,3),
                padding='same',
                activation='relu',
                name='conv2d_4'))

model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_2'))  

model.add(Dropout(0.2))

model.add(Conv2D(filters=512,  
                 kernel_size=(3,3),  
                 padding='same',  
                 activation='relu',
                 name='conv2d_5')) 

model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_3'))  

model.add(Flatten(name='flatten'))

model.add(Dropout(0.5))

model.add(Dense(2048, activation='relu', name='FC1'))

model.add(Dense(10, activation='softmax', name='softmax'))

model.summary()
''' 
# Create CN layer 1  
model.add(Conv2D(filters=16, #16  
                 kernel_size=(5,5), #(5,5)  
                 padding='same',  
                 input_shape=(28,28,1),  
                 activation='relu',
                 name='conv2d_1')) 
# Create Max-Pool 1  
model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_1'))  
  
# Create CN layer 2  
model.add(Conv2D(filters=36,  #36
                 kernel_size=(5,5), #(5,5)
                 padding='same',  
                 activation='relu',
                 name='conv2d_2'))  
  
# Create Max-Pool 2  
model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_2'))  
'''
# Create CN layer 3
model.add(Conv2D(filters=64,  #64
                 kernel_size=(3,3), #(5,5)
                 padding='same',
                 activation='relu',
                 name='conv2d_3'))  

# Create Max-Pool 3  steps_per_epoch
model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_3'))  
'''
# Add Dropout layer  
model.add(Dropout(0.25, name='dropout_1'))

model.add(Flatten(name='flatten_1'))

model.add(Dense(128, activation='relu', name='dense_1'))  
model.add(Dropout(0.5, name='dropout_2'))

model.add(Dense(10, activation='softmax', name='dense_2'))

model.summary()  
print("")
'''
# 定義訓練方式  
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

# 開始訓練  
train_history = model.fit(x=X_Train4D_norm,  
                          y=y_TrainOneHot, validation_split=0.1,  
                          epochs=100, batch_size=100, callbacks=[reduce_lr],verbose=2)
'''
# 圖像增強 data argmentation--------------------------------------------------------------------------
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

datagen.fit(X_Train4D_norm)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_history = model.fit_generator(datagen.flow(X_Train4D_norm, y_TrainOneHot, batch_size=100),
                            steps_per_epoch=X_Train4D_norm.shape[0] // 10,
                            epochs=100,
                            validation_data=(X_Val, y_Val),
                            callbacks=[reduce_lr],
                            verbose=2
                            )

import matplotlib.pyplot as plt 

def plot_image(image):  
    fig = plt.gcf()  
    fig.set_size_inches(2,2)  
    plt.imshow(image, cmap='binary')  
    plt.show()  
  
def plot_images_labels_predict(images, labels, prediction, idx, num=10):  
    fig = plt.gcf()  
    fig.set_size_inches(12, 14)  
    if num > 25: num = 25  
    for i in range(0, num):  
        ax=plt.subplot(5,5, 1+i)  
        ax.imshow(images[idx], cmap='binary')  
        title = "l=" + str(labels[idx])  
        if len(prediction) > 0:  
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))  
        else:  
            title = "l={}".format(str(labels[idx]))  
        ax.set_title(title, fontsize=10)  
        ax.set_xticks([]); ax.set_yticks([])  
        idx+=1  
    plt.show()  
  
def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()

def saveResult(result):
 with open ('result.csv', mode='w',newline="\n") as write_file:
  writer = csv.writer(write_file)
  writer.writerow(["ImageId","Label"])
  for i in range(len(result)):
   writer.writerow([i+1,result[i]])

show_train_history(train_history, 'acc', 'val_acc')

show_train_history(train_history, 'loss', 'val_loss')

print("\t[Info] Making prediction of X_Test4D_norm")  
prediction = model.predict_classes(X_Test4D_norm)  # Making prediction and save result to prediction 
saveResult(prediction)
print("end")
print("original small structure")