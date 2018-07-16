from keras.utils import np_utils  
from sklearn.model_selection import train_test_split
import pickle
import os
import numpy as np
import csv

def _change_ont_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

def load_mnist(normalize=False, flatten=False, one_hot_label=False):
    """
    Parameters
    ----------
    normalize : Normalize the pixel values
    flatten : Flatten the images as one array
    one_hot_label : Encode the labels as a one-hot array

    Returns
    -------
    (Trainig Image, Training Label), (Test Image, Test Label)
    """

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    if one_hot_label:
        dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_ont_hot_label(dataset['test_label'])

    return dataset['train_img'], dataset['train_label'], dataset['test_img'], dataset['test_label']

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

# Load the MNIST dataset
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

X_Train, y_Train, X_Test, y_Test = load_mnist(normalize=False, flatten=False)

X_Test = loadTestData() 

np.random.seed(10)  

# resize the data  
X_Train4D = X_Train.reshape(X_Train.shape[0], 28, 28, 1).astype('float32')  
X_Test4D = X_Test.reshape(X_Test.shape[0], 28, 28, 1).astype('float32')

# Standardize feature data  
X_Train4D_norm = X_Train4D / 255.0  
X_Test4D_norm = X_Test4D /255.0  
  
# Label Onehot-encoding  
y_TrainOneHot = np_utils.to_categorical(y_Train)

# get validation data
X_Train4D_norm, X_Val, y_TrainOneHot, y_Val = train_test_split(X_Train4D_norm, y_TrainOneHot, test_size=0.2)

from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,Activation,BatchNormalization

model = Sequential() 

# Create CN layer 1  
model.add(Conv2D(filters=16, #16  
                 kernel_size=(5,5), #(5,5)  
                 padding='same',  
                 input_shape=(28,28,1),  
                 #activation='relu',
                 name='conv2d_1')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
# Create Max-Pool 1  
model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_1'))  

# Create CN layer 2  
model.add(Conv2D(filters=36,  #36
                 kernel_size=(5,5), #(5,5)
                 padding='same',  
                 #activation='relu',
                 name='conv2d_2'))  
model.add(BatchNormalization())
model.add(Activation('relu'))
  
# Create Max-Pool 2  
model.add(MaxPool2D(pool_size=(2,2), name='max_pooling2d_2'))  

# Add Dropout layer  
model.add(Dropout(0.25, name='dropout_1'))

model.add(Flatten(name='flatten_1'))

model.add(Dense(128, activation='relu', name='dense_1'))  
model.add(Dropout(0.5, name='dropout_2'))

'''
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
'''

model.add(Dense(10, activation='softmax', name='dense_2'))

model.summary()  

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
            #zoom_range=0.1,
            vertical_flip=False)  # randomly flip images

datagen.fit(X_Train4D_norm)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.0000001, verbose=1)

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
print("original structure batch norm")