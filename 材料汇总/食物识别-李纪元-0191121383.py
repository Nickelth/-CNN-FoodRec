#!/usr/bin/env python
# coding: utf-8

# In[25]:


import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import optimizers
from keras.applications.vgg16 import VGG16
import tensorflow.keras
from tensorflow import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image


# In[2]:


def getPic(path):

    ImageDatas = []
    ImageDir = os.listdir(path)
    for ImageFile in ImageDir:
        fpath = os.path.join(path,ImageFile)
        img = Image.open(fpath)
        W = img.size[1]
        H = img.size[0]
        ImageDatas.append(np.array(img))
        img.close()  
    ImageDatas=np.array(ImageDatas)
    
    cnt = ImageDatas.shape[0]
    ImageDatas = ImageDatas.reshape((cnt, H, W, 3))
    print(ImageDatas.shape)
    
    return ImageDatas


# In[3]:


def getDataset(path):
    dset = np.empty([0,320,200,3],dtype = 'float32')
    dset = np.asarray(dset)
    foodlist = ['donuts','egg_tart','hamburger','ice_cream','pizza','steak']
    for i in range(6):
        tmpdir = os.path.join(path,foodlist[i])
        dset = np.concatenate((dset,getPic(tmpdir)),axis = 0)
    
    return dset


# In[4]:


def genePath(base_dir):
    train_dir = os.path.join(base_dir, 'train') 
    validation_dir = os.path.join(base_dir, 'validation') 
    test_dir = os.path.join(base_dir, 'test')
    
    print(base_dir)
    print(train_dir)
    print(validation_dir)
    print(test_dir)
    return train_dir,validation_dir,test_dir


# In[5]:


def getGenerator(data,path,batchsize,imagesize,flag):
    datagen = None
    if flag == 1:
        datagen = ImageDataGenerator(
                    rescale = 1./255,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    #以下4行做图像标准化
                    featurewise_center=True, 
                    featurewise_std_normalization=True,
                    samplewise_center=True, 
                    samplewise_std_normalization=True
                    )
    else:
        datagen = ImageDataGenerator(rescale = 1./255,
                                #以下4行做图像标准化
                                featurewise_center=True, 
                                featurewise_std_normalization=True,
                                samplewise_center=True, 
                                samplewise_std_normalization=True
                                )
    datagen.fit(data)

    datagenerator = datagen.flow_from_directory(
                    path,
                    target_size = (imgsize[0],imgsize[1]),
                    batch_size = batchsize,
                    )
    return datagenerator


# In[6]:


def freeze(conv_base,layer_name):
    print('trainable weights: ', len(conv_base.trainable_weights))
    conv_base.trainable = True
    
    
    f = False 
    for layer in conv_base.layers:
        if layer.name == layer_name:
            f = True
        if f == True:
            layer.trainable = True
        else:
            layer.trainable = False
    
    print('trainable weights: ', len(conv_base.trainable_weights))
    return conv_base


# In[7]:


def setCallback(mon,fac,pat1,pat2):
    callbacks_list = [
        tensorflow.keras.callbacks.ReduceLROnPlateau( # 不再改善时降低学习率
            monitor = mon,
            factor = fac,
            patience = pat1,
        ),
        tensorflow.keras.callbacks.EarlyStopping( # 不再改善时中断训练
            monitor = mon,
            patience = pat2,
        )
    ]
    return callbacks_list


# In[8]:


def modelTrain(model,train_generator,validation_generator,trainsize,valsize,batchsize,callbacks_list):
    model.compile(optimizer=optimizers.SGD(lr=1e-2),
                    loss='categorical_crossentropy',
                    metrics=['acc'])
    
    history = model.fit_generator(
                        train_generator,
                        steps_per_epoch=trainsize/batchsize,
                        epochs=300,
                        validation_data=validation_generator,
                        validation_steps=valsize/batchsize,
                        callbacks = callbacks_list)
    return history
    


# In[9]:


def plotAccLoss(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()


# In[10]:


def saveModel(model,model_version):
    s = 'food_recognize_model' + model_version + '.h5'
    print(s)
    model.save(s)


# In[11]:


def Predict(img_path,model):
    img = image.load_img(img_path,target_size = (320,200))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis = 0)
    preds = model.predict(x)
    plt.imshow(img)
    return preds


# In[12]:


#记得指定这是第几版模型
model_version = '19'
base_dir = 'D:\\food_dataset\\food_dataset'
imgsize = [320,200]
train_dir,validation_dir,test_dir = genePath(base_dir)


# In[13]:


X_train = getDataset(train_dir)
trainsize = X_train.shape[0]
X_validation = getDataset(validation_dir)
valsize = X_validation.shape[0]
X_test = getDataset(test_dir)


# In[14]:


batchsize = 30
imagesize = [320,200]
train_generator = getGenerator(X_train,train_dir,batchsize,imagesize,1)
validation_generator = getGenerator(X_validation,validation_dir,batchsize,imagesize,0)
test_generator = getGenerator(X_test,test_dir,batchsize,imagesize,0)


# In[15]:


conv_base =  VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=(imgsize[0], imgsize[1], 3))
conv_base = freeze(conv_base,'block3_conv1')


conv_base.summary()


# In[16]:


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())


model.add(layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l1_l2(l1 = 0.001, l2 = 0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l1_l2(l1 = 0.001, l2 = 0.001)))
model.add(layers.Dense(6, activation='softmax'))


# In[17]:


model.summary()


# In[18]:


callbacks_list = setCallback('val_loss',0.2,1,2)


# In[19]:


history = modelTrain(model,train_generator,validation_generator,trainsize,valsize,batchsize,callbacks_list)


# In[20]:


plotAccLoss(history)


# In[21]:


saveModel(model,model_version)


# In[22]:


model.evaluate_generator(test_generator)


# In[23]:


img_path = 'D:\\a eggtart.jpg'


# In[26]:


res = Predict(img_path,model)
print(res)


# In[ ]:




