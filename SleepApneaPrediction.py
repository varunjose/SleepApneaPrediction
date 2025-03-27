from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import  MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
import pickle
from tensorflow.keras.applications import VGG19

main = tkinter.Tk()
main.title("Deep Learning of Facial Depth Maps for Obstructive Sleep Apnea Prediction")
main.geometry("1000x650")

global filename
global classifier
global X, Y
global model_acc
disease =['OSA Detected', 'No OSA Detected']

def upload():
    global filename
    global dataset
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    

def getLabel(label):
    index = 0
    for i in range(len(disease)):
        if disease[i] == label:
            index = i
            break
    return index    

def preprocess():
    global filename
    global X, Y
    text.delete('1.0', END)
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (64,64))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(64,64,3)
                    lbl = getLabel(name)
                    X.append(im2arr)
                    Y.append(lbl)
                    print(name+" "+root+"/"+directory[j]+" "+str(lbl))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Total dataset processed image size = "+str(len(X)))
    text.update_idletasks()
    test = X[3]
    test = cv2.resize(test,(400,400))
    cv2.imshow("Process Sampled image",test)
    cv2.waitKey(0)
    
    
def buildVGGModel():
    global X, Y
    global model_acc
    text.delete('1.0', END)
    global classifier
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()    
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()           
    else:
        vgg19 = VGG19(input_shape=(X.shape[1], X.shape[2], X.shape[3]), include_top=False, weights="imagenet")#defining VGG19 layer
        vgg19.trainable = False
        classifier = Sequential()
        classifier.add(vgg19)#transfer learning with VGG19
        classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) #define CNN layer with image input size as 64 X 64 with 3 RGB colours
        classifier.add(MaxPooling2D(pool_size = (2, 2))) #max pooling layer to collect filter data
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu')) #defining another layer to further filter images
        classifier.add(MaxPooling2D(pool_size = (2, 2))) #max pooling layer to collect filter data
        classifier.add(Flatten()) #convert images from 3 dimension to 1 dimensional array
        classifier.add(Dense(output_dim = 256, activation = 'relu')) #defining output layer
        classifier.add(Dense(output_dim = Y.shape[1], activation = 'softmax')) #this output layer will predict 1 disease from given 21 disease images
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) #compile cnn model
        hist = classifier.fit(X, Y, batch_size=16, epochs=10, shuffle=True, verbose=2) #build CNN model with given X and Y images
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    f = open('model/history.pckl', 'rb')
    model_acc = pickle.load(f)
    f.close()
    acc = model_acc['accuracy']
    accuracy = acc[3] * 100
    text.insert(END,"Propose VGG19 Prediction Accuracy : "+str(accuracy)+"\n\n")

       
def predict():
    global classifier
    text.delete('1.0', END)
    file = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(file)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict_disease = np.argmax(preds)
    img = cv2.imread(file)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Predicted Result : '+disease[predict_disease], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Predicted Result : '+disease[predict_disease], img)
    cv2.waitKey(0)
    
    
       
def graph():
    global model_acc
    accuracy = model_acc['accuracy']
    loss = model_acc['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['VGG-19 Accuracy', 'VGG-19 Loss'], loc='upper left')    
    plt.title('VGG-19 Accuracy & Loss Graph')
    plt.show()

    

font = ('times', 15, 'bold')
title = Label(main, text='Deep Learning of Facial Depth Maps for Obstructive Sleep Apnea Prediction', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload OSH Faces Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

vggButton = Button(main, text="Build VGG-19 Model", command=buildVGGModel)
vggButton.place(x=480,y=100)
vggButton.config(font=font1)


predictButton = Button(main, text="Upload Test Data & Predict OSH", command=predict)
predictButton.place(x=10,y=150)
predictButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=300,y=150)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
