from tkinter import *
from PIL import Image, ImageTk
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from matplotlib.image import imread
from os import listdir
from numpy import asarray
from keras.utils.image_utils import load_img
from keras.utils.image_dataset import load_image
from numpy import asarray
from numpy import save
from keras.utils import load_img, img_to_array
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Normalization,LeakyReLU
from keras.optimizers import Adam
from keras import models
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import subprocess

def callback(event):
        if event.keysym == 'Return':
                load_image()
def pdf():
        global r
        if r ==0:
                label3.configure(text='Please choose an image!!!',font = ('Verdana', 15))
        if r ==1:
                subprocess.Popen(['Bệnh đốm vi khuẩn(Bacterial_spot).pdf'], shell=True)
        if r ==2:
                subprocess.Popen(['Bệnh bạc lá sớm(Early_blight).pdf'], shell=True)
        if r ==4:
                subprocess.Popen(['Bệnh sương mai(Late_blight).pdf'], shell=True)
        if r ==5:
                subprocess.Popen(['Khuôn lá(Leaf_Mold).pdf'], shell=True)
        if r ==6:
                subprocess.Popen(['Đốm lá Septoria(Septoria_leaf_spot).pdf'], shell=True)
        if r ==7:
                subprocess.Popen(['Nhện Ve Nhện hai đốm(Spider_mites Two-spotted_spider_mite).pdf'], shell=True)
        if r ==8:
                subprocess.Popen(['Đốm(Target_Spot).pdf'], shell=True)
        if r ==9:
                subprocess.Popen(['Virus khảm cà chua(Tomato_mosaic_virus).pdf'], shell=True)
        if r ==10:
                subprocess.Popen(['Virus xoăn vàng lá cà chua(Tomato_Yellow_Leaf_Curl_Virus).pdf'], shell=True)
def load_image():
        global r
        if title.get()=='':
                label3.configure(text='Please choose an image!!!',font = ('Verdana', 15))
        else:
                image_path = title.get()
                image = Image.open(image_path)
                image = image.resize((260, 260))
                image_tk = ImageTk.PhotoImage(image)
                label.config(image=image_tk)
                label.image = image_tk
                title.delete(0,END)
                model5= load_model('TOMATO (2).h5')
                img = plt.imread(image_path)
                plt.imshow(img)
                img = load_img(image_path, target_size = (120,120))
                plt.imshow(img)
                img = img_to_array(img)
                img = img.reshape(1, 120,120,3)
                img = img.astype('float32')
                img = img/255
                u = np.argmax(model5.predict(img), axis = -1)
                data = {1:'Bệnh đốm vi khuẩn(Bacterial_spot)', 2:'Bệnh bạc lá sớm(Early_blight)', 3:'Cây khỏe mạnh(healthy)', 4:'Bệnh sương mai(Late_blight)',
                5:'Khuôn lá(Leaf_Mold)', 6:'Đốm lá Septoria(Septoria_leaf_spot)', 7:'Nhện Ve Nhện hai đốm(Spider_mites Two-spotted_spider_mite)', 8:'Đốm(Target_Spot)',
                9:'Virus khảm cà chua(Tomato_mosaic_virus)', 10:'Virus xoăn vàng lá cà chua(Tomato_Yellow_Leaf_Curl_Virus)'} 
                data[u[0]]
                label4.configure(text = data[u[0]])
                r = int(u[0])
 
def open_file():
    file_path = filedialog.askopenfilename()
    entry_var.set(file_path)
    label3.configure(text='')
r =0
w = Tk()
entry_var = StringVar()
open_button = Button(w, text="Open file",font = ('Verdana', 12), command=open_file)
open_button.place(x = 50, y = 110)
w.title('Project_AI')
w.bind('<Key>', callback)
w.geometry("1050x350")
title = Entry(w,textvariable=entry_var,font = ('Verdana', 15))
title.place(x = 50, y = 70)
label = Label(w)
label.place(x = 550, y = 20)
label3 = Label(w,font = ('Verdana', 11))
label3.place(x = 50, y = 300)
label4 = Label(w,font = ('Verdana', 11))
label4.place(x = 550, y = 300)
label5 = Label(w,text='Name: Trần Văn Danh',font = ('Verdana', 15))
label5.place(x = 50, y = 170)
label7 = Label(w,text='ID: 20146233',font = ('Verdana', 15))
label7.place(x = 50, y = 220)
label6 = Label(w, text='Tomato Detection using CNN',font = ('Verdana', 15))
label6.place(x = 40, y = 10)
Button1 = Button(w, text="Check",font = ('Verdana', 12), command=load_image)
Button1.place(x = 350, y = 70)

Button2 = Button(w, text="Solution",font = ('Verdana', 12), command=pdf)
Button2.place(x = 890, y = 150)
w.mainloop()
