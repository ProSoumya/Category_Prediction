import pickle
import requests
import numpy as np
import tensorflow as tf


from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences


#Intitiate the Flask App
app= Flask('Category Model Using Reviews Data')


#Initialize the global variables so that it can be use everywhere
global category_lable
global labels

#Initialize the Predict classes 
category_lable={'beauty':0,'grocery':1,'phone':2,'music':3,'office':4,'lawn':5,'toy':6,'pet':7}
labels = list(category_lable.keys())

#Initializing the max lenght of the test sequence and trunckation type and padding type 
max_length=120
trunc_type='post'
padding_type='post'

#Load the saved model
new_model= tf.keras.models.load_model('./models/Amazon_Review.h5')

#Load the tokenizer
with open('./models/tokenizer.pickle','rb') as handle:
    loaded_tokenizer=pickle.load(handle)


#User defined Predict function 
def Predict_category(txt):
    # print("*****************************************")
    text= {txt}
    # print(text)
    # print(f'String coverted is to {Str(txt)}' )
    seq=loaded_tokenizer.texts_to_sequences(text)
    # print('Sequence After Tokenizer',seq)
    # print('************************************************************')
    padded_seq=pad_sequences(seq,maxlen=max_length,truncating=trunc_type,padding=padding_type)
    # print('Padded Sequence After Padding ',padded_seq)
    # print('************************************************************')
    pred_seq= new_model.predict(padded_seq)
    # print(pred_seq)
    # print('************************************************************')
    return labels[np.argmax(pred_seq)]

@app.route('/')
def  home():
    return render_template('form.html')

@app.route('/result',methods=['POST'])
def result():
    if request.method =='POST':
        text=request.form.get('input')
        pred_final = Predict_category(text)
    return render_template('result.html',text=text, Predicted_Category= pred_final)

if __name__=='__main__':
    app.run(host='localhost',port=8000,debug=True) 