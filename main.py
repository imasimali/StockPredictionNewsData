from flask import Flask, request, jsonify
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)
model = keras.models.load_model("invisor2.h5",custom_objects={'KerasLayer':hub.KerasLayer})

textual_file=pd.read_csv("data.csv")
num=pd.read_csv("AAPL.csv")

textual_file['combined'] = textual_file[textual_file.columns[1:27]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)

textual_file['combined'].replace("[^a-zA-Z]"," ",regex=True, inplace=True)
textual_file['combined']=textual_file['combined'].apply(lambda x : x.lower())

textual_file=textual_file['combined']
num=num['Close']

scaler2=MinMaxScaler(feature_range=(0,1))
num=num.values.reshape(-1,1)
num_s=scaler2.fit_transform(num)

newp=model.predict([textual_file,num_s])
y_new_inverse = scaler2.inverse_transform(newp)
res=y_new_inverse[0][0]
print("The Predicted value is : ",res)

@app.route('/')
def hello_world():
  return '<h1>Hello, World!</h1>'

@app.route("/get")
def get_prediction():
  # stock = request.args["stock"]
  # newp=model.predict([text,num_s])
  # y_new_inverse = scaler2.inverse_transform(newp)
  # res=y_new_inverse[0][0]

  return 'Prediction is '#.format(res)


app.run(host='0.0.0.0', port=8080)