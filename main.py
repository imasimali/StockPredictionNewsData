from flask import Flask, request, jsonify
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)
model = keras.models.load_model("invisor2.h5",custom_objects={'KerasLayer':hub.KerasLayer})
text=pd.read_csv("data.csv")
num=pd.read_csv("AAPL.csv")

text['combined'] = text[text.columns[1:27]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)

text['combined'].replace("[^a-zA-Z]"," ",regex=True, inplace=True)
text['combined']=text['combined'].apply(lambda x : x.lower())

text=text['combined']

num=num['Close']

scaler2=MinMaxScaler(feature_range=(0,1))
num=num.values.reshape(-1,1)
num_s=scaler2.fit_transform(num)

@app.route('/')
def hello_world():
  return '<h1>Hello, World!</h1>'

@app.route("/get")
def get_prediction():
  # stock = request.args["stock"]
  newp=model.predict([text,num_s])
  y_new_inverse = scaler2.inverse_transform(newp)
  res=y_new_inverse[0][0]

  return 'Prediction is '.format(res)


app.run(host='0.0.0.0', port=8080)