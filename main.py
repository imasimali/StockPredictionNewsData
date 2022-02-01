from flask import Flask, request, jsonify, json
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import urllib.request
from cherrypicker import CherryPicker

app = Flask(__name__)
run_with_ngrok(app)
CORS(app)

model = tf.keras.models.load_model("invisor2.h5",custom_objects={'KerasLayer':hub.KerasLayer})

@app.route('/')
def hello_world():
  return '<h1>Hello, World!</h1>'

@app.route("/get")
def get_prediction():
  stock = request.args["stock"]

  with urllib.request.urlopen("https://invisor.axemgit2.repl.co/api/allNews?stock="+stock) as url:
    Newsdata = json.loads(url.read().decode())
  
  picker = CherryPicker(Newsdata)
  flat = picker.flatten().get()
  textual_file = pd.DataFrame(flat)
  textual_file = textual_file.iloc[:15]

  with urllib.request.urlopen("https://invisor.axemgit2.repl.co/api/histData?stock="+stock) as url:
    num_json = json.loads(url.read().decode())
  num = pd.DataFrame(reversed(num_json))

  textual_file['combined'] = textual_file[textual_file.columns[1:27]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)

  textual_file['combined'].replace("[^a-zA-Z]"," ",regex=True, inplace=True)
  textual_file['combined']=textual_file['combined'].apply(lambda x : x.lower())

  textual_file=textual_file['combined']

  # num=num['close']
  # num = num.iloc[:15]

  scaler2=MinMaxScaler(feature_range=(0,1))
  num=num.values.reshape(-1,1)
  num_s=scaler2.fit_transform(num)

  newp=model.predict([textual_file,num_s])
  y_new_inverse = scaler2.inverse_transform(newp)
  res=y_new_inverse[0][0]
  prediction = json.dumps(res.item())

  return jsonify(prediction[:6])

app.run()
