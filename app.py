import numpy as np
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
df = pd.read_csv('rec.csv')
def recommend1(prize):
    lower = prize-5000
    upper = prize+5000
    filter = df[(df['price(in Rs.)'] < upper) & (df['price(in Rs.)'] > lower)]
    dic = {}
    cnt = 0
    while (cnt != 5):
        f1 = filter.sample(1)
        # print(f1['name'].values[0])
        if f1['name'].values[0] not in dic:
            cnt += 1
            dic[f1['name'].values[0]] = [f1['img_link'].values[0], f1['price(in Rs.)'].values[0]]
    return dic
# Load the model from the pickle file
with open('new_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction.html', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Get form data
        brand = request.form['brand']
        type_ = request.form['type']
        processor = request.form['processor']
        ram = int(request.form['ram'])
        os = request.form['os']
        touchscreen = int(request.form['touchscreen'])
        ips = int(request.form['ips'])
        ssd = int(request.form['ssd'])
        hdd = int(request.form['hdd'])
        gpu = request.form['gpu']
        weight = float(request.form['weight'])

        # Create a feature vector for prediction
        features = [brand, type_, processor, ram, os, weight ,touchscreen, ips, ssd, hdd, gpu]

        print(features)

        # Convert categorical features to numerical values if necessary
        # This conversion should match the preprocessing steps used during model training

        # Predict the price using the loaded model
        predicted_price = model.predict([features])[0]
        predicted_price = np.round(np.exp(predicted_price),2)

        top5 = recommend1(np.round(predicted_price))

        list = []


        for i, j in top5.items():
            name = i
            link = j[0]
            price = j[1]
            l = [name , link , price]
            list.append(l)


    #     # Render the prediction result
    #     return render_template('prediction.html', result=predicted_price)
    # else:
    #     return render_template('prediction.html')

        return render_template('prediction.html', result=predicted_price, list = list ,  form_data=request.form)
    else:
        return render_template('prediction.html')



@app.route('/index.html')
def homepage():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
