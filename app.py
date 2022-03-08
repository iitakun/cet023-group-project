#!/usr/bin/env python
# coding: utf-8

# In[5]:


from flask import Flask


# In[6]:


app = Flask(__name__)


# In[7]:


from flask import request, render_template
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        glucose = request.form.get("glucose")
        BMI = request.form.get("BMI")
        age = request.form.get("age")
        print(glucose, BMI, age)
        df = pd.read_csv("Diabetes.csv")
        df = df[['Glucose','BMI','Age']]
        updated_df=df.append({'Glucose':glucose, 'BMI':BMI, 'Age':age}, ignore_index=True)
        scaler = MinMaxScaler()
        scaler.fit(updated_df)
        scaled_updated_df = scaler.transform(updated_df)
        glucose_scaled = scaled_updated_df[-1,0]
        BMI_scaled = scaled_updated_df[-1,1]
        age_scaled = scaled_updated_df[-1,2]
        print(glucose_scaled, BMI_scaled, age_scaled)
        model = joblib.load("model_DT")
        pred = model.predict([[float(glucose_scaled), float(BMI_scaled), float(age_scaled)]])
        print(pred)
        if pred[0] == 0:
            decision = "No"
        elif pred[0] == 1:
            decision = "Yes"
        print(decision)
        s_dt = "Predicted probability of having diabetes based on Decision Tree model is: " + str(decision)
        model = joblib.load("model_RF")
        pred = model.predict([[float(glucose), float(BMI), float(age)]])
        print(pred)
        if pred[0] == 0:
            decision = "No"
        elif pred[0] == 1:
            decision = "Yes"
        print(decision)
        s_rf = "Predicted probability of having diabetes based on Random Forest model is: " + str(decision)
        model = joblib.load("model_NN")
        pred = model.predict([[float(glucose), float(BMI), float(age)]])
        print(pred)
        if pred[0] == 0:
            decision = "No"
        elif pred[0] == 1:
            decision = "Yes"
        print(decision)
        s_nn = "Predicted probability of having diabetes based on Neural Network model is: " + str(decision)
        return(render_template("index.html", result1=s_dt, result2=s_rf, result3=s_nn))
    else:
        return(render_template("index.html", result1="2", result2="2", result3="2"))


# In[ ]:


if __name__=="__main__":
    app.run()


# In[ ]:




