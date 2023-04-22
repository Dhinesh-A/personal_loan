import flask
from flask import Flask, render_template,request
import pickle
import numpy as np
import sklearn
model = pickle.load(open('modelrf.pkl','rb'))
app= Flask(__name__)
@app.route('/')
def page1():
    return render_template('page1.html')
@app.route('/page2')
def page2():
    return render_template('page2.html')
@app.route('/getdata',methods=['POST','GET'])
def pred():
    loadid = request.form['Loan_ID']
    print(loadid)
    gender = request.form['Gender']
    print(loadid, gender)
    married = request.form['Married']
    dependents = request.form['Dependents']
    education = request.form['Education']
    selfemployed = request.form['Self_Employed']
    application = request.form['ApplicantIncome']
    coapplicantincome = request.form['CoapplicantIncome']
    loanamount = request.form['LoanAmount']
    loanamountterm = request.form['Loan_Amount_Term']
    credithistroy = request.form['Credit_History']
    proppertyarea = request.form['Property_Area']
    int_features = [[np.log(float(loadid)), int(gender), int(married),
                     int(dependents), int(education), int(selfemployed),
                     int(application), np.log(float(coapplicantincome)), np.log(float(loanamount)),
                     np.log(float(loanamountterm)), np.log(float(credithistroy)), int(proppertyarea)]]
    print(int_features)
    prediction = model.predict(int_features)
    print(type(prediction))
    t = prediction[0]
    print(t)
    if t > 0.5:
        prediction_text = 'you are eligible to loan,Loan will be sanctioned'
    else:
        prediction_text = 'you are Not eligible to loan'
    print(prediction_text)
    return render_template('display.html',prediction_results = prediction_text)
if __name__=="__main__":
    app.run(debug=True)