#step -1 # Importing flask module in the project is mandatory 
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
#Step -2 Flask constructor takes the name of  
# current module (__name__) as argument.app = Flask(__name__)

app = Flask(__name__)

#Step -3 Load Trained  Model
model = pickle.load(open('Product.pkl', 'rb'))

transform = pickle.load(open('Product_t.pkl', 'rb'))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Step -4 The route() function of the Flask class is a decorator,  
# which tells the application which URL should call  
# the associated function


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    c  = [np.array(int_features)]
    c = transform.transform(c)
    result = model.predict(c)
    if result[0]==1:
        myresult="Purchased"
        
         
    else:
        myresult="Not Purchased"
        


    res=myresult

   

    return render_template('index.html', prediction_text='The Customer {} the product '.format(res))

# main driver function
 # run() method of Flask class runs the application  
    # on the local development server.
if __name__ == "__main__":
    app.run(debug=True)

