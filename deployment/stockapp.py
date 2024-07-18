# Import the required libraries for the machine learning application.
import joblib
import numpy as np
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

# Unpickle Regressor and scaler.
scaler = joblib.load('C:/Users/512GB/OneDrive/Documents/Educative.IO/Educative projects/Time series with lstm and flask/scaler.pkl')
model = joblib.load('C:/Users/512GB/OneDrive/Documents/Educative.IO/Educative projects/Time series with lstm and flask/model1.pkl')

def predict_openprice(model, x):
    return model.predict(x)

def main():
    st.title('Predicting the open price')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Predicting the open price </h2>
    </div>
    """


    st.markdown(html_temp, unsafe_allow_html=True)
    close = st.number_input('Enter the close price')
    high = st.number_input('Enter the high')
    low = st.number_input('Enter the low')
    adj = st.number_input('adjusted_close price')
    volume = st.number_input('Enter the volume')

    # Store inputs into dataframe.
    X = [close,  high, low, adj, volume]
    X = np.array(X).reshape(1,-1)
    X = scaler.transform(X) # Transforming the input values

    # If button is pressed.
    if st.button("Predict"):
        result = predict_openprice(model, X)
        st.success("Open price is : {:.2f} mm".format(result[0]))        


if __name__=='__main__':
    main()