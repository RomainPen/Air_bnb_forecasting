import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys
import xgboost
from xgboost import XGBRegressor
import shap


# Load the model :
model_folder = os.path.join(os.path.dirname(__file__), '..', 'MODEL')
model_path = os.path.join(model_folder, 'reg_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)


# import x_train for shapley value :
df_folder = os.path.join(os.path.dirname(__file__), '..', 'DATA')
df_path = os.path.join(df_folder, 'x_train.csv')
with open(df_path, 'r') as file:
    x_train = pd.read_csv(file, sep=";")


# Load the image : 
image_folder = os.path.join(os.path.dirname(__file__), '..', 'PPT_and_report')
image_path = os.path.join(image_folder, 'marseille_port.jpg')
image = Image.open(image_path) 


# Predict price :
def predict_location(individual_features) :
    df_for_pred = pd.DataFrame(individual_features)
    predict = model.predict(df_for_pred)[0]
    return predict


# Main function :
def main():
    # Image :
    st.image(image, caption='Port de Marseille')

    # Title :
    st.title('What is the rental price for a house in Marseille ? :house:')


    with st.sidebar :
        
        # Date :
        now = datetime.now()+ timedelta(hours=1)
        time = now.strftime("%H:%M:%S")
        st.write("Heure à Paris : ",time)
        
        # Features params :
        traveler = st.slider("number of traveler", 1, 15)
        bathroom = st.slider("number of bathroom", 0, 10)
        free_parking_on_site = st.select_slider("Free parking on site", [0,1])
        free_street_parking = st.select_slider("Free parking on street", [0,1])
        heating = st.select_slider("heating", [0,1])
        seaview = st.select_slider("seaview", [0,1])
        AC = st.select_slider("AC (climatisation)", [0,1])
        wifi = st.select_slider("wifi", [0,1])
        accepted_animals = st.select_slider("accepted animals", [0,1])
        tv = st.select_slider("tv", [0,1])
        microwave_oven = st.select_slider("microwave_oven", [0,1])
        smoker = st.select_slider("smoker", [0,1])
        backyard = st.select_slider("backyard", [0,1])
        workspace = st.select_slider("workspace", [0,1])
        private_garden = st.select_slider("private garden", [0,1])
        swimming_pool = st.select_slider("swimming pool", [0,1])
        surface = st.slider("surface (m2)",0,1000)
        transport_access = st.select_slider("transport access", [0,1])

    feature_dict = {'traveler': [traveler],'bathroom': [bathroom], 'free_parking_on_site': [free_parking_on_site],
                    'free_street_parking': [free_street_parking], 'heating': [heating], 'seaview': [seaview], 'AC': [AC],
                    'wifi': [wifi], 'accepted_animals': [accepted_animals], 'tv': [tv], 'microwave_oven': [microwave_oven], 
                    'smoker': [smoker], "backyard": [backyard], "workspace":[workspace], 'private_garden': [private_garden],
                    'swimming_pool': [swimming_pool], 'surface': [surface],'transport_access': [transport_access]}

    # Predict button :
    if st.button('Predict the rental price') :
        prediction = predict_location(feature_dict)
        
        # Print the prediction :
        st.success(prediction)


        # Explain the prediction with shapley method :
        
        st.subheader('Summary Plot')
        explainer = shap.Explainer(model.predict, x_train)
        shap_values = explainer(pd.DataFrame(feature_dict))
        
        shap.initjs()
        shap.plots.force(shap_values[0])

        fig2, ax = plt.subplots(nrows=1, ncols=1)
        shap.plots.waterfall(shap_values[0], max_display=20)
        st.pyplot(fig2)
        
        
    
# __name__ :
if __name__ == '__main__' :
    main()