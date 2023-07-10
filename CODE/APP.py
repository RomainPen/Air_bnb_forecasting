import streamlit as st
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
        surface = st.text_input("surface (m2)", "100")
        traveler = st.number_input("number of traveler", 1, 15)
        bathroom = st.slider("number of bathroom", 0, 10)
        
        private_garden = 0
        if st.checkbox("private garden") : 
            private_garden = 1
            
        workspace = 0
        if st.checkbox("workspace") : 
            workspace = 1   
            
        swimming_pool = 0
        if st.checkbox("swimming pool") : 
            swimming_pool = 1
            
        transport_access = 0
        if st.checkbox("transport access") : 
            transport_access = 1   

        free_parking_on_site = 0
        if st.checkbox("free parking on site") : 
            free_parking_on_site = 1
            
        free_street_parking = 0
        if st.checkbox("free street parking") : 
            free_street_parking = 1   
            
        smoker = 0
        if st.checkbox("smoker") : 
            smoker = 1
            
        accepted_animals = 0
        if st.checkbox("accepted animals") : 
            accepted_animals = 1

        AC = 0
        if st.checkbox("AC") : 
            AC = 1   
            
        wifi = 0
        if st.checkbox("wifi") : 
            wifi = 1
            
        tv = 0
        if st.checkbox("tv") : 
            tv = 1   

        microwave_oven = 0
        if st.checkbox("microwave oven") : 
            microwave_oven = 1
            
        heating = 0
        if st.checkbox("heating") : 
            heating = 1   
            
        backyard = 0
        if st.checkbox("backyard") : 
            backyard = 1
            
        seaview = 0
        if st.checkbox("seaview") : 
            seaview = 1


    feature_dict = {'traveler': [traveler],'bathroom': [bathroom], 'free_parking_on_site': [free_parking_on_site],
                    'free_street_parking': [free_street_parking], 'heating': [heating], 'seaview': [seaview], 'AC': [AC],
                    'wifi': [wifi], 'accepted_animals': [accepted_animals], 'tv': [tv], 'microwave_oven': [microwave_oven], 
                    'smoker': [smoker], "backyard": [backyard], "workspace":[workspace], 'private_garden': [private_garden],
                    'swimming_pool': [swimming_pool], 'surface': [float(surface)],'transport_access': [transport_access]}

    # Predict button :
    if st.button('Predict the rental price') :
        prediction = predict_location(feature_dict)
        
        # Print the prediction :
        st.success(prediction)

        # Explain the prediction with shapley method :
        st.subheader('Summary Plot')
        explainer = shap.Explainer(model.predict, x_train)
        shap_values = explainer(pd.DataFrame(feature_dict))

        fig, ax = plt.subplots(nrows=1, ncols=1)
        shap.plots.waterfall(shap_values[0], max_display=20)
        st.pyplot(fig)
        
        
    
# __name__ :
if __name__ == '__main__' :
    main()