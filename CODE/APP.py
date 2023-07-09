import streamlit as st
from PIL import Image

import pandas as pd
import pickle5 as pickle
import os

# Get the absolute path of the MODEL folder
model_folder = os.path.join(os.path.dirname(__file__), '..', 'MODEL')

# Load the trained model
model_path = os.path.join(model_folder, 'reg_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)


image_folder = os.path.join(os.path.dirname(__file__), '..', 'PPT_and_report')
image_path = os.path.join(image_folder, 'marseille_port.jpg')

image = Image.open(image_path) 
st.image(image, caption='Port de Marseille')

st.title('What is the rental price for a house in Marseille ? :house:')

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
                'free_street_parking': [free_street_parking], 'heating': [heating],
                'seaview': [seaview], 'AC': [AC],'wifi': [wifi], 'accepted_animals': [accepted_animals], 'tv': [tv],
                'microwave_oven': [microwave_oven], 
                'smoker': [smoker], "backyard":[backyard], "workspace":[workspace], 'private_garden': [private_garden],
                'swimming_pool': [swimming_pool], 'surface': [surface],'transport_access': [transport_access]}


#predict price :
def predict_location():
    df_for_pred = pd.DataFrame(feature_dict)
    predict = model.predict(df_for_pred)[0]
    
    return st.success(predict)
    
trigger = st.button('Predict', on_click=predict_location)
