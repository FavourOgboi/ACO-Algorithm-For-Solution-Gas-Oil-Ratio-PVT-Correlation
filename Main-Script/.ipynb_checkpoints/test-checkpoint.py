import sys
import os
import numpy as np
import streamlit as st

# Adding the parent directory of 'Linear_Regression_Module' to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AntColony_PyCode.ant_colony_optimization import AntColonyOptimization
from pvt_Data.pvt_data import pvt_data



# Streamlit app
def main():
    st.title("GOR Prediction using ACO")

    # Input fields for parameters
    bubble_point_pressure = st.number_input("Enter Bubble Point Pressure:", min_value=0.0, value=0.0)
    api_gravity = st.number_input("Enter API Gravity:", min_value=0.0, value=0.0)
    gas_gravity = st.number_input("Enter Gas Gravity:", min_value=0.0, value=0.0)
    reservoir_temperature = st.number_input("Enter Reservoir Temperature:", min_value=0.0, value=0.0)

    # Button to predict GOR
    if st.button("Predict GOR"):
        estimated_gor = predict_gor_with_aco(pvt_data, bubble_point_pressure, api_gravity, gas_gravity, reservoir_temperature)
        
        if estimated_gor is not None:
            st.success(f"Estimated GOR using ACO: {estimated_gor:.2f}")
        else:
            st.error("Error in GOR estimation.")

if __name__ == "__main__":
    main()
