import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import nbformat
from nbconvert import HTMLExporter
import sys
import os


from AntColony_PyCode.ant_colony_optimization import AntColonyOptimization
from Linear_Regression_Module.linear_regression_model import fit_linear_regression_model, predict_gor
# getting PVT data
from pvt_Data.pvt_data import pvt_data

def predict_gor_with_aco(pvt_data, bubble_point_pressure, api_gravity, gas_gravity, reservoir_temperature):
    # Adjust ACO parameters for better tuning
    num_ants = 20  # Increase number of ants for better exploration
    num_iterations = 200  # More iterations for convergence
    decay = 0.95
    alpha = 1.0
    beta = 2.0

    # Implement a new ACO instance and run
    aco = AntColonyOptimization(pvt_data, num_ants=num_ants, num_iterations=num_iterations, decay=decay, alpha=alpha, beta=beta)
    shortest_path, _ = aco.run()

    # Extract GOR values corresponding to the optimized parameters from the shortest path found
    optimized_gor_values = [pvt_data[i]['actual_gor'] for i in shortest_path]

    # Calculate a weighted predicted GOR based on the distance from the input parameters
    weights = []
    for i in shortest_path:
        distance = (
            abs(bubble_point_pressure - pvt_data[i]['bubble_point_pressure']) +
            abs(api_gravity - pvt_data[i]['api_gravity']) +
            abs(gas_gravity - pvt_data[i]['gas_gravity']) +
            abs(reservoir_temperature - pvt_data[i]['reservoir_temperature'])
        )
        weights.append(1 / (1 + distance))  # Weight inversely proportional to distance

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Calculate the weighted average of the GOR values
    predicted_gor = sum(g * w for g, w in zip(optimized_gor_values, normalized_weights))

    return predicted_gor

# Function to convert notebook to HTML
def convert_notebook_to_html(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    html_exporter = HTMLExporter()
    html_exporter.exclude_input = False  # Include input cells
    body, _ = html_exporter.from_notebook_node(nb)
    return body

# Streamlit app structure
st.sidebar.title("Explore")
selected = st.sidebar.radio("Navigation", ["Introduction", "Volvo ACO Algorithm", "PVT ACO Algorithm", "Data Analysis", "GOR Calculation", "Institution Student Project"])

if selected == "Introduction":
    st.title('Ant Colony Optimization for GOR Calculation')
    st.image("C:/Users/hp/Documents/GitHub/ACO-Algorithm-For-Solution-Gas-Oil-Ratio-PVT-Correlation/Image/1701501705964.jfif", caption="Project Overview")
    
    st.write("""
    Hello, this is my research, which aims to investigate the creative use of the Ant Colony Optimization (ACO) algorithm to determine the Gas-Oil Ratio (GOR). GOR is a crucial metric in the oil and gas sector that has a big impact on both economic viability and production efficiency.
    
    My aim to use cutting-edge computational approaches to improve the accuracy and dependability of GOR estimations gave rise to this study. I see a chance to close the gap caused by traditional approaches' inability to adjust to the various factors involved in gas-oil interactions. I want to create a strong framework for GOR estimation by utilizing the capabilities of ACO, a bio-inspired optimization algorithm that emulates the ants' natural food-finding activity. In addition to increasing accuracy, I want to provide a more dynamic method of problem-solving.
    
    Extensive research and testing with real-world datasets designed to capture the nuances of gas and oil properties under various reservoir circumstances form the foundation of my work. I have used two main datasets: a thorough PVT sample dataset that offers crucial insights into the dynamics of the gas-oil ratio, and the Volvo dataset, which includes important automotive sector metrics.
    
    I encourage you to interact with the several components of this program that will help you comprehend the ACO algorithm, investigate the datasets, analyze the data, and eventually take part in the GOR computation process. Every element is intended to be instructive and easy to use, taking you step-by-step through the process of understanding and successfully utilizing this sophisticated algorithm.
    
    I invite you to learn more about my project, which combines innovation and real-world application in the oil and gas sector. Come along with me as we transform the calculation of GOR and advance our knowledge of reservoir dynamics by using clever data analysis and optimization strategies.
    """)

# ACO Explanation Section
elif selected == "Volvo ACO Algorithm":
    st.title('Ant Colony Optimization (ACO) On Algorithm Volvo Data')
    st.write("""
    The Ant Colony Optimization (ACO) algorithm is applied to the Volvo dataset, which contains critical parameters for production analysis. This dataset includes:
    
    - **Date of Production**: The date on which the production data was recorded.
    - **Wellbore Name**: The identifier for the wellbore being analyzed.
    - **On Stream Hours**: The total hours the well has been actively producing.
    - **Average Downhole Pressure**: The average pressure recorded downhole during production.
    - **Average Downhole Temperature**: The average temperature recorded downhole.
    - **Average Tubing Pressure**: The average pressure in the tubing during production.
    - **Average Annulus Pressure**: The pressure in the annulus surrounding the wellbore.
    - **Average Choke Size**: The size of the choke used in the production process.
    - **Average Wellhead Pressure**: The pressure measured at the wellhead.
    - **Average Wellhead Temperature**: The temperature at the wellhead.
    - **Choke Size**: The size of the choke at the time of measurement.
    - **Bore Oil Volume**: The volume of oil produced.
    - **Bore Gas Volume**: The volume of gas produced.
    - **Bore Water Volume**: The volume of water produced.
    - **Flow Kind**: The type of flow (e.g., production).
    - **Well Type**: The classification of the well (e.g., OP for operating).
    - **Calculated GOR**: The gas-to-oil ratio calculated from the produced volumes.

    Using ACO on this dataset allows for effective optimization of production strategies, enhancing decision-making and operational efficiency in the oil production industry.
    """)
    
    # Path to the ACO notebook
    aco_notebook_path = "C:/Users/hp/Documents/GitHub/ACO-Algorithm-For-Solution-Gas-Oil-Ratio-PVT-Correlation/volvo-data-ACO-algorithm/Volvo-ACO-Algorithm-Explained.ipynb"
    
    # Convert and display the ACO notebook
    try:
        aco_notebook_html = convert_notebook_to_html(aco_notebook_path)
        st.components.v1.html(aco_notebook_html, height=1000, scrolling=True)
    except Exception as e:
        st.error(f"An error occurred while displaying the ACO notebook: {e}")

elif selected == "PVT ACO Algorithm":
    st.title('Ant Colony Optimization (ACO) On Algorithm PVT Sample Data')
    st.write("""
    The Ant Colony Optimization (ACO) algorithm is also utilized with the PVT dataset, which is essential for understanding gas-oil relationships in reservoir management. This dataset consists of the following parameters:
    
    - **Bubble Point Pressure**: The pressure at which gas begins to come out of the solution in the oil.
    - **API Gravity**: A measure of how heavy or light petroleum liquid is compared to water.
    - **Gas Gravity**: The ratio of the density of gas to the density of air.
    - **Reservoir Temperature**: The temperature within the reservoir.
    - **Actual GOR**: The actual gas-to-oil ratio at the given conditions.

    The ACO algorithm enhances the modeling of gas-oil interactions by optimizing the estimation of GOR based on these parameters. This leads to improved predictions and more effective recovery strategies, ultimately benefiting reservoir management and production planning.
    """)
    
    # Path to the ACO notebook
    aco_notebook_path = "C:/Users/hp/Documents/GitHub/ACO-Algorithm-For-Solution-Gas-Oil-Ratio-PVT-Correlation/IpynbFolder/ACO_Algorithm_Steps_Explained.ipynb"
    
    # Convert and display the ACO notebook
    try:
        aco_notebook_html = convert_notebook_to_html(aco_notebook_path)
        st.components.v1.html(aco_notebook_html, height=1000, scrolling=True)
    except Exception as e:
        st.error(f"An error occurred while displaying the ACO notebook: {e}")

# Data Analysis Section
elif selected == "Data Analysis":
    st.title('Data Analysis')
    st.write("""
    In this section, I conduct an exploratory data analysis (EDA) on the Volvo dataset to uncover key insights and trends related to production performance and efficiency.
    """)

    # Path to the Data Analysis notebook
    data_analysis_notebook_path = 'C:/Users/hp/Documents/GitHub/ACO-Algorithm-For-Solution-Gas-Oil-Ratio-PVT-Correlation/volvo-data-ACO-algorithm/Exploratory-Data-Analysis-Volvo.ipynb'

    # Convert and display the Data Analysis notebook
    try:
        data_analysis_notebook_html = convert_notebook_to_html(data_analysis_notebook_path)
        st.components.v1.html(data_analysis_notebook_html, height=1000, scrolling=True)
    except Exception as e:
        st.error(f"An error occurred while displaying the Data Analysis notebook: {e}")

elif selected == "GOR Calculation":
    st.title('GOR Calculation Process')
    
    st.write("he Gas-to-Oil Ratio (GOR) is a critical measure in the oil and gas sector, indicating the relationship between gas and oil production. High GOR values can signify changes in reservoir conditions, while low GOR values often reflect efficient oil extraction.")

    st.write("To get started, please enter the required parameters: Bubble Point Pressure, API Gravity, Gas Gravity, and Reservoir Temperature. Our tool will provide you with an estimated GOR, empowering you to make informed decisions about your production strategies.")

    st.write("Understanding and monitoring GOR trends is essential for optimizing reservoir performance and enhancing recovery operations.")

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

elif selected == "Institution Student Project":
    st.title('Institution')
    st.image("C:/Users/hp/Documents/GitHub/ACO-Algorithm-For-Solution-Gas-Oil-Ratio-PVT-Correlation/Image/fupre_logo.jpg", caption="FUPRE Logo")
    st.write("""This project is part of a final year project at the Federal University of Petroleum Resources Effurun (FUPRE).
    It aims to leverage advanced algorithms for practical applications in the oil and gas industry.""")