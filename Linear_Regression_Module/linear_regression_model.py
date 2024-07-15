from sklearn.linear_model import LinearRegression
import numpy as np

def fit_linear_regression_model(pvt_data):
    X = np.array([[data['bubble_point_pressure'], data['api_gravity'], data['gas_gravity'], data['reservoir_temperature']] for data in pvt_data])
    y = np.array([data['actual_gor'] for data in pvt_data])
    
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_gor(model, bubble_point_pressure, api_gravity, gas_gravity, reservoir_temperature):
    try:
        estimated_gor = model.predict([[bubble_point_pressure, api_gravity, gas_gravity, reservoir_temperature]])
        return estimated_gor
    except Exception as e:
        print(f"Error: {e}")
        return None
