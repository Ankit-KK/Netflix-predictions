import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
import plotly.graph_objs as go




Data = pd.read_csv('Netflix-Subscriptions.csv')
Data['Time Period'] = pd.to_datetime(Data['Time Period'],
                                     format='%d/%m/%Y')
time_series = Data.set_index('Time Period')['Subscribers']


# Define the ARIMA model parameters
p, d, q = 1, 1, 1

# Fit the ARIMA model

model = ARIMA(time_series, order=(p, d, q))
results = model.fit()

# Streamlit app
st.set_page_config(page_title="Netflix Quarterly Subscription Predictions")

st.markdown("<h1 style='text-align: center; color: #333; font-size: 36px; font-weight: bold; margin-bottom: 20px;'>Netflix Quarterly Subscription Predictions</h1>", unsafe_allow_html=True)

# Input field for number of quarters
quarters = st.number_input('Enter the number of quarters:', min_value=1, value=1)

# Predict button
if st.button('Predict'):
    if quarters is None:
        st.warning("Please enter the number of quarters.")
    else:
        # Make predictions based on user input
        future_steps = quarters
        predictions = results.predict(start=len(time_series), end=len(time_series) + future_steps - 1)
        predictions = predictions.astype(int)

        # Create a DataFrame with the original data and predictions
        forecast = pd.DataFrame({'Original': time_series, 'Predictions': predictions})

        # Initialize the figure
        fig = go.Figure()

        # Add trace for predictions
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast['Predictions'],
            mode='lines',
            name='Predictions'
        ))

        # Add trace for original data
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast['Original'],
            mode='lines',
            name='Original Data'
        ))

        # Update layout
        fig.update_layout(
            title='Netflix Quarterly Subscription Predictions',
            xaxis_title='Time Period',
            yaxis_title='Subscribers',
            legend=dict(x=0.1, y=0.9),
            showlegend=True
        )

        # Display the plot
        st.plotly_chart(fig)
