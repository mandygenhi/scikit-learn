# Notes
# python3 ./run.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Data Collection
uk_renewable_energy_df = pd.read_csv('uk_renewable_energy.csv')
emissions_by_country_df = pd.read_csv('emissions_by_country.csv')

# Step 2: Data Preprocessing
# Assuming the data is already preprocessed and in the appropriate format
# No code snippet provided

# Step 3: Data Integration
merged_df = pd.merge(uk_renewable_energy_df, emissions_by_country_df, on='Year')

# Step 4: Data Analysis
# Example: Visualizing UK renewable energy consumption over time
plt.plot(uk_renewable_energy_df['Year'], uk_renewable_energy_df['Energy from renewable & waste sources'])
plt.xlabel('Year')
plt.ylabel('Energy from renewable & waste sources')
plt.title('UK Renewable Energy Consumption Over Time')
plt.show()

# Step 5: Correlation Analysis
correlation = merged_df['Energy from renewable & waste sources'].corr(merged_df['CO2 emissions'])

# Step 6: Time-Series Analysis
# Example: ARIMA modeling for UK renewable energy consumption
energy_ts = uk_renewable_energy_df.set_index('Year')['Energy from renewable & waste sources']
model = ARIMA(energy_ts, order=(1, 1, 1))  # Define ARIMA model parameters
model_fit = model.fit()  # Fit the ARIMA model
predictions = model_fit.predict(start=len(energy_ts), end=len(energy_ts)+n-1, typ='levels')  # Make predictions

# Step 7: Forecasting
n = 5  # Number of periods to forecast
forecasted_values = model_fit.forecast(steps=n)

# Step 8: Reporting and Visualization
plt.plot(uk_renewable_energy_df['Year'], uk_renewable_energy_df['Energy from renewable & waste sources'], label='Actual')
plt.plot(range(len(energy_ts), len(energy_ts) + n), forecasted_values, label='Forecast')
plt.xlabel('Year')
plt.ylabel('Energy from renewable & waste sources')
plt.title('UK Renewable Energy Consumption Forecast')
plt.legend()
plt.show()

# Print the forecasted values
print("Forecasted values:")
for i in range(len(forecasted_values)):
  print(f"Year {uk_renewable_energy_df['Year'].max() + i + 1}: {forecasted_values[i]}")
