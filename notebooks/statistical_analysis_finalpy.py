#Traffic Volume: Statistical and Categorical Data Analysis.


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

# Work whether the script is launched from VS Code or a terminal.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'traffic_clean.csv'


# %% Load cleaned dataset
data = pd.read_csv(DATA_FILE)

print(data.head())


# %% Prepare the five numerical variables
# Convert date_time into datetime format
data['date_time'] = pd.to_datetime(data['date_time'])

# Extract the hour of the day (0–23)
data['hour'] = data['date_time'].dt.hour

# Select our five numerical variables
variables = ['temp_f', 'rain_1h', 'clouds_all',
             'hour', 'traffic_volume']

analysis = data[variables].copy()

# Clear names and units for use in graphs (dataset column names remain unchanged)
variable_labels = {
    'temp_f': 'Temperature (°F)',
    'rain_1h': 'Rainfall in One Hour (mm)',
    'clouds_all': 'Cloud Coverage (%)',
    'hour': 'Hour of Day (24-Hour Clock)',
    'traffic_volume': 'Traffic Volume (Vehicles per Hour)'
}

# Short versions fit more clearly on the two heatmaps
heatmap_labels = ['Temp (°F)', 'Rain (mm)', 'Clouds (%)',
                  'Hour (0–23)', 'Traffic (veh/h)']


print(analysis.head())


# %% Descriptive statistics
# Select rows with complete observations
analysis_valid = analysis.dropna()

# Calculate descriptive statistics
statistics = pd.DataFrame({
    'Mean': analysis_valid.mean(),
    'Median': analysis_valid.median(),
    'Variance': analysis_valid.var(),
    'Std Dev': analysis_valid.std(),
    'Minimum': analysis_valid.min(),
    'Maximum': analysis_valid.max()
})

print(statistics.round(3))


# %% Z-scores and outlier counts
# Calculate Z-scores for all five variables
z_scores = (analysis_valid - analysis_valid.mean()) / analysis_valid.std()

# Display the first five Z-scores
print(z_scores.head().round(3))

# Count observations with absolute Z-scores greater than 3
outliers = (z_scores.abs() > 3).sum()

print("\nNumber of potential outliers:")
print(outliers)


# %% Covariance matrix
# Calculate the covariance matrix
cov_matrix = analysis_valid.cov()

# Display the covariance matrix
print(cov_matrix.round(3))


# %% Covariance heatmap

# Adjust the color scale
limit = abs(cov_matrix).to_numpy().max()

plt.figure(figsize=(10, 7))

plt.imshow(cov_matrix, cmap='coolwarm',
           norm=SymLogNorm(linthresh=1, vmin=-limit, vmax=limit))

plt.colorbar(label='Covariance (SymLog Scale)')

# Label the variables
plt.xticks(range(5), heatmap_labels, rotation=45)
plt.yticks(range(5), heatmap_labels)

# Display covariance values inside each square
for i in range(5):
    for j in range(5):
        value = cov_matrix.iloc[i, j]
        plt.text(j, i, f'{value:.3g}',
                 ha='center', va='center', fontsize=9,
                 color='white' if abs(value) > 100000 else 'black')

plt.title('Sample Covariance Heatmap')
plt.tight_layout()
plt.savefig('../figures/smp_covHM.png', dpi=150)
plt.show(block=True)


# %% Correlation matrix
# Calculate the correlation matrix
corr_matrix = analysis_valid.corr()

# Display the correlation matrix
print(corr_matrix.round(3))


# %% Correlation heatmap
# Create the correlation heatmap
plt.figure(figsize=(10, 7))

plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Add color scale
plt.colorbar(label='Correlation')

# Label the variables
plt.xticks(range(5), heatmap_labels, rotation=45)
plt.yticks(range(5), heatmap_labels)

# Display correlation values inside each square
for i in range(5):
    for j in range(5):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                 ha='center', va='center', fontsize=9)

plt.title('Pearson Correlation Heatmap')
plt.tight_layout()
plt.show(block=True)


# %% Histograms
# Create a separate histogram for each variable
for column in variables:
    plt.figure(figsize=(8, 5))

    plt.hist(analysis_valid[column], bins=20, edgecolor='black')

    plt.title(f'Histogram of {variable_labels[column]}')
    plt.xlabel(variable_labels[column])
    plt.ylabel('Number of Hourly Observations')

    plt.tight_layout()
    plt.show(block=True)


# Additional histogram showing only rainy hours
rainy_hours = analysis_valid[analysis_valid['rain_1h'] > 0]

plt.figure(figsize=(8, 5))

plt.hist(rainy_hours['rain_1h'], bins=30, edgecolor='black')

plt.title('Rainfall Distribution During Rainy Hours')
plt.xlabel('Rainfall in One Hour (mm)')
plt.ylabel('Number of Hourly Observations')

plt.tight_layout()
plt.show(block=True)


# %% Boxplots
# Create a separate boxplot for each variable
for column in variables:
    plt.figure(figsize=(8, 5))

    plt.boxplot(analysis_valid[column])
    plt.xticks([])  # Only one box; its x-position has no meaning

    plt.title(f'Boxplot of {variable_labels[column]}')
    plt.ylabel(variable_labels[column])

    plt.tight_layout()
    plt.show(block=True)


# Additional boxplot showing only rainy hours
rainy_rainfall = analysis_valid.loc[
    analysis_valid['rain_1h'] > 0, 'rain_1h'
]

plt.figure(figsize=(8, 5))

plt.boxplot(rainy_rainfall, showfliers=False)
plt.xticks([])  # Only one box; its x-position has no meaning

plt.title('Rainfall During Rainy Hours (Outliers Hidden)')
plt.ylabel('Rainfall in One Hour (mm)')

plt.tight_layout()
plt.show(block=True)


# %% Scatterplots
# Variables to compare against traffic volume
x_variables = ['temp_f', 'rain_1h', 'clouds_all', 'hour']

# Create a separate scatterplot for each variable
for column in x_variables:
    plt.figure(figsize=(8, 5))

    plt.scatter(analysis_valid[column],
                analysis_valid['traffic_volume'],
                s=5, alpha=0.2)

    plt.title(f'{variable_labels[column]} vs Traffic Volume')
    plt.xlabel(variable_labels[column])
    plt.ylabel('Traffic Volume (Vehicles per Hour)')

    plt.tight_layout()
    plt.show(block=True)


# %% Hourly and daily traffic line graphs
# Calculate average traffic volume for each hour
hourly_traffic = analysis_valid.groupby('hour')['traffic_volume'].mean()

# Create line graph
plt.figure(figsize=(10, 6))

plt.plot(hourly_traffic.index, hourly_traffic.values,
         marker='o')

plt.title('Average Traffic Volume by Hour')
plt.xlabel('Hour of Day (24-Hour Clock)')
plt.ylabel('Average Traffic Volume (Vehicles per Hour)')

plt.xticks(range(0, 24))
plt.grid(True)

plt.tight_layout()
plt.show(block=True)



# Calculate average daily traffic volume
daily_traffic = data.groupby(
    data['date_time'].dt.date
)['traffic_volume'].mean()

# Create daily traffic line graph
plt.figure(figsize=(12, 6))

plt.plot(daily_traffic.index, daily_traffic.values)

plt.title('Average Daily Traffic Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Average Traffic Volume (Vehicles per Hour)')

plt.grid(True)
plt.tight_layout()
plt.show(block=True)


# %% Categorical frequency tables
# Categorical Data Analysis

# Display weather condition frequencies
print("Weather Descriptions:")
print(data['weather_description'].value_counts())

# Display holiday frequencies
print("\nHolidays:")
print(data['holiday'].value_counts())


# %% Traffic volume by weather description
# Select observations with valid weather and traffic data
weather_data = data.dropna(subset=['weather_description', 'traffic_volume'])

# Select the 10 most common weather descriptions
top_weather = weather_data['weather_description'].value_counts().head(10).index

# Calculate average traffic volume for each weather condition
weather_traffic = weather_data[
    weather_data['weather_description'].isin(top_weather)
].groupby('weather_description')['traffic_volume'].mean()

# Create bar chart
plt.figure(figsize=(10, 6))

weather_traffic.sort_values().plot(kind='barh')

plt.title('Average Traffic Volume by Weather Condition')
plt.xlabel('Average Traffic Volume (Vehicles per Hour)')
plt.ylabel('Weather Condition')

plt.tight_layout()
plt.show(block=True)


# %% Holiday versus non-holiday traffic
# Select observations with valid traffic data
holiday_data = data.dropna(subset=['traffic_volume']).copy()

# Identify holidays and non-holidays
holiday_data['holiday_status'] = holiday_data['holiday'].apply(
    lambda x: 'Non-Holiday' if x == 'No Holiday' else 'Holiday'
)

# Calculate average traffic volume
holiday_traffic = holiday_data.groupby('holiday_status')['traffic_volume'].mean()

print(holiday_traffic.round(2))

# Display the number of observations
print("\nNumber of Observations:")
print(holiday_data['holiday_status'].value_counts())

# Create bar chart
plt.figure(figsize=(8, 5))

holiday_traffic.plot(kind='bar', edgecolor='black')

plt.title('Average Traffic Volume: Holiday vs Non-Holiday')
plt.xlabel('Day Type')
plt.ylabel('Average Traffic Volume (Vehicles per Hour)')

plt.xticks(rotation=0)
plt.tight_layout()
plt.show(block=True)


# %% Rush-hour traffic by weather conditions
# Select observations with valid weather and traffic data
rush_data = data.dropna(
    subset=['weather_description', 'traffic_volume']
).copy()

# Keep only weekdays (Monday through Friday)
rush_data = rush_data[rush_data['date_time'].dt.dayofweek < 5]

# Extract the hour
rush_data['hour'] = rush_data['date_time'].dt.hour

# Keep morning and afternoon rush hours
rush_data = rush_data[
    rush_data['hour'].between(6, 8) |
    rush_data['hour'].between(15, 17)
].copy()

# Label the two rush-hour periods
rush_data['rush_period'] = rush_data['hour'].apply(
    lambda h: 'Morning' if h < 9 else 'Afternoon'
)

# Select five weather conditions
weather_types = [
    'sky is clear',
    'light rain',
    'moderate rain',
    'light snow',
    'mist'
]

rush_data = rush_data[
    rush_data['weather_description'].isin(weather_types)
]

# Calculate average traffic and observation counts
rush_summary = rush_data.groupby(
    ['weather_description', 'rush_period']
)['traffic_volume'].agg(['mean', 'count'])

# Arrange the results into tables
rush_averages = rush_summary['mean'].unstack().reindex(weather_types)
rush_counts = rush_summary['count'].unstack().reindex(weather_types)

print("Average Traffic Volume:")
print(rush_averages.round(2))

print("\nNumber of Observations:")
print(rush_counts)

# Create grouped bar chart
rush_averages.plot(kind='bar', figsize=(10, 6), edgecolor='black')

plt.title('Weekday Rush-Hour Traffic by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Average Traffic Volume (Vehicles per Hour)')
plt.xticks(rotation=30)
plt.legend(title='Rush Hour')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show(block=True)


# %% Weekday versus weekend traffic
# Select observations with valid traffic data
weekday_data = data.dropna(subset=['traffic_volume']).copy()

# Identify weekdays and weekends
weekday_data['day_type'] = weekday_data['date_time'].dt.dayofweek.apply(
    lambda day: 'Weekend' if day >= 5 else 'Weekday'
)

# Calculate average traffic volume by hour and day type
hourly_comparison = weekday_data.groupby(
    ['hour', 'day_type']
)['traffic_volume'].mean().unstack()

# Create line graph
hourly_comparison.plot(figsize=(10, 6), marker='o')

plt.title('Average Hourly Traffic: Weekdays vs Weekends')
plt.xlabel('Hour of Day (24-Hour Clock)')
plt.ylabel('Average Traffic Volume (Vehicles per Hour)')

plt.xticks(range(0, 24))
plt.grid(True)

plt.tight_layout()
plt.show(block=True)
