import pandas as pd

# Load the CSV
df = pd.read_csv('FordPrices.csv')

#Clean and Preprocess Data
print("Cleaning and processing data")

#Check for missing/null values
print("\nChecking for missing Data")
print(df.isnull().sum())

#Check the types of data
print("\nDistinguishing the data types")
print(df.dtypes)

#Check for unique values
print("\nChecking for unique values")
print(df.nunique().sum())

# Print statistics
print("\n Printing Statistics")

# Keep 'year' column as string for presentation
df['year'] = df['year'].astype(str)

# Generate subset and convert 'year' back to int for calculations
stats = df[['year', 'price', 'mileage']].copy()
stats['year'] = stats['year'].astype(int)

# Generate raw summary
summary = stats.describe()

# Format numeric values with commas and appropriate decimals
formatted_stats = summary.copy()
for col in ['year', 'price', 'mileage']:
    if col == 'year':
        formatted_stats[col] = formatted_stats[col].map(lambda x: f"{int(x):,}")
    elif col == 'price':
        formatted_stats[col] = formatted_stats[col].map(lambda x: f"${x:,.2f}")
    else:
        formatted_stats[col] = formatted_stats[col].map(lambda x: f"{x:,.2f} Miles")

# Print final formatted table
print("\n Formatted Car Statistics (with Year as whole number):\n")
print(formatted_stats)


# Preview the data
#(df.to_string())


