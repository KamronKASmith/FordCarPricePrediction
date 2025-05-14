import pandas as pd
# Load the CSV
df = pd.read_csv('../Data/CarPrice.csv')

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

#Check statistics
print("\nPrinting Statistics")
#Keeps the year column in string format
df['Year'] = df['Year'].astype(str)

#Generate statistics
stats = df[['Year', 'Selling_Price', 'KM_Driven']].copy()
stats['Year'] = df['Year'].astype(int)

summary = stats.describe()

#Format numeric values with commas and 2 decimals
formatted_stats = summary.copy()
for col in ['Year', 'Selling_Price', 'KM_Driven']:
    formatted_stats[col] = formatted_stats[col].map(lambda x: f"{int(x):,}" if col == 'Year' else f"{x:,.2f}")

# Print the formatted table
print("\nFormatted Car Statistics (with Year as string):\n")
print(formatted_stats)

# Preview the data
#print(df.to_string())


