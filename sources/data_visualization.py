import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("FordPrices.csv")

def plot_price_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['price'], bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Car Prices")
    plt.xlabel("Car Price ($)")
    plt.ylabel("Car Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_price_vs_mileage(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['mileage'], df['price'], alpha=0.5, color='tomato')
    plt.title("Price vs Mileage")
    plt.xlabel("Mileage")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_price_by_model(df):
    top_models = df['model'].value_counts().nlargest(10).index
    filtered_df = df[df['model'].isin(top_models)]

    plt.figure(figsize=(12, 6), facecolor='white')  # Explicitly set background
    filtered_df.boxplot(column='price', by='model', grid=False)
    plt.title("Price Distribution by Model")
    plt.suptitle("")  # ✅ Suppress pandas' auto-title
    plt.xlabel("Model")
    plt.ylabel("Price (£)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)

def plot_model_counts(df):
    model_counts = df['model'].value_counts()

    plt.figure(figsize=(12, 6), facecolor='white')
    model_counts.plot(kind='bar', color='slateblue', edgecolor='black')

    plt.title("Number of Cars per Model", fontsize=14)
    plt.xlabel("Car Model", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()

def plot_year_vs_price(df):
    df_clean = df[['year', 'price']].dropna()

    plt.figure(figsize=(10, 6), facecolor='white')
    plt.scatter(df_clean['year'], df_clean['price'], alpha=0.5, color='dodgerblue', edgecolors='black')
    plt.title("Car Year vs Price")
    plt.xlabel("Year")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_year_vs_mileage(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['year'], df['mileage'], alpha=0.5, color='mediumseagreen')
    plt.title("Car Model Year vs Mileage")
    plt.xlabel("Year")
    plt.ylabel("Mileage")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_avg_mpg_by_fuel(df):
    plt.figure(figsize=(10, 6), facecolor='white')

    avg_mpg = df.groupby('fuelType')['mpg'].mean().sort_values(ascending=False)

    avg_mpg.plot(kind='bar', color='cornflowerblue', edgecolor='black')
    plt.title("Average MPG per Fuel Type")
    plt.xlabel("Fuel Type")
    plt.ylabel("Average MPG")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()

def plot_transmission_counts(df):
    plt.figure(figsize=(8, 5), facecolor='white')

    # Count how many cars are in each transmission category
    transmission_counts = df['transmission'].value_counts()

    transmission_counts.plot(kind='bar', color='steelblue', edgecolor='black')

    plt.title("Count of Cars by Transmission Type")
    plt.xlabel("Transmission")
    plt.ylabel("Car Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()

def plot_engine_size_vs_price(df):
    df_clean = df[['engineSize', 'price']].dropna()

    plt.figure(figsize=(10, 6), facecolor='white')
    plt.scatter(df_clean['engineSize'], df_clean['price'], alpha=0.6, color='darkorange', edgecolors='black')
    plt.title("Engine Size vs Price", fontsize=14)
    plt.xlabel("Engine Size (Litres)", fontsize=12)
    plt.ylabel("Price (£)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_tax_vs_engine_size(df):
    df_clean = df[['tax', 'engineSize']].dropna()

    plt.figure(figsize=(10, 6), facecolor='white')
    plt.scatter(df_clean['engineSize'], df_clean['tax'], alpha=0.6, color='orangered', edgecolors='black')

    plt.title("Sales Tax vs Engine Size", fontsize=14)
    plt.xlabel("Engine Size (Litres)", fontsize=12)
    plt.ylabel("Sales Tax ($)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_tax_vs_year(df):
    df_clean = df[['year', 'tax']].dropna()

    plt.figure(figsize=(10, 6), facecolor='white')
    plt.scatter(df_clean['year'], df_clean['tax'], alpha=0.6, color='mediumorchid', edgecolors='black')

    plt.title("Sales Tax vs Car Model Year", fontsize=14)
    plt.xlabel("Model Year", fontsize=12)
    plt.ylabel("Sales Tax ($)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_price_distribution(df)
plot_year_vs_price(df)
plot_year_vs_mileage(df)
plot_price_vs_mileage(df)
plot_price_by_model(df)
plot_model_counts(df)
plot_avg_mpg_by_fuel(df)
plot_transmission_counts(df)
plot_engine_size_vs_price(df)
plot_tax_vs_engine_size(df)
plot_tax_vs_year(df)