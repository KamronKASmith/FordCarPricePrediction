import matplotlib.pyplot as plt
import pandas as pd

def plot_price_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['Selling_Price'], bins=50, color='skyblue', edgecolor='black')
    plt.title("Car Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

