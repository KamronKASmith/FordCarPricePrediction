import pandas as pd
from model_training import train_model

#Introduction
def intro():
    print("~" * 41)
    print("=" * 41)
    print("🚗Introducing the Ford Price Predictor🚗")
    print("=" * 41)
    print("~" * 41)
    print("This program used up to 7 Machine Learning Models to accurately estimate the market value of a wsed Ford vehicle.\n")
    input("Press ENTER to begin entering car details...\n")

#Functions that list unique values from the datasheet to gather testing parameters
def fleet(df):
    print("🛒 Ford Models Available:")

    #Cleans whitespaces from the datasheet causing duplicates
    cleaned_models = df['model'].str.strip().str.title()

    unique_models = sorted(cleaned_models.drop_duplicates())
    for i, model in enumerate(unique_models, start=1):
        print(f"{i:>2}. {model}")

def taxbracket(df):
    print("💸 Available Road Tax Values in Dataset:\n")
    unique_taxes = sorted(df['tax'].dropna().unique())
    for i, tax in enumerate(unique_taxes, start=1):
        print(f"{i:>2}. ${tax}")

def MPG(df):
    print("⛽ Available MPG Values in Dataset:\n")
    unique_mpg = sorted(df['mpg'].dropna().unique())
    for i, mpg in enumerate(unique_mpg, start=1):
        print(f"{i:>2}. {mpg:.1f} MPG")
    print(f"\n🔢 Total unique MPG values: {len(unique_mpg)}\n")

def list_unique_engine_sizes(df):
    print("🔧 Available Engine Sizes in Dataset:\n")
    unique_engines = sorted(df['engineSize'].dropna().unique())
    for i, size in enumerate(unique_engines, start=1):
        print(f"{i:>2}. {size:.1f} L")
    print(f"\n🔢 Total unique engine sizes: {len(unique_engines)}\n")

#Collect User Input
def get_user_input():
    model = input("What model are you looking for? ").strip()
    year = int(input("From 1996 to 2020, which year are you looking for? ").strip())
    transmission = input("Automatic, Semi-Auto, or Manual Transmission? ").strip()
    mileage = int(input("How many miles has it done? ").strip())
    fuelType = input("Petrol, Diesel, Hybrid or Electric? ").strip()
    tax = int(input("Now pick a tax amount from 0 - 580: ").strip())
    mpg = float(input("Between 20 and 200, how many Miles per Gallon does it get? ").strip())
    engineSize = float(input("From 1.0 to a 5.0, what sized engine does it have? ").strip())

    print("\nOkay, Let's see what price you're looking at...\n")
    return pd.DataFrame([{
        'model': model,
        'year': year,
        'transmission': transmission,
        'mileage': mileage,
        'fuelType': fuelType,
        'tax': tax,
        'mpg': mpg,
        'engineSize': engineSize
    }])



model_list = {
    "linear": "Linear Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "tree": "Decision Tree Regressor",
    "random": "Random Forest Regressor",
    "boost": "Gradient Boosting Regressor (XGBoost)",
    "light": "LightGBM Regressor"
}

model_options = {
    "1": "linear",
    "2": "ridge",
    "3": "lasso",
    "4": "tree",
    "5": "random",
    "6": "boost",
    "7": "light",
    "8": "all",
    "0": "exit"
}


def select_models():
    print("📊 Select model(s) for prediction:")
    for key, code in model_options.items():
        if code == "all":
            print(f"{key}. All models")
        elif code == "exit":
            print(f"{key}. Exit")
        else:
            print(f"{key}. {model_list[code]}")

    selection = input("\nEnter numbers separated by commas (e.g. 1,3,5), or '8' for all, '0' to exit: ").strip().lower()

    if "0" in selection:
        print("\n👋 Exiting Ford Price Predictor. Goodbye!\n")
        exit()

    if "8" in selection or "all" in selection:
        return list(model_list.keys())  # all real model types

    # Parse multi-select
    entries = selection.split(',')
    selected = [model_options.get(num.strip()) for num in entries if
                num.strip() in model_options and model_options[num.strip()] not in ("all", "exit")]

    if not selected:
        print("\n❗ Invalid selection, please try again.\n")
        return select_models()

    return selected


def main():
    #Loads the data sheet for all functions that need them
    df = pd.read_csv('Sources/FordPrices.csv')

    intro()
    fleet(df)

    #Stores the users input into a data frame
    sample = get_user_input()

    print("\n✅The model will be scored using the following criteria:\n"
          "1. R² Score:\n The statistical evaluation of how a regression model fits the data. The closer to 1, the better.\n"
          "\n2. Mean of Absolute Errors (MAE):\n Measurement of the average magnitude of errors, The closer to 0, the better.\n"
          "\n3. Root of the Mean of Square Errors (RMSE):\n The square root of the expectation of the squared difference between"
          " estimated and actual values. The closer to 0, the better.\n")

    print("🔍 Select one or more models to use for prediction:")

    selected_models = select_models()

    print("\n⚙️ Training models and generating predictions...")

    for model_type in selected_models:
        model = train_model(df, model_type)
        prediction = model.predict(sample)[0]
        print(f"\n💰 Predicted Price using {model_list[model_type]} model: ${prediction:,.2f}")

    print("\n🎯 Thank you for using the Ford Price Predictor!\n")

if __name__ == '__main__':
    main()
