**Problem Definition:<br>**
This Project aims to predict the price of a used ford vehicle. Ford is one of the top selling car manufacturers in the US with almost 2 million vehicles sold in 2024 and a US market share of 13%. The prediction will be made using characteristics provided by the users. This is aimed at those who know what they are looking for but don’t have a clue how much it’s truly worth. This lack of information allows the uninformed to fall victim to scams and traps which may land them in a financial disaster. It is estimated that over 100 million Americans have a car payment, imagine how many of them are overpaying on a car. This program allows users to get an understanding of the expected amount to pay for a used vehicle based on their description to help prevent overspending but furthermore it can help others looking to sell their used vehicle get a grasp of current market trends to prevent underpricing/consumer manipulation.

**Target Variable & Input Features:<br>**
The target is to predict the price. Seeing as cars are made up of many parts, there are various things that can be taken into consideration such as:
Safety Ratings, Car Name, Fuel Type, Aspiration, Number of Doors, Body Style (Coupes, Sedans, SUVs, Pickups), Wheel Drive FWD, RWD, AWD, Engine Location, Car Dimensions, Curb Weight, Engine Cylinders, Engine Size (Liters), Horsepower, Peak RPM, MPG, Car Price
For this project I chose to use Model name, Model year, Milage, Transmission, Fuel type, Sales Tax, MPG and Engine Size, as these are the most common considerations when buying a car. 

Provided by this data sheet which consisted of almost 20,000 rows of data.<br>
https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction<br>
*Please note that prices and predictions made by the models are based on this data sheet I obtained legally, hence trust these values in a vacuum. A more accurate data sheet = More real-world application.

> Data Cleaning process – 
My data cleaning process consisted focused on ensuring the datasheet doesn’t contain blank cells and whitespace after data entry. 
Checking for duplicates wouldn’t apply here as there are multiple car models, and other details that are duplicated in the sheet for price prediction.


**Innovative Feature Engineering:<br>**
Initially I was going to pick the most effective regression model, then I tried different ones when I got the idea to allow the user to simply select the ones they’d want to use. Like scientists running an experiment multiple times to gather consistency in the results, here I’m giving the user multiple predictions from various models to give them an ideal ballpark idea of how much they should expect to spend.



**Model Selection:<br>**
For this program I chose to use regression models are they are best suited for predicting numeric values and, in a supervised training simulation, would garner the best results. The models included are:<br>
🔹 1.	Linear Regression<br>
🔹 2.	Ridge Regression<br>
🔹 3.	Lasso Regression<br>
🔹 4.	Decision Tree Regressor<br>
🔹 5.	Random Forest Regressor<br>
🔹 6.	Gradient Boosting Regressor via XGBoost<br>
🔹 7.	LightGBM Regressor<br>



**Datasheet Graphical Representation and Insights & Justifying Feature Choices:<br>**
> I just wanted to make my own suite of graphical representation of the data in the dataset<br>

🔹 1. Price Distribution Histogram: Visualizes how car prices are spread across the dataset and helps spot price outliers and the most common price range.
We can see that most cars are in the $10,000 - $20,000 range.<br>
🔹 2. Car Year vs Price Scatter Plot: Illustrates how car value depreciates over time and clear positive correlation between newer year and higher price.
Newer used cars cost more money<br>
🔹 3. Year vs Mileage Scatter Plot: Displays how newer cars generally have lower mileage and confirms year and mileage are inversely related.
As we can see older cars have higher mileage.<br>
🔹 4. Price vs Mileage Scatter Plot: Shows the inverse relationship between how much a car has been driven and how much it's worth and supports mileage as a strong predictive feature.
Ideally cars with lower mileage cost more as they are seen as newer cars, and this theory remains true as seen on the plot.<br>
🔹 5. Car Model vs Count (Bar Chart): Displays the number of cars per model in the dataset and highlights the most common and least represented Ford models.
Fiestas are most common.<br>
🔹 6. Box Plot of Price by Model: Compares price distributions for the top 10 most common Ford models and reveals variability and pricing patterns across models.
We can see that Focuses, Kugas and S-Maxs cost the most money.<br>
🔹 7. Fuel Type vs MPG (Bar Chart): Compares the average fuel efficiency (MPG) across fuel types and shows that fuel type influences performance and cost-efficiency.
This graph displays fuel efficiency based on the different types of fuel. Measured in MPG we can see that Hybrid Cars attain the highest MPG.<br>
🔹 8. Transmission Type vs Model Count (Bar Chart): Counts how many vehicles have each transmission type (Manual, Auto, etc.) and useful for understanding data distribution for encoding.
This graph purely shows the number of cars and their respective transmissions, displayed here we can see that manual cars have occupied most of the data sheet.<br>
🔹 9. Engine Size vs Price Scatter Plot: Visualizes how car price scales with engine size and typically reveals higher engine size = higher price trend.
This graph shows how bigger engines affect the cars price as the 5L engines are in the $30,000 - $50,000 range, whereas the 1 – 2 Liter engines are common in the sub $30,000 range.<br>
🔹 10. Tax vs Engine Size Scatter Plot: Highlights any tax scaling with engine displacement and suggests how tax policies may influence vehicle design.
Here we can see a slight influence on sales tax based on the car’s engine size, we can directly corelate this influence to the fact that cars with bigger engines are more expensive as seen from the graph above.<br>
🔹 11. Tax vs Model Year Scatter Plot: Displays how road tax varies by car manufacturing year and may reflect government incentives or emissions rules.
From this graph we can see that the car model year doesn’t have that much impact on the amount of sales tax, assumingly this is because sales tax is based on a % of the car’s valuation and not when it was manufactured.



**UI & Deployment:<br>**
Initially opted for a CLI, then dove into the streamlit suite as they provide UI customisation and deployment for free :)

**Other Notes:<br>**
🔹 I wanted to implement live data based on current markets but from what I understand car market analysis reports are made quarterly so unless I may be able to get current data from dealerships and other outlets that report their sales on a more frequent basis     I dont think live data would be a possibility.<br>
🔹	The std on the datasheet statistic stands for standard deviation, which is a statistical measure of how spread out the values are in a dataset.<br>
🔹  I was using the transform function to train the model, but it kept breaking and then I found out about pipeline and used that instead, problem solved.<br>
🔹	Ridge and Lasso Convergence warning - This means the optimizer didn’t finish minimizing the loss before hitting the default iteration limit. It doesn't invalidate the results — but suggests the fit could be improved with more iterations.<br>

**References:<br>**
🚗Car price Prediction 📊 https://www.kaggle.com/code/zabihullah18/car-price-prediction/notebook <br>
Car Price Prediction https://www.kaggle.com/code/mohaiminul101/car-price-prediction/notebook<br>
🚙Used Car💸Price 🚗Prediction📈 https://www.kaggle.com/code/satyaprakashshukl/used-car-price-prediction<br>

> Coding Notes:<br>
cd C:\Users\kamro\PycharmProjects\DSCarPricePrediction
streamlit run ui.py


