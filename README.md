# Used Car Price Prediction

A machine learning project that predicts the selling price of used cars based on features like vehicle age, mileage, engine, max power, fuel type, transmission type, and kilometers driven.

## Features
- Data Cleaning and Preprocessing
- Feature Engineering
- Random Forest Regression Model
- Feature Importance Visualization
- Streamlit Web App

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

## Model Accuracy
- R2 Score: 0.94

## Project Structure

UsedCarProject/
│
├── data/
│   └── cardekho_dataset.csv
│
├── screenshots/
│   ├── streamlite.png
│   ├── predicted result.png
│   └── feature_importance.png
│
├── app.py
├── main.py
├── requirements.txt
├── README.md

## How to Run

1. Install required libraries

pip install -r requirements.txt

2. Run the Streamlit app

py -m streamlit run app.py
