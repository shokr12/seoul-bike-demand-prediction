# 🚴 Seoul Bike Sharing Demand Analysis

A machine learning project designed to predict hourly bike rental demand in Seoul using weather, temporal, and seasonal features. This repository showcases end-to-end data preprocessing, chronological train-test splitting, regression modeling using Random Forest, and model interpretability.

---

## 🎯 Project Purpose

In urban environments, maintaining a stable supply of rental bikes is crucial for public convenience. Predicting demand in advance helps fleet operators optimize bike distribution. This project builds a predictive model to estimate **Rented Bike Count** based on temporal factors (time of day, day of the week, holidays) and weather conditions (temperature, humidity, solar radiation, rain, and snow).

---

## 🛠️ Tech Stack & Tools

The project is built entirely in Python using standard data science and machine learning libraries:

*   **Language:** Python 3.8+
*   **Data Manipulation:** `pandas`, `numpy` (time-series sorting, custom mapping, datetime extraction)
*   **Machine Learning:** `scikit-learn` (`RandomForestRegressor` and regression evaluation metrics)
*   **Data Visualization:** `matplotlib` (feature importance plot)

---

## 🧠 Key Skills & Concepts Demonstrated

This codebase demonstrates key practical skills required for real-world machine learning engineering:

### 1. Feature Engineering & Datetime Parsing
Instead of feeding raw columns directly to the model, custom domain-specific features are engineered to capture human patterns:
*   **Temporal Splits:** Extracting `DayOfWeek`, `Month`, and identifying `IsWeekend`.
*   **Human Activity Indicators:** Creating binary markers for `IsRushHour` (7-9 AM, 5-7 PM) and `IsNight` (10 PM - 5 AM).
*   **Categorical Encoding:** Mapping categorical columns like `Seasons`, `Holiday`, and `Functioning Day` to binary or ordinal values.

### 2. Time-Based Data Splitting
For time-series or sequential data, random train-test splitting leaks future data into the past.
*   **Sorting by Date:** The dataset is ordered chronologically by `Date`.
*   **Chronological Split:** The first 80% is used for training, and the final 20% is reserved for testing, ensuring realistic evaluation of future demand predictions.

### 3. Ensemble Modeling & Optimization
*   **Random Forest Regressor:** A robust ensemble method is trained utilizing parallel jobs (`n_jobs=-1`) for fast execution.
*   **Model Evaluation:** Performance is measured using standard regression metrics:
    *   **$R^2$ Score** (Coefficient of Determination)
    *   **RMSE** (Root Mean Squared Error)
    *   **MAE** (Mean Absolute Error)

### 4. Feature Importance & Model Interpretability
*   Extracting the internal feature importances from the Random Forest model to understand what drives bike sharing demand (e.g., Temperature, Hour, Functioning Day) and visualizing them in a horizontal bar chart.

---

## 📁 Repository Structure

*   [`main.py`](file:///c:/Users/mshok/Desktop/ML%20Project/main.py) — The main Python entrypoint containing the preprocessing, training, evaluation, and plotting logic.
*   `SeoulBikeData.csv` — The raw dataset containing hourly weather and bike sharing counts in Seoul.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8 or newer installed. Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Usage

Run the analysis script from the project root:

```bash
python main.py
```

### What happens when you run `main.py`?
1.  Loads the dataset and performs feature engineering.
2.  Splits the dataset chronologically (80% Train, 20% Test).
3.  Trains the Random Forest model.
4.  Prints performance metrics ($R^2$, RMSE, MAE) to the console.
5.  Generates and displays a horizontal bar chart of feature importances.

