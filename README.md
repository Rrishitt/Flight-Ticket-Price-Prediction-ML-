# ✈️ Flight Price Prediction — ML Project

> **Predicting flight ticket prices before they take off.**
> Turning messy airline data into 💰 smarter price predictions using Machine Learning.

---

## 🚀 Project Overview

Flight tickets are basically chaos with a price tag. 😭✈️

The same route can have wildly different prices depending on the **airline, number of stops, travel class, duration, departure timing, and—most importantly—how many days are left before the journey.**

This project builds a **Machine Learning regression system** to predict the price of a flight ticket based on these features.

🎯 **Goal:** Given flight details → predict the ticket price.

This is a classic **supervised learning + regression** problem with a mix of categorical and numerical features.

---

## 🧠 Problem Statement

Given information about a flight, can we predict its ticket price?

### 📥 Input

The model receives information such as:

* ✈️ Airline
* 🛫 Source city
* 🛬 Destination city
* 🕐 Departure time
* 🕘 Arrival time
* 🛑 Number of stops
* 💺 Seat class
* ⏱️ Flight duration
* 📅 Days left before departure
* 🔢 Flight code

### 📤 Output

💰 **Predicted Flight Ticket Price**

In other words:

```text
Flight Information
        ↓
Feature Engineering
        ↓
Machine Learning Model
        ↓
💰 Predicted Price
```

---

# 📂 Dataset

The project uses three CSV files:

| File                    | Description                                      |
| ----------------------- | ------------------------------------------------ |
| `train.csv`             | 🧠 Training dataset containing features + target |
| `test.csv`              | 🔮 Test dataset where the target price is hidden |
| `sample_submission.csv` | 📤 Submission format                             |

---

## 📊 Feature Dictionary

| Feature       | Type                     | Description                      |
| ------------- | ------------------------ | -------------------------------- |
| `airline`     | 🏷️ Categorical          | Airline company                  |
| `flight`      | 🔢 Categorical           | Flight code/information          |
| `source`      | 📍 Categorical           | City where the journey begins    |
| `departure`   | 🕐 Categorical           | Departure time period            |
| `stops`       | 🔢 Categorical/Numerical | Number of stops                  |
| `arrival`     | 🕘 Categorical           | Arrival time period              |
| `destination` | 📍 Categorical           | Destination city                 |
| `class`       | 💺 Categorical           | Seat/travel class                |
| `duration`    | ⏱️ Numerical             | Total journey duration in hours  |
| `days_left`   | 📅 Numerical             | Days between booking and journey |
| `price`       | 💰 Target                | Flight ticket price              |

---

# 🔍 Why This Problem Is Interesting

Flight pricing isn't just:

> `longer flight = more expensive` ❌

There are multiple interacting factors.

For example:

```text
Airline ───────┐
               │
Stops ─────────┤
               │
Class ─────────┤
               ├──→ 💰 Ticket Price
Duration ──────┤
               │
Days Left ─────┤
               │
Route ─────────┘
```

One of the most interesting variables is **`days_left`**.

A flight booked months in advance can have a completely different price from the same flight booked a few days before departure.

📈 That's where the ML part gets spicy.

---

# 🛠️ Tech Stack

### 🐍 Programming

* Python

### 📦 Data Science

* Pandas
* NumPy

### 📊 Visualization

* Matplotlib
* Seaborn

### 🤖 Machine Learning

* Scikit-learn
* Regression Models
* Feature Engineering
* Cross-Validation
* Hyperparameter Tuning

### 🧪 Evaluation

* RMSE
* MAE
* R² Score

---

# ⚙️ Machine Learning Pipeline

The overall workflow:

```text
📂 Raw Dataset
      ↓
🧹 Data Cleaning
      ↓
🔎 Exploratory Data Analysis
      ↓
🛠️ Feature Engineering
      ↓
🔤 Categorical Encoding
      ↓
📏 Numerical Feature Processing
      ↓
✂️ Train / Validation Split
      ↓
🤖 Model Training
      ↓
🧪 Cross Validation
      ↓
🎯 Hyperparameter Optimization
      ↓
📈 Model Evaluation
      ↓
🔮 Test Prediction
      ↓
📤 Submission
```

---

# 🔎 Exploratory Data Analysis

Before throwing the dataset into a model, the goal is to understand:

### 💰 Price Distribution

How are ticket prices distributed?

```text
Cheap ─────────────────────────────── Expensive
  🟢        🟡          🟠              🔴
```

The dataset contains flight prices spanning from roughly:

**₹0 → ₹10,000+**

with substantial variation across different flight characteristics.

---

## 📅 Days Left vs Price

One of the key questions:

> Does booking earlier actually mean paying less?

This relationship can be explored through:

```text
Days Left
   ↓
Booking Timing
   ↓
Ticket Price
```

This feature is particularly interesting because airline pricing can change dynamically as departure approaches.

---

# 🧩 Feature Engineering

Raw columns aren't always model-ready.

Some features require transformation before they can become useful signals.

### 🏷️ Categorical Features

Examples:

```text
airline
source
destination
departure
arrival
class
```

These need to be converted into numerical representations.

### ⏱️ Duration

Duration can be transformed into a consistent numerical representation.

For example:

```text
2h 30m
   ↓
2.5 hours
```

### ✈️ Stops

Stop information can similarly be transformed into a numerical/ordinal representation.

---

# 🤖 Model Development

Multiple regression algorithms can be evaluated to determine which modelling approach captures the pricing patterns most effectively.

Potential candidates include:

### 📉 Linear Models

Useful as a baseline.

```text
Linear Regression
```

### 🌳 Tree-Based Models

Better suited for nonlinear relationships.

```text
Decision Tree
Random Forest
Gradient Boosting
```

### ⚡ Advanced Boosting

For complex tabular datasets:

```text
XGBoost
LightGBM
CatBoost
```

The final model can then be selected based on validation performance rather than blindly assuming that the most complicated model is automatically the best one.

> **More complexity ≠ automatically more accuracy.** 🧠

---

# 📏 Model Evaluation

Since the target is a continuous numerical value, regression metrics are used.

### 📉 RMSE

Root Mean Squared Error:

```text
RMSE = √(mean((Actual - Predicted)²))
```

Lower = better. 📉

---

### 📊 MAE

Mean Absolute Error:

```text
MAE = mean(|Actual - Predicted|)
```

This provides an intuitive measure of the average prediction error.

---

### 🎯 R² Score

Measures how much variation in ticket prices is explained by the model.

```text
R² → closer to 1 = better explanatory power
```

---

# 🧪 Cross-Validation

Instead of trusting a single train-validation split, **K-Fold Cross-Validation** can be used.

```text
Dataset
   │
   ├── Fold 1 → Validation
   ├── Fold 2 → Validation
   ├── Fold 3 → Validation
   ├── Fold 4 → Validation
   └── Fold 5 → Validation
```

This provides a more robust estimate of how the model performs on unseen data.

---

# 🏆 Results

The project evaluates models using validation performance and selects the approach that provides the strongest generalization.

### 📌 Key takeaway

Flight pricing is a **nonlinear prediction problem** where categorical information and booking timing can interact in complicated ways.

The project demonstrates how:

> **Raw airline data → engineered features → ML model → price prediction**

can turn a messy tabular dataset into a practical predictive system. 🚀

---

# 📤 Submission

Predictions are generated for the test dataset and formatted according to:

```text
sample_submission.csv
```

Expected structure:

|  id |           price |
| --: | --------------: |
|   0 | Predicted Price |
|   1 | Predicted Price |
|   2 | Predicted Price |
| ... |             ... |

---

# 💡 Key Learnings

### 1️⃣ Feature Engineering Matters

Raw data rarely comes perfectly packaged for ML.

Good preprocessing can significantly improve model performance.

### 2️⃣ Tabular Data ≠ Easy Data

Just because the dataset is a CSV doesn't mean the problem is simple.

Categorical variables + nonlinear interactions + pricing dynamics = 👀

### 3️⃣ Booking Timing Is Important

`days_left` provides a potentially powerful signal for understanding ticket pricing behaviour.

### 4️⃣ Validation > Vibes

A model shouldn't be selected because:

> "XGBoost is popular bro." 💀

It should be selected based on measurable validation performance.

### 5️⃣ Model Complexity Needs Justification

A sophisticated model is useful only when it actually improves generalization.

---

# 🧠 Business Perspective

This project isn't just about predicting a number.

Accurate flight-price prediction can potentially support:

### 👤 Consumers

* Decide **when to book**
* Identify potentially expensive periods
* Compare pricing patterns

### ✈️ Airlines

* Understand pricing behaviour
* Analyse demand patterns
* Improve revenue-management strategies

### 🧳 Travel Platforms

* Build price prediction tools
* Recommend booking windows
* Personalize travel recommendations

---

# 📌 Project Structure

```text
flight-price-prediction/
│
├── 📂 data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── 📓 notebooks/
│   └── flight_price_prediction.ipynb
│
├── 📂 src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── model.py
│
├── 📂 outputs/
│   └── submission.csv
│
├── 📄 README.md
└── 📄 requirements.txt
```

---

# 🚀 Future Improvements

There is plenty of room to push this project further.

### 🔥 Next-Level Ideas

* [ ] Hyperparameter optimization with Optuna
* [ ] Advanced CatBoost / LightGBM tuning
* [ ] SHAP-based model explainability
* [ ] Feature importance analysis
* [ ] Price prediction dashboard
* [ ] Interactive Streamlit application
* [ ] Booking-time recommendations
* [ ] Automated model retraining
* [ ] Experiment tracking with MLflow
* [ ] Ensemble modelling

The endgame:

```text
📊 Dataset
   ↓
🤖 ML Model
   ↓
💰 Price Prediction
   ↓
📈 Price Trend Analysis
   ↓
🧠 "Should I book now?"
   ↓
🔥 Intelligent Flight Booking Assistant
```

---

# 🎯 Project Objective

The ultimate objective is to demonstrate an end-to-end Machine Learning workflow for **flight ticket price prediction**, covering:

> **Data → EDA → Feature Engineering → Modelling → Validation → Prediction → Deployment**

Because predicting prices is cool.

But understanding **why prices move** is where things get interesting. 🧠✈️💸

---

## ⭐ If You Found This Interesting

Feel free to explore the notebook, experiment with different models, and improve the pipeline.

**Star ⭐ the repository if you found it useful!**

```text
Built with 🐍 Python + 📊 Data Science + 🤖 Machine Learning
```

### ✈️ Predict smarter.

### 💰 Book smarter.

### 🧠 Build smarter.
