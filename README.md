# 🏠 Tehran House Price Prediction

A machine learning project for predicting residential property prices in Tehran using property characteristics and location information. The project covers data cleaning, feature engineering, outlier detection, feature normalization, polynomial feature transformation, and regression-based price prediction.

---

## 📌 Overview

This project processes a Tehran housing dataset and develops a regression model to estimate property prices based on:

* **Area** (m²)
* **Number of rooms**
* **District**

The original dataset contains property information such as parking, warehouse, elevator, address, and price. The address information is transformed into Tehran district information before being used for modeling.

The final model uses **Polynomial Regression with degree 2**, implemented through `PolynomialFeatures` and `LinearRegression`.

---

## 📊 Dataset

The project uses a `house_price.csv` dataset containing Tehran residential property listings.

The original dataset includes:

| Feature      | Description                          |
| ------------ | ------------------------------------ |
| `Area`       | Property area in square meters       |
| `Room`       | Number of rooms                      |
| `Parking`    | Whether the property has parking     |
| `Warehouse`  | Whether the property has a warehouse |
| `Elevator`   | Whether the property has an elevator |
| `Address`    | Property neighborhood                |
| `Price`      | Property price                       |
| `Price(USD)` | Property price in USD                |

The dataset is processed to transform neighborhood information into a numerical **District** feature.

---

## 🔄 Project Workflow

```text
Raw Housing Data
       ↓
Data Type Inspection
       ↓
Neighborhood → District Mapping
       ↓
Boolean Feature Conversion
       ↓
Feature Selection
       ↓
Missing Value Handling
       ↓
Area Filtering
       ↓
Outlier Detection
       ↓
Feature Normalization
       ↓
Train / Test Split
       ↓
Polynomial Feature Engineering
       ↓
Linear Regression
       ↓
Model Evaluation
       ↓
House Price Prediction
```

---

## 🧹 Data Cleaning & Preprocessing

Several preprocessing steps are applied before model training.

### 1. Neighborhood Mapping

The original `Address` feature contains neighborhood names. These neighborhoods are mapped to their corresponding Tehran districts.

For example:

```text
Shahran → Region 5
Shahrake Gharb → Region 2
Velenjak → Region 1
Narmak → Region 8
```

Properties categorized as **Outskirts** are excluded from the modeling dataset.

The districts are then converted into numerical values based on their approximate relative price level.

### 2. Boolean Conversion

Boolean features such as `Parking`, `Warehouse`, and `Elevator` are converted from `True`/`False` into `1`/`0`.

### 3. Feature Selection

Correlation analysis is used to identify features related to `Price(USD)`.

The final model uses:

* `Area`
* `Room`
* `District`

### 4. Area Filtering

Properties with an area greater than **500 m²** are removed to reduce the influence of unusually large properties.

### 5. Outlier Detection

An **Isolation Forest** is used to identify potential outliers.

```python
IsolationForest(
    contamination=0.05,
    random_state=42
)
```

After preprocessing and outlier filtering, the modeling dataset contains **2,573 observations**.

---

## ⚙️ Feature Normalization

The selected input features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()
x = scaler.fit_transform(x)
```

This places the features on comparable scales before polynomial feature generation and model training.

---

## 🧠 Model Development

### Polynomial Feature Engineering

To capture potential non-linear relationships between property characteristics and price, second-degree polynomial features are generated.

```python
poly = PolynomialFeatures(degree=2)

x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
```

### Regression Model

A `LinearRegression` model is then trained on the generated polynomial features.

```python
regr = linear_model.LinearRegression()
regr.fit(x_train_poly, y_train)
```

Therefore, although the final estimator is `LinearRegression`, the overall model represents a **second-degree polynomial regression**.

---

## 🧪 Train / Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

A fixed random state of `42` is used to make the split reproducible.

```python
train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42
)
```

---

## 📈 Model Performance

The model is evaluated on the test set using R², Mean Squared Error (MSE), and Mean Absolute Error (MAE).

| Metric       |               Result |
| ------------ | -------------------: |
| **R² Score** |           **0.8227** |
| **MSE**      | **4,763,312,235.42** |
| **MAE**      |       **$41,826.72** |

The test R² of **0.8227** means that the model explains approximately **82.27% of the variance** in the test-set house prices.

The model's training R² is **0.8243**, which is close to the test R².

---

## 🔍 Overfitting Check

The training and testing R² values are:

```text
Training R²: 0.8243
Testing R²:  0.8227
```

The relatively small difference between the two values suggests that the model does not show a large train/test performance gap in this experiment.

---

## 🏠 Example Prediction

The notebook includes an interactive command-line prediction example.

The user provides:

```text
Area: 80 m²
Rooms: 2
District: 1
```

The trained preprocessing pipeline is then applied to the input:

```text
Input
  ↓
StandardScaler
  ↓
PolynomialFeatures
  ↓
Trained Regression Model
  ↓
Predicted Price
```

Example output:

```text
Area: 80.0m²
Rooms: 2
District: 1

Estimated Price: $115,235.40
```

---

## 🖼️ Visualizations

The notebook includes visual analysis of the housing data, including the relationship between **property area and price** and other exploratory analysis used during feature selection and preprocessing.

---

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

Main machine learning techniques:

* Feature engineering
* Correlation analysis
* Standardization
* Isolation Forest
* Polynomial Features
* Linear Regression
* R²
* MSE
* MAE

---

## 📁 Project Structure

```text
Tehran-House-Price-Prediction/
│
├── house_price_prediction.ipynb
├── house_price.csv
├── README.md
├── requirements.txt

```

---

## 🚀 Installation & Usage

### Prerequisites

Install Python and the required dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Clone the Repository

```bash
git clone https://github.com/nowherewalrus/Tehran-House-Price-Prediction.git
cd Tehran-House-Price-Prediction
```

### Run the Notebook

```bash
jupyter notebook House_Price_Prediction.ipynb
```

Make sure `house_price.csv` is located in the project directory.

---

## ⚠️ Limitations

This project is an educational machine learning project and should not be considered a professional real-estate valuation system.

Several aspects of the preprocessing could be improved in future versions:

* The district feature is represented numerically according to an approximate price ordering.
* The model uses only three final input features.
* The dataset represents a specific collection of Tehran property listings.
* A single train/test split is used rather than cross-validation.
* The model does not include potentially important features such as property age, floor number, building age, or more detailed location information.

The predicted prices should therefore be interpreted as **model estimates rather than actual market valuations**.

---

## 🎯 Future Improvements

Potential improvements include:

* Use **one-hot encoding** or other appropriate categorical encoding for location.
* Add additional property characteristics such as building age, floor, and total floors.
* Compare polynomial regression with models such as Random Forest and Gradient Boosting.
* Apply **cross-validation** for more robust evaluation.
* Perform systematic hyperparameter tuning.
* Analyze feature importance and model coefficients.
* Build a web-based prediction interface using **Streamlit** or **Flask**.
* Save the trained preprocessing pipeline and model for deployment.

---

## 📚 Project Focus

This project demonstrates an end-to-end classical machine learning workflow:

**Data → Cleaning → Feature Engineering → Outlier Detection → Normalization → Polynomial Modeling → Evaluation → Prediction**
