# **Tehran House Price Prediction - Machine Learning Project**

## 🏙️ **Project Overview**
This project implements a **machine learning model** to predict housing prices in Tehran, Iran, based on property features. Using **polynomial regression**, the model analyzes area, number of rooms, and district location to estimate property values in USD.

## 🎯 **Key Features**
- **Data Processing Pipeline**: Cleans and preprocesses Tehran housing data with specialized district mapping
- **Polynomial Regression Model**: Implements degree-2 polynomial features for capturing non-linear relationships
- **Outlier Detection**: Uses Isolation Forest algorithm to remove anomalous property listings
- **Geographic Encoding**: Converts Tehran neighborhoods to numeric district values based on market value
- **Interactive Prediction**: Allows users to input property features and get instant price estimates

## 📊 **Dataset**
- **Source**: Tehran housing market listings
- **Features**: Area (m²), Number of Rooms, Parking, Warehouse, Elevator, Location, Price (USD)
- **Size**: ~2,900 property listings after cleaning
- **Special Processing**: Tehran neighborhoods mapped to 20 districts with value-based encoding

## 🔧 **Technical Stack**
- **Python** with Pandas, NumPy for data manipulation
- **Scikit-learn** for machine learning (Linear Regression, StandardScaler, IsolationForest)
- **Matplotlib** for data visualization
- **Jupyter Notebook** for development and analysis

## 🏗️ **Model Performance**
- **R² Score**: 0.8227 (82.27% variance explained)
- **Mean Absolute Error**: $41,827
- **No Overfitting**: Training and test scores are consistent
- **Key Predictors**: Area, District, and Number of Rooms show highest correlation with price

## 📈 **Key Insights**
1. **District is Crucial**: Location encoding significantly impacts prediction accuracy
2. **Non-linear Relationships**: Area-price relationship benefits from polynomial features
3. **Outlier Management**: Removing properties >500m² and statistical outliers improves model stability
4. **Feature Selection**: Only features with >0.2 correlation to price were retained

## 🚀 **How to Use**

### **1. Installation**
```bash
git clone https://github.com/yourusername/tehran-house-price-prediction.git
cd tehran-house-price-prediction
pip install -r requirements.txt
```

### **2. Run the Model**
```python
# Interactive prediction example
python predict.py
# Enter: 85.5 (area), 2 (rooms), 20 (district - highest value)
# Output: Estimated Price: $XXX,XXX.XX
```

### **3. Train Your Own Model**
```python
# Full pipeline from raw data to trained model
jupyter notebook House_Price_Prediction.ipynb
```

## 📁 **Project Structure**
```
├── House_Price_Prediction.ipynb     # Main Jupyter notebook
├── predict.py                       # Production prediction script
├── requirements.txt                 # Dependencies
├── house_price.csv                  # Raw dataset
├── scaler.pkl                       # Saved scaler (optional)
├── model.pkl                        # Saved model (optional)
└── README.md                        # This file
```

## 🔍 **Methodology**
1. **Data Cleaning**: Boolean encoding, missing value handling, district mapping
2. **Feature Engineering**: Polynomial expansion (degree=2), standardization
3. **Model Training**: Linear regression with polynomial features
4. **Evaluation**: R² score, MSE, MAE, overfitting check
5. **Deployment**: Interactive prediction interface

## 🎓 **What I Learned**
- **Real-world Data Challenges**: Handling Tehran-specific geographic encoding
- **Feature Engineering Importance**: Polynomial features for non-linear relationships
- **Model Evaluation**: Balancing complexity with generalization
- **Production Considerations**: Data scaling consistency between training and prediction

## 🔮 **Future Improvements**
- Incorporate more features (age of building, floor number, amenities)
- Implement different algorithms (Random Forest, Gradient Boosting)
- Create web interface for easier access
- Add time-series analysis for market trends
- Include confidence intervals in predictions

## 👥 **Contributing**
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 **License**
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**
- Tehran real estate data providers
- Scikit-learn documentation and community
- Open-source machine learning resources

---

**Note**: This model provides estimates based on historical data and should be used as a reference tool alongside professional real estate advice.
