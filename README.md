# Lab Journal: Data Normalization and Data Type Conversion

**Experiment No:** 14
**Date:** April 1, 2026
**Student Name:** jayvee 

---

## 🎯 Objective

To understand and implement various data normalization techniques and data type conversion methods using Python libraries such as Pandas, NumPy, and Scikit-learn.

---

## 📚 Theory

### Data Normalization

Data normalization is a preprocessing technique used to transform features to a common scale without distorting differences in the ranges of values. This is essential for machine learning algorithms that rely on distance calculations (e.g., KNN, SVM) or gradient descent optimization.

#### Types of Normalization:

1. **Min-Max Normalization**
   - Scales data to a fixed range [0, 1]
   - Formula: `X_normalized = (X - X_min) / (X_max - X_min)`

2. **Z-Score Normalization (Standardization)**
   - Transforms data to have mean = 0 and standard deviation = 1
   - Formula: `X_normalized = (X - μ) / σ`

3. **Decimal Scaling Normalization**
   - Scales data by dividing by a power of 10
   - Formula: `X_normalized = X / 10^d` (where d is the number of digits in max value)

### Data Type Conversion

Converting categorical data to numerical format is crucial for machine learning algorithms.

#### Encoding Techniques:

1. **Label Encoding**
   - Converts categorical labels to integers (0, 1, 2, ...)
   - Suitable for ordinal data

2. **One-Hot Encoding**
   - Creates binary columns for each category
   - Suitable for nominal data with no inherent order

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation and DataFrame operations |
| `numpy` | Numerical computations |
| `sklearn.preprocessing` | LabelEncoder for encoding categorical variables |

---

## 💻 Implementation

### Part A: Data Normalization

#### Sample Dataset

| product | price | units_sold | discount |
|---------|-------|------------|----------|
| laptop  | 1000  | 50         | 0.10     |
| mobile  | 500   | 200        | 0.20     |
| tablet  | 300   | 150        | 0.15     |
| desktop | 1500  | 30         | 0.05     |

#### 1. Min-Max Normalization

```python
# Single column normalization
df['price_min_max'] = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())

# Multiple column normalization
cols = ['price', 'units_sold', 'discount']
df_cols = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())
```

**Output:**

| product | price | price_min_max |
|---------|-------|---------------|
| laptop  | 1000  | 0.583333      |
| mobile  | 500   | 0.166667      |
| tablet  | 300   | 0.000000      |
| desktop | 1500  | 1.000000      |

#### 2. Z-Score Normalization

```python
df['price_z_score'] = (df['price'] - df['price'].mean()) / df['price'].std()
```

**Output:**

| product | price | price_z_score |
|---------|-------|----------------|
| laptop  | 1000  | 0.325435       |
| mobile  | 500   | -0.604379      |
| tablet  | 300   | -0.976304      |
| desktop | 1500  | 1.255248       |

#### 3. Decimal Scaling Normalization

```python
df['price_decimal_scaling'] = df['price'] / 100000
```

**Output:**

| product | price | price_decimal_scaling |
|---------|-------|------------------------|
| laptop  | 1000  | 0.010                  |
| mobile  | 500   | 0.005                  |
| tablet  | 300   | 0.003                  |
| desktop | 1500  | 0.015                  |

---

### Part B: Data Type Conversion

#### Sample Dataset

| order_id | customer_gender | payment_method | product_category | city |
|----------|-----------------|----------------|------------------|------|
| 101      | m               | upi            | electronics      | new york |
| 102      | f               | credit card    | fashion          | los angeles |

#### 1. Label Encoding

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['customer_gender_encoded'] = le.fit_transform(df['customer_gender'])
```

**Output:**

| order_id | customer_gender | customer_gender_encoded |
|----------|-----------------|-------------------------|
| 101      | m               | 1                       |
| 102      | f               | 0                       |

#### 2. One-Hot Encoding

```python
df_encoded = pd.get_dummies(df, columns=['payment_method'])
```

**Output:** Creates separate binary columns for each payment method:
- `payment_method_cash`
- `payment_method_credit card`
- `payment_method_debit card`
- `payment_method_upi`

---

### Part C: Amazon Products Dataset Analysis

Applied normalization and encoding techniques on a real-world Amazon products dataset containing:
- Product_ID, Product_Name, Price, Rating, Reviews, Units_Sold

---

## 📊 Results Analysis

### Normalization Comparison

| Technique | Range | Preserves Outliers | Use Case |
|-----------|-------|-------------------|----------|
| Min-Max | [0, 1] | Yes | When bounded range needed |
| Z-Score | (-∞, +∞) | No | When distribution matters |
| Decimal Scaling | (-1, 1) | Yes | When simple scaling needed |

### When to Use Each Technique

- **Min-Max Normalization:** Best for algorithms requiring bounded input (neural networks, deep learning)
- **Z-Score Normalization:** Best for algorithms assuming Gaussian distribution (linear regression, logistic regression)
- **Label Encoding:** Best for ordinal categorical data (low, medium, high)
- **One-Hot Encoding:** Best for nominal categorical data (colors, cities, payment methods)

---

## ✅ Conclusion

1. Successfully implemented three normalization techniques (Min-Max, Z-Score, Decimal Scaling)
2. Applied Label Encoding and One-Hot Encoding for categorical data conversion
3. Understood the importance of data preprocessing for machine learning pipelines
4. Normalization ensures fair contribution of all features to model training
5. Encoding techniques convert categorical data to machine-readable numerical format

---

## 🔗 References

1. [Scikit-learn Preprocessing Documentation](https://scikit-learn.org/stable/modules/preprocessing.html)
2. [Pandas Documentation](https://pandas.pydata.org/docs/)
3. [Data Normalization Techniques - Towards Data Science](https://towardsdatascience.com/)

---

## 📁 File Structure

```
├── data normalisation and data type conversion.py    # Main Python script
├── README.md                                         # Lab Journal (this file)
└── amazon_products_dataset_Expt-14.csv              # Dataset file
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone [repository-url]

# Navigate to directory
cd [repository-name]

# Run the script
python3 "data normalisation and data type conversion.py"
```


---
