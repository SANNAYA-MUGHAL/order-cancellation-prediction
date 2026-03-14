# 🛒 Order Cancellation Analysis & Prediction

A machine learning project to analyze and predict order cancellations in e-commerce/grocery platforms.

**Author:** Sana Liaqat | [SANNAYA-MUGHAL](https://github.com/SANNAYA-MUGHAL)

---

## 📌 Project Overview

Order cancellations are a major operational challenge in e-commerce platforms. This project uses machine learning to:
- Analyze key drivers of order cancellations
- Build a predictive model to identify high-risk orders
- Provide actionable business recommendations

This project is inspired by real-world operational experience managing a large-scale retail technology platform, where reducing incident and cancellation rates was a core responsibility.

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Records** | 5,000 orders |
| **Features** | 10 (delivery time, payment method, order value, etc.) |
| **Target** | Binary — Cancelled (1) / Not Cancelled (0) |
| **Cancellation Rate** | ~18.8% |
| **Split** | 70% Train / 15% Validation / 15% Test |
| **Source** | Synthetic dataset based on real e-commerce patterns |

### Features Used:
- `delivery_time_days` — Estimated delivery duration
- `payment_method` — Credit card, Cash on delivery, Digital wallet
- `order_value` — Total order amount (AED)
- `num_items` — Number of items in order
- `customer_previous_cancellations` — Historical cancellation count
- `hour_of_order` — Time of day order was placed
- `day_of_week` — Day order was placed
- `customer_rating` — Customer satisfaction rating (1-5)
- `promo_applied` — Whether a promotional code was used
- `area_type` — Urban / Suburban / Rural

---

## 🤖 Models

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **Logistic Regression** | 88.67% | 0.932 |
| **Decision Tree** | 97.19% | 0.995 |

**Primary Model:** Logistic Regression — chosen for interpretability and generalizability.

**Loss Function:** Binary Cross-Entropy (Log Loss)  
**Regularization:** L2 (Ridge) with C=1.0  
**Training Command:**
```bash
python order_cancellation_analysis.py
```

---

## 📁 Project Structure

```
order-cancellation-prediction/
│
├── order_cancellation_analysis.py   # Main ML pipeline
├── orders_dataset.csv               # Generated dataset
├── eda_analysis.png                 # EDA visualizations
├── model_evaluation.png             # Confusion matrix + ROC curve
├── feature_importance.png           # Feature importance chart
└── README.md                        # Project documentation
```

---

## 🔧 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 📈 Key Findings

1. **Delivery Time** is the strongest predictor — 4+ day delivery = 2x cancellation risk
2. **Cash on Delivery** orders cancel 35% more than digital payments
3. **Repeat cancellers** (2+ history) are highest risk customers
4. **High-value orders** (>200 AED) need proactive monitoring
5. **Low-rated customers** (1-2 stars) cancel 40% more frequently

---

## 💡 Business Recommendations

- Prioritize fast delivery for high-value orders
- Incentivize digital payment methods
- Proactive SMS/push alerts for high-risk orders
- Dedicated support team for premium orders

---

## 🔬 Environment

```
Python 3.x
scikit-learn
numpy random seed: 42
```

## Results: Accuracy 88.67% | AUC 0.932
