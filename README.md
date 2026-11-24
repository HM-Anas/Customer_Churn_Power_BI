# Customer Churn Prediction & Power BI Dashboard

An end-to-end banking customer churn analysis project using **Power BI, DAX, Python, and Machine Learning**.

## Dataset
- 10,000 customer records
- 12 columns
- 2,037 churned customers
- Overall churn rate: 20.37%

## Dashboard
Five interactive Power BI pages:
1. Customer Churn Overview
2. Customer Demographics
3. Churn Analysis
4. Advanced Analytics
5. Geographic Distribution & Churn Prediction

## Machine Learning
Random Forest Classifier evaluated using an 80:20 stratified train-test split.

| Metric | Score |
|---|---:|
| Accuracy | 86.65% |
| Precision | 79.17% |
| Recall | 46.68% |
| F1-score | 58.73% |
| ROC-AUC | 85.79% |

## Repository Files
- `customer_churn_data.csv` — Dataset
- `customer_churn_powerbi_simple_rf.py` — Python classification script
- `customer_churn_predictions_powerbi.csv` — Customer-level predictions

## Technologies
Python, Pandas, Scikit-learn, Power BI, DAX.

## Run the Python Script

```bash
pip install pandas scikit-learn
python customer_churn_powerbi_simple_rf.py
```

## Author
**Anas Jamshed**

GitHub: [HM-Anas](https://github.com/HM-Anas)
