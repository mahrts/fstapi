# Model Card: Classification model for census data salary range

## Model Details

This model is a **Random Forest classifier** built using **scikit-learn** to predict an individual's income range based on census data.

- **Model type:** Random Forest Classifier (ensemble learning)  
- **Framework:** scikit-learn  
- **Task:** Binary classification  
- **Target:** salary range (`<=50K` or `>50K`)  
- **Input:** Tabular features with both demographic and employment-related data 
- **Deployment:** REST API using FastAPI, hosted on Render  
- **Inference:** Real-time predictions via HTTP endpoints 

## Intended Use
### ✅ Appropriate Use Cases
- Predict salary range with live web app
- Showcase clean mlops for training and deploymeny phase: emphasis on FastAPI deployment
- Experimentation with sklearn classification model

## Training Data
The model was trained on the **Adult Census Income dataset** from the UCI Machine Learning Repository:  
https://archive.ics.uci.edu/dataset/20/census+income

### Dataset Characteristics
- Based on U.S. Census data  
- Contains demographic and employment attributes 

### Example Features
- Age  
- Workclass  
- Education  
- Marital status  
- Occupation  
- Relationship  
- Race  
- Sex  
- Hours per week  
- ...
### Target Variable
- Salary range (`<=50K`, `>50K`)  

### Preprocessing Steps
- Handling missing or unknown values  
- Encoding categorical variables (e.g., one-hot encoding)  
- Feature alignment between training and inference pipeline
- Split 20% of the data for evaluation 

---

## Metrics
The model was evaluated on the test set using standard classification metrics:
- **Precision:** 0.732 
- **Recall:** 0.631
- **F1-score:** 0.678
- **fbeta:** 0.678

### Notes
- Metrics are computed on a held-out test set  
- Metrics on data slices can be computed via the deployed app on render

## Ethical consideration

### Historical Bias

The data reflects historical societal patterns and may encode inequalities.

Risk of Misuse

Using this model in real-world decision-making could result in discriminatory outcomes.

### Automation Risk

Predictions may be very different with human judgment.

## Caveats and Recommendations

### Limitations

- The dataset might be outdated and may not reflect current economic conditions

### Recommendations
- Use cross-validation for more robust performance estimates
- Monitor API usage and inputs in deployment