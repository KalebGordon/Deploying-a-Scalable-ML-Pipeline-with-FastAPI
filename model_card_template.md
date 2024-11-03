# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Developer:** Kaleb Gordon
- **Date:** 11/03/2024
- **Version:** Python 3.12, Scikit-learn
- **Type:** RandomForestClassifier
- **Goal:** To classify individuals based on census data, predicting if an individual earns over a certain income threshold.

## Intended Use
This model is intended to support research on income classification using demographic census data. This model is only being utilized for educational purposes.

## Training Data
This model was trained on a dataset of census data, which includes demographic data such as:
- **Workclass**
- **Education**
- **Marital Status**
- **Occupation**
- **Relationship**
- **Race**
- **Sex**

This data was split into training and testing sets with an 80-20 split.

## Evaluation Data
The evaluation was performed on a held-out test set derived from the original dataset, comprising 20% of the data. Performance was also evaluated on subsets of the data, broken up by categorical features.

## Metrics
This model was evaluated using:

- **Precision**
- **Recall**
- **F1 Score**

The following metrics were reported on various slices of categorical features:

| Feature         | Value                | Precision | Recall | F1 Score |
|-----------------|----------------------|-----------|--------|----------|
| workclass       | Federal-gov          | 0.7879    | 0.7429 | 0.7647   |
| workclass       | Private              | 0.7373    | 0.6394 | 0.6849   |
| education       | Bachelors            | 0.7431    | 0.7200 | 0.7314   |
| education       | Masters              | 0.8279    | 0.8599 | 0.8436   |
| marital-status  | Married-civ-spouse   | 0.7334    | 0.6877 | 0.7098   |
| race            | White                | 0.7390    | 0.6366 | 0.6840   |
| sex             | Female               | 0.7176    | 0.5236 | 0.6055   |

Please see `slice_output.txt` for a full breakdown of the metrics. 

The overall metrics were:

 Precision | Recall | F1 Score |
|-----------|--------|----------|
| 0.7516    | 0.6327 | 0.6869   |

## Ethical Considerations
- **Bias in Data**: The model is trained on census data, which may reflect societal biases.
- **Sensitive Attributes**: Features like race and gender could introduce bias in predictions.
- **Transparency**: The model is only used for educational purposes.

## Caveats and Recommendations
- **Limitations**: Performance varies across slices.
- **Further Validation**: Additional fairness checks and bias mitigation are recommended before deployment.
- **Production Use**: Testing and monitoring is essential before deployment.
