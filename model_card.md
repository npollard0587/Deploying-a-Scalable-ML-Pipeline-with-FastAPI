# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained on U.S. Census Bureau data. It predicts whether an individual's income is greater than $50,000 per year or less than or equal to $50,000 per year based on demographic and employment-related features.

## Intended Use

The model is intended as an example for demonstrating machine learning pipelines, data processing, model training and testing, and deployment with FastAPI. It should not be used in real-world decision making.

## Training Data

The model was trained using the provided census.csv dataset. The dataset includes information such as age, occupation, marital status, hours worked per week, and other demographic features.

## Evaluation Data

The dataset was split into training and testing sets using a train-test split. The test set was used to evaluate the model's performance after training.

## Metrics

The model was evaluated using Precision, Recall, and F1 score.

Precision: 0.7419
Recall: 0.6384
F1 Score: 0.6863

## Ethical Considerations

The dataset contains sensitive demographic information such as race, sex, and nationality. Because of this, the model could reflect biases that exist in the data. Predictions from this model should be interpreted carefully and should not be used to make major decisions. 

## Caveats and Recommendations

This model was built for educational purposes and has not been validated for production use. Additional testing, fairness analysis, and model evaluation would be needed before any real-world use.