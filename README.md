# Regression and Classification Using Scikit-learn

## Project overview
This project demonstrates two supervised machine-learning tasks using Python and Scikit-learn:

- **Regression:** Predict a continuous target using the built-in Diabetes dataset.
- **Classification:** Predict a diagnosis category using the built-in Wisconsin Diagnostic Breast Cancer dataset.

The project covers train/test splitting, model training, feature scaling where appropriate, and evaluation with task-specific metrics.

## Models included
### Regression
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors Regressor (k=5)

### Classification
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors Classifier (k=5)

## Tools
- Python 3.10+
- pandas
- scikit-learn

## How to run
1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the project:
   ```bash
   python main.py
   ```
5. The script prints evaluation metrics and creates:
   - `regression_results.csv`
   - `classification_results.csv`

The datasets are bundled with Scikit-learn, so no separate dataset download is required.

## Evaluation metrics
Regression models are compared using MAE, RMSE, and R². Classification models are compared using accuracy, precision, recall, F1-score, and ROC-AUC.

## Note
The results are an educational baseline on one train/test split. They are not a guarantee of performance on new data. The breast-cancer classification example is for learning only and is not intended for clinical use.
