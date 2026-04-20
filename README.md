# VentureGuard

VentureGuard is a Streamlit-based predictive analytics application designed to help founders and investors evaluate startup outcomes using real startup data.

## Live Demo

Production app: https://ventureguard.streamlit.app/

Best viewed on desktop. Upload `startup data.csv` to begin.

## What This App Does

- Cleans and validates uploaded startup CSV data
- Performs exploratory data analysis (EDA) with interactive visualizations
- Trains and compares classification models:
  - KNN
  - Decision Tree
  - Naive Bayes
  - SVM
- Reports holdout metrics and 5-fold cross-validation metrics (Accuracy, F1, ROC-AUC)
- Recommends the best model by CV ROC-AUC
- Runs regression analysis (Linear and Polynomial)
- Provides K-Means clustering analysis
- Supports interactive startup outcome prediction
- Allows export of classification metrics as CSV

## Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## Repository Structure

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version for Streamlit Cloud
- `assets/` - README screenshots
- `README.md` - Project documentation
- `.gitignore` - Git ignore rules

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the app:

```bash
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app and select this repository.
4. Set branch to `main`.
5. Set main file path to `app.py`.
6. Deploy.

## Screenshots

### Home Page

![Home](assets/home.png)

### Classification Page

![Classification](assets/classification.png)

## Dataset Requirement

The uploaded dataset must include a `status` column (with values such as `acquired` and `closed`) along with relevant startup feature columns used for modeling.

## Limitations

- Model outcomes are predictive and should not be treated as causal conclusions.
- Performance depends heavily on the quality and representativeness of the uploaded dataset.
- Results can vary across regions, sectors, and time periods not well represented in the data.

## Author

Built by Varad Miwale
