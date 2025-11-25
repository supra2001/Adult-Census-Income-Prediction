# Income Prediction Pipeline using Adult Census Dataset

A complete end-to-end Machine Learning Pipeline designed to predict whether an individual earns > $50K or ≤ $50K annually based on demographic and socio-economic attributes from the UCI Adult Census Dataset.

- This project demonstrates:

- Modular ML project architecture

- Data Ingestion → Transformation → Model Training → Prediction Pipeline

- Outlier handling, preprocessing, feature scaling

- Hyperparameter tuning using GridSearchCV

- Model comparison across multiple algorithms

- Logging, exception handling, and artifact management
ML_Pipeline_Income_Prediction/
│
├── artifacts/
│   ├── data_ingestion/
│   │   ├── raw.csv
│   │   ├── train.csv
│   │   └── test.csv
│   ├── data_transformation/
│   │   └── preprocessor.pkl
│   └── model_trainer/
│       └── model.pkl
│
├── notebook/
│   └── data/
│       └── income_cleandata.csv
│
├── logs/
│   └── *.log
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
│
├── setup.py
├── requirements.txt
└── README.md


📘 Project Overview

The goal of this project is to classify whether a person earns more than 50K a year, using demographic attributes such as:

- Age

- Education level

- Workclass

- Marital status

- Occupation

- Hours per week

- Capital gain/loss

- Native country

And more

The pipeline is modular, maintainable, and production-ready, with clean separation of concerns.

⚙️ Key Features
1️⃣ Data Ingestion

- Loads cleaned dataset from the notebook/data directory

- Splits into train (70%) and test (30%)

- Saves artifacts in:

  artifacts/data_ingestion/raw.csv, train.csv, test.csv

2️⃣ Data Transformation

- Handles outliers using IQR capping

- Applies median imputation for missing values

- Scales features using StandardScaler

- Uses ColumnTransformer and Pipeline

- Saves the preprocessor as:
  artifacts/data_transformation/preprocessor.pkl

3️⃣ Model Training

Trains and tunes 3 models:

- Logistic Regression

- Decision Tree

- Random Forest

###### Uses GridSearchCV for hyperparameter optimization.

Outputs:

- Best model name

- Best accuracy score

- Saves final model to:
  artifacts/model_trainer/model.pkl

4️⃣ Prediction Pipeline

Provides:

- CustomClass → Converts user inputs to DataFrame

- PredictionPipeline → Loads model + preprocessor and predicts output

🧠 Best Model Selection

The pipeline:

- Evaluates all models

- Compares accuracy

- Selects the best-performing one automatically

- Logged and printed:

- Best Model Found, Model Name is : <model>, Accuracy_Score: <value>

🗂️ Logging & Exception Handling

*All logs stored in /logs/<timestamp>/logfile.log*

*Custom exception class produces descriptive error messages*:

- file name

- line number

- root cause

📦 Installation
git clone <repo_link>
cd Income_Prediction_Pipeline
pip install -r requirements.txt


For development install:

pip install -e .

🚀 Running the Training Pipeline
python src/pipeline/training_pipeline.py


This will:

- Ingest data

- Transform data

- Train and save the best model

🧪 Running Predictions

Example:

from src.pipeline.prediction_pipeline import CustomClass, PredictionPipeline

input_data = CustomClass(
    age=35,
    workclass=2,
    education_num=13,
    marital_status=1,
    occupation=4,
    relationship=2,
    race=1,
    sex=1,
    capital_gain=0,
    capital_loss=0,
    hours_per_week=40,
    native_country=1
)

df = input_data.get_data_DataFrame()
pipeline = PredictionPipeline()
result = pipeline.predict(df)
print(result)

📊 Dataset

The project uses a cleaned version of the Adult Census dataset with the following columns:

age, workclass, fnlwgt, education, education-num, marital-status,
occupation, relationship, race, sex, capital-gain, capital-loss,
hours-per-week, country, salary


Target variable: salary (>50K or <=50K)

🛠️ Tech Stack
Component	Tools Used
Language	Python
ML	Scikit-Learn
Logging	Python Logging
Exception Handling	Custom Classes
Model Storage	Pickle
Processing	Pandas, NumPy
📄 License

This project is free to use for educational and learning purposes.