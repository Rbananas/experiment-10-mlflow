# Experiment 10: Machine Learning Lifecycle Using MLflow

## Aim
To study and implement the machine learning lifecycle using MLflow, including model development, experiment tracking, model logging, and model deployment.

## Technologies Used
- Python
- MLflow
- Scikit-learn
- Pandas
- Iris dataset
- Random Forest Classifier

## Project Description
This project demonstrates the machine learning lifecycle using MLflow. A Random Forest Classifier is trained on the Iris dataset. MLflow tracks experiment runs, model parameters, accuracy metrics, and model artifacts.

## Experiments
Three configurations are evaluated:

| Run Name | Number of Estimators | Maximum Depth |
|---|---:|---:|
| random-forest-100-5 | 100 | 5 |
| random-forest-50-3 | 50 | 3 |
| random-forest-200-10 | 200 | 10 |

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the Experiment

```bash
python train.py
```

## Start the MLflow Dashboard

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Open http://localhost:5000 and select the `Iris_Classification` experiment.

## Model Deployment

```bash
mlflow models serve -m model -p 5001 --host 127.0.0.1 --env-manager=local
```

Send a prediction request to `http://localhost:5001/invocations` using the `input.json` file.

## Conclusion
The experiment demonstrates model training, experiment tracking, model logging, saving
