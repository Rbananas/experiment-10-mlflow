
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import shutil
import os

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Set the MLflow experiment
mlflow.set_experiment("Iris_Classification")

# Three configurations from the reference experiment
configurations = [
    (100, 5),
    (50, 3),
    (200, 10)
]

for n_estimators, max_depth in configurations:

    # Start a named MLflow run
    with mlflow.start_run(
        run_name=f"random-forest-{n_estimators}-{max_depth}"
    ):

        # Create the Random Forest model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        # Train the model
        model.fit(X_train, y_train)

        # Generate predictions
        predictions = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters and metric
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset", "scikit-learn Iris")
        mlflow.log_metric("accuracy", accuracy)

        # Log model as an MLflow artifact
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print(
            f"Run: random-forest-{n_estimators}-{max_depth}"
        )
        print(f"n_estimators: {n_estimators}")
        print(f"max_depth: {max_depth}")
        print(f"Model Accuracy: {accuracy:.4f}")
        print("-" * 40)

        # Save the final configuration locally for deployment
        if (n_estimators, max_depth) == (200, 10):

            # Remove the previous local model if it exists
            if os.path.exists("model"):
                shutil.rmtree("model")

            mlflow.sklearn.save_model(
                model,
                "model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

            print("Final model saved to ./model")
