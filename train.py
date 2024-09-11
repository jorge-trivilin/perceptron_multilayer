#!/usr/bin/env python3
# train.py
import pandas as pd
import numpy as np
from perceptron_multilayer_random_search import PerceptronMultilayer as PerceptronMultilayer # type: ignore
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.model_selection import train_test_split, KFold  # type: ignore
from sklearn.metrics import log_loss  # type: ignore
from joblib import Parallel, delayed  # type: ignore
import logging
import argparse
from scipy.stats import uniform, randint  # type: ignore

# Define global seed
RANDOM_SEED = 42


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )  # 'force=True' ensures the configuration is reapplied.
    logger = logging.getLogger(__name__)
    return logger


def create_preprocessor(numeric_features, categorical_features, logger):
    logger.info("Creating preprocessor")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def train_and_evaluate(
    X_train,
    y_train,
    X_val,
    y_val,
    params,
    numeric_features,
    categorical_features,
    logger,
):
    try:
        # Create a new preprocessor for each fold
        preprocessor = create_preprocessor(
            numeric_features, categorical_features, logger
        )
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)

        # Convert y to Numpy array if it's not already
        y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
        y_val = y_val.to_numpy() if isinstance(y_val, pd.Series) else y_val

        input_size = X_train_processed.shape[1]
        perceptron = PerceptronMultilayer(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            output_size=1,
            learning_rate=params["learning_rate"],
            logger=logger,
            random_seed=RANDOM_SEED,
        )

        # Train the model
        perceptron.fit(
            X_train_processed,
            y_train,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
        )

        # Calculate metrics for the training set
        y_train_pred = perceptron.predict_proba(X_train_processed)
        train_log_loss = log_loss(y_train, y_train_pred)
        train_accuracy = perceptron.evaluate(X_train_processed, y_train)

        # Calculate metrics for the validation set
        y_val_pred = perceptron.predict_proba(X_val_processed)
        val_log_loss = log_loss(y_val, y_val_pred)
        val_accuracy = perceptron.evaluate(X_val_processed, y_val)

        logger.info(
            f"Train Log Loss: {train_log_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )
        logger.info(
            f"Validation Log Loss: {val_log_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

    except Exception as e:
        logger.error(f"Error during training and evaluation: {e}")
        # Return a very high loss and low accuracy to penalize the error
        train_log_loss, train_accuracy, val_log_loss, val_accuracy = (
            float("inf"),
            0.0,
            float("inf"),
            0.0,
        )

    return train_log_loss, train_accuracy, val_log_loss, val_accuracy


def random_search_cv(X, y, param_distributions, n_iter=2, n_splits=5, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    logger.info(
        f"Starting random search with {n_iter} iterations and {n_splits}-fold cross-validation"
    )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    def evaluate_params(params):
        fold_scores = []
        for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            (
                train_log_loss,
                train_accuracy,
                val_log_loss,
                val_accuracy,
            ) = train_and_evaluate(
                X_train_fold,
                y_train_fold,
                X_val_fold,
                y_val_fold,
                params,
                numeric_features,
                categorical_features,
                logger,
            )

            fold_scores.append(
                (fold, train_log_loss, train_accuracy, val_log_loss, val_accuracy)
            )
            logger.info(f"Fold {fold}:")
            logger.info(
                f"  Train: Log Loss = {train_log_loss:.4f}, Accuracy = {train_accuracy:.4f}"
            )
            logger.info(
                f"  Validation: Log Loss = {val_log_loss:.4f}, Accuracy = {val_accuracy:.4f}"
            )

        mean_train_log_loss = np.mean([s[1] for s in fold_scores])
        mean_train_accuracy = np.mean([s[2] for s in fold_scores])
        mean_val_log_loss = np.mean([s[3] for s in fold_scores])
        mean_val_accuracy = np.mean([s[4] for s in fold_scores])

        logger.info("Mean scores across all folds:")
        logger.info(
            f"  Train: Log Loss = {mean_train_log_loss:.4f}, Accuracy = {mean_train_accuracy:.4f}"
        )
        logger.info(
            f"  Validation: Log Loss = {mean_val_log_loss:.4f}, Accuracy = {mean_val_accuracy:.4f}"
        )

        return (
            params,
            fold_scores,
            mean_train_log_loss,
            mean_train_accuracy,
            mean_val_log_loss,
            mean_val_accuracy,
        )

    random_params = []
    for _ in range(n_iter):
        params = {k: v.rvs() for k, v in param_distributions.items()}
        random_params.append(params)

    results = [
        r
        for r in Parallel(n_jobs=-1)(
            delayed(evaluate_params)(params) for params in random_params
        )
        if r is not None
    ]

    if results:
        best_result = min(results, key=lambda x: x[4])  # x[4] is mean_val_log_loss
        (
            best_params,
            best_fold_scores,
            best_mean_train_log_loss,
            best_mean_train_accuracy,
            best_mean_val_log_loss,
            best_mean_val_accuracy,
        ) = best_result
        logger.info(f"Random search completed. Best parameters: {best_params}")
        logger.info("Best mean scores:")
        logger.info(
            f"  Train: Log Loss = {best_mean_train_log_loss:.4f}, Accuracy = {best_mean_train_accuracy:.4f}"
        )
        logger.info(
            f"  Validation: Log Loss = {best_mean_val_log_loss:.4f}, Accuracy = {best_mean_val_accuracy:.4f}"
        )
        return best_params, results
    else:
        logger.error("No valid results were found during random search.")
        return None, []


def main(data_fraction):
    logger = configure_logging()
    logger.info("Starting the training process")

    # Load data
    logger.info("Loading data")
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    # Sample the data if fraction is less than 1
    if data_fraction < 1.0:
        logger.info(f"Sampling {data_fraction*100}% of the data")
        X = X.sample(frac=data_fraction, random_state=RANDOM_SEED)
        y = y[X.index]
    else:
        logger.info("Using 100% of the data")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    logger.info(
        f"Data split: Training set size: {len(X_train)}, Test set size: {len(X_test)}"
    )

    param_distributions = {
        "hidden_size": randint(4, 32),
        "learning_rate": uniform(
            0.001, 0.099
        ),  # This will sample between 0.001 and 0.1
        "epochs": randint(500, 1001),
        "batch_size": randint(32, 65),
    }

    # random search
    best_params, all_results = random_search_cv(
        X_train, y_train, param_distributions, n_iter=5
    )

    # Check if best_params is valid
    if best_params is None:
        logger.error("Random search did not return valid best parameters. Exiting.")
        exit(1)

    logger.info("Random search results:")
    for (
        params,
        fold_scores,
        mean_train_log_loss,
        mean_train_accuracy,
        mean_val_log_loss,
        mean_val_accuracy,
    ) in sorted(all_results, key=lambda x: x[4])[:5]:
        logger.info(f"Params: {params}")
        logger.info(
            f"Mean Train: Log Loss = {mean_train_log_loss:.4f}, Accuracy = {mean_train_accuracy:.4f}"
        )
        logger.info(
            f"Mean Validation: Log Loss = {mean_val_log_loss:.4f}, Accuracy = {mean_val_accuracy:.4f}"
        )
        logger.info("Fold Results:")
        for (
            fold,
            train_log_loss,
            train_accuracy,
            val_log_loss,
            val_accuracy,
        ) in fold_scores:
            logger.info(f"  Fold {fold}:")
            logger.info(
                f"    Train: Log Loss = {train_log_loss:.4f}, Accuracy = {train_accuracy:.4f}"
            )
            logger.info(
                f"    Validation: Log Loss = {val_log_loss:.4f}, Accuracy = {val_accuracy:.4f}"
            )
        logger.info("")

    # Train final model with best parameters
    logger.info("Training final model with best parameters")
    final_preprocessor = create_preprocessor(
        numeric_features, categorical_features, logger
    )
    X_train_processed = final_preprocessor.fit_transform(X_train)
    X_test_processed = final_preprocessor.transform(X_test)

    final_model = PerceptronMultilayer(
        input_size=X_train_processed.shape[1],
        hidden_size=best_params["hidden_size"],
        output_size=1,
        learning_rate=best_params["learning_rate"],
        logger=logger,
        random_seed=RANDOM_SEED,
    )
    final_model.fit(
        X_train_processed,
        y_train,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
    )

    # Final evaluation
    logger.info("Evaluating final model on test set")
    y_test_pred = final_model.predict_proba(X_test_processed)
    test_log_loss = log_loss(y_test, y_test_pred)
    test_accuracy = final_model.evaluate(X_test_processed, y_test)

    logger.info(f"Final results for {data_fraction*100}% of the data:")
    logger.info(f"  Test Log Loss: {test_log_loss:.4f}")
    logger.info(f"  Test Accuracy: {test_accuracy:.4f}")

    logger.info("Training process completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Perceptron with specified data fraction"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=51.0,
        help="Fraction of data to use (0.0 to 1.0)",
    )
    args = parser.parse_args()

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    main(args.fraction)
