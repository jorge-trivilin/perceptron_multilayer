import pandas as pd
import numpy as np
from perceptron_multilayer_grid_search import PerceptronMultilayer as PerceptronMultilayer  # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.model_selection import train_test_split, KFold  # type: ignore
from sklearn.metrics import log_loss, accuracy_score  # type: ignore
from joblib import Parallel, delayed  # type: ignore
import logging
import argparse


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = configure_logging()


def create_preprocessor(numeric_features, categorical_features):
    logger.info("Creating preprocessor")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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
    X_train, y_train, X_val, y_val, params, numeric_features, categorical_features
):
    try:
        # Cria um novo preprocessor para cada fold
        preprocessor = create_preprocessor(numeric_features, categorical_features)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)

        # Converta y para Numpy array, se não for
        y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
        y_val = y_val.to_numpy() if isinstance(y_val, pd.Series) else y_val

        input_size = X_train_processed.shape[1]
        perceptron = PerceptronMultilayer(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            output_size=1,
            learning_rate=params["learning_rate"],
            logger=logger,
        )

        # Treina o modelo
        perceptron.fit(
            X_train_processed,
            y_train,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
        )

        # Faz a previsão
        y_val_pred = perceptron.predict_proba(X_val_processed)
        val_log_loss = log_loss(y_val, y_val_pred)
        val_accuracy = accuracy_score(y_val, (y_val_pred >= 0.5).astype(int))

    except Exception as e:
        logger.error(f"Error during training and evaluation: {e}")
        # Retorne uma perda muito alta e acurácia baixa para penalizar o erro
        val_log_loss = float("inf")
        val_accuracy = 0.0

    return val_log_loss, val_accuracy


def grid_search_cv(
    X, y, param_grid, numeric_features, categorical_features, n_splits=5
):
    logger.info(f"Starting grid search with {n_splits}-fold cross-validation")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def evaluate_params(params):
        scores = []
        for train_index, val_index in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            # Chama a função de treinamento e avaliação
            val_log_loss, val_accuracy = train_and_evaluate(
                X_train_fold,
                y_train_fold,
                X_val_fold,
                y_val_fold,
                params,
                numeric_features,
                categorical_features,
            )

            # Adiciona os resultados à lista de scores
            scores.append((val_log_loss, val_accuracy))

        # Calcula a média dos resultados
        mean_log_loss = np.mean([s[0] for s in scores])
        mean_accuracy = np.mean([s[1] for s in scores])
        return params, mean_log_loss, mean_accuracy

    # Executa o grid search de forma paralela e filtra os resultados não válidos
    results = [
        r
        for r in Parallel(n_jobs=-1)(
            delayed(evaluate_params)(params) for params in param_grid
        )
        if r is not None
    ]

    if results:
        # Acessa os melhores parâmetros
        best_params = min(results, key=lambda x: x[1])[0]
        logger.info(f"Grid search completed. Best parameters: {best_params}")
        return best_params, results
    else:
        logger.error("No valid results were found during grid search.")
        return None, []


def main(data_fraction):
    logger.info("Starting the training process")

    # Load data
    logger.info("Loading data")
    df = pd.read_csv(
        "/media/kz/HDD/Development/perceptron_multilayer/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    # Sample the data if fraction is less than 1
    if data_fraction < 1.0:
        logger.info(f"Sampling {data_fraction*100}% of the data")
        X = X.sample(frac=data_fraction, random_state=42)
        y = y[X.index]
    else:
        logger.info("Using 100% of the data")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(
        f"Data split: Training set size: {len(X_train)}, Test set size: {len(X_test)}"
    )

    # Defining parameter grid with 3 (hidden_size) * 3 (learning_rate) * 2 (epochs) * 2 (batch_size) = 36 iterations.
    param_grid = [
        {
            "hidden_size": hidden_size,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
        }
        for hidden_size in [4, 8, 16]
        for lr in [0.001, 0.01, 0.1]
        for epochs in [500, 1000]
        for batch_size in [32, 64]
    ]  # 36 iterations * 5 folds = 180 training and evaluation during grid search process.

    # Grid search
    best_params, all_results = grid_search_cv(
        X_train, y_train, param_grid, numeric_features, categorical_features
    )

    # Verificar se best_params é válido
    if best_params is None:
        logger.error("Grid search did not return valid best parameters. Exiting.")
        exit(1)  # Ou levante uma exceção, dependendo da sua estratégia de erro

    logger.info("Grid search results:")
    for params, log_loss, accuracy in sorted(all_results, key=lambda x: x[1])[:5]:
        logger.info(f"Params: {params}")
        logger.info(f"Log Loss: {log_loss:.4f}, Accuracy: {accuracy:.4f}\n")

    # Train final model with best parameters
    logger.info("Training final model with best parameters")
    final_preprocessor = create_preprocessor(numeric_features, categorical_features)
    X_train_processed = final_preprocessor.fit_transform(X_train)
    X_test_processed = final_preprocessor.transform(X_test)

    final_model = PerceptronMultilayer(
        input_size=X_train_processed.shape[1],
        hidden_size=best_params["hidden_size"],
        output_size=1,
        learning_rate=best_params["learning_rate"],
        logger=logger,
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
    test_accuracy = accuracy_score(y_test, (y_test_pred >= 0.5).astype(int))

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
        default=1.0,
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
