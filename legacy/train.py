# train.py
import pandas as pd
import numpy as np
from perceptron_multilayer import PerceptronMultilayer
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.model_selection import train_test_split, KFold  # type: ignore
from sklearn.metrics import log_loss, accuracy_score  # type: ignore
from joblib import Parallel, delayed  # type: ignore
import logging


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = configure_logging()

# Função para criar o pré-processador
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# Função para treinar o perceptron
def train_perceptron(
    X_train_fold,
    y_train_fold,
    X_val_fold,
    y_val_fold,
    fold,
    numeric_features,
    categorical_features,
):
    fold_logger = logging.getLogger(f"{__name__}.fold{fold}")
    fold_logger.info(f"Training perceptron for fold {fold}")

    # Criar e ajustar o preprocessor apenas para este fold
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train_fold)
    X_val_processed = preprocessor.transform(X_val_fold)

    input_size = X_train_processed.shape[1]
    hidden_size = 4
    output_size = 1

    perceptron = PerceptronMultilayer(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        logger=fold_logger,
    )

    perceptron.fit(X_train_processed, y_train_fold, epochs=1000)

    # Previsões e cálculos de métricas
    y_train_pred = perceptron.predict_proba(X_train_processed)
    y_val_pred = perceptron.predict_proba(X_val_processed)

    train_log_loss = log_loss(y_train_fold, y_train_pred)
    val_log_loss = log_loss(y_val_fold, y_val_pred)
    val_accuracy = accuracy_score(y_val_fold, y_val_pred.round())

    fold_logger.info(f"Fold {fold} results:")
    fold_logger.info(f"  Training Log Loss: {train_log_loss:.4f}")
    fold_logger.info(f"  Validation Log Loss: {val_log_loss:.4f}")
    fold_logger.info(f"  Validation Accuracy: {val_accuracy:.4f}")

    return train_log_loss, val_log_loss, val_accuracy


# Função para realizar cross-validation
def cross_validate_perceptron(X_train, y_train, n_splits=5):
    logger.info(f"Starting cross-validation with {n_splits} splits")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Executar cross-validation em paralelo
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(train_perceptron)(
            X_train.iloc[train_index],
            y_train.iloc[train_index],
            X_train.iloc[val_index],
            y_train.iloc[val_index],
            fold,
            numeric_features,
            categorical_features,
        )
        for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1)
    )

    # Processar resultados
    train_log_losses, val_log_losses, val_accuracies = zip(*results)

    mean_train_log_loss = np.mean(train_log_losses)
    mean_val_log_loss = np.mean(val_log_losses)
    mean_val_accuracy = np.mean(val_accuracies)

    logger.info("Cross-validation completed")
    logger.info(f"Mean cross-validation Training Log Loss: {mean_train_log_loss:.4f}")
    logger.info(f"Mean cross-validation Validation Log Loss: {mean_val_log_loss:.4f}")
    logger.info(f"Mean cross-validation accuracy: {mean_val_accuracy:.4f}")

    return mean_train_log_loss, mean_val_log_loss, mean_val_accuracy


def main():
    logger.info("Starting the training process")

    # Carregar e preparar os dados
    logger.info("Loading and preparing data")
    df = pd.read_csv(
        "/media/kz/HDD/Development/perceptron_multilayer/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    # df = data.sample(frac=0.1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(
        f"Data split: Training set size: {len(X_train)}, Test set size: {len(X_test)}"
    )

    # Cross-validation
    logger.info("Starting cross-validation")
    (
        mean_train_log_loss,
        mean_val_log_loss,
        mean_val_accuracy,
    ) = cross_validate_perceptron(X_train, y_train)

    # Treinar o modelo final
    logger.info("Training final model on complete training set")
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    input_size = X_train_processed.shape[1]
    hidden_size = 4
    output_size = 1

    perceptron = PerceptronMultilayer(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        logger=logger,
    )

    perceptron.fit(X_train_processed, y_train, epochs=1000)

    # Avaliação final
    logger.info("Evaluating final model on test set")
    y_test_pred = perceptron.predict_proba(X_test_processed)
    test_log_loss = log_loss(y_test, y_test_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred.round())

    logger.info("Final results:")
    logger.info(f"  Test Log Loss: {test_log_loss:.4f}")
    logger.info(f"  Test Accuracy: {test_accuracy:.4f}")

    logger.info("Training process completed")


if __name__ == "__main__":
    # Definir colunas numéricas e categóricas
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
    main()
