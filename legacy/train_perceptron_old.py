# train_perceptron.py
import pandas as pd
import numpy as np
from perceptron_multilayer import PerceptronMultilayer
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.impute import SimpleImputer  # type: ignore
from sklearn.model_selection import train_test_split, KFold  # type: ignore


df = pd.read_csv(
    "/media/kz/HDD/Development/perceptron_multilayer/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Converter TotalCharges para numérico
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Separar features e target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# Dividir em conjunto de treino e teste (feito antes do pré-processamento para evitar data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Definir colunas numéricas e categóricas
numeric_features = ["tenure", "MonthlyCharges", "TotalPCharges"]
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

# Criar o pipeline de pré-processamento
def create_preprocessor():
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


# Função para calcular o log loss
def log_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Função para realizar cross-validation com perceptron multicamadas
def cross_validate_perceptron(X_train, y_train, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
        # Dividir os dados entre treino e validação para o fold atual
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Criar o preprocessador e ajustá-lo aos dados de treino do fold
        preprocessor = create_preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train_fold)
        X_val_processed = preprocessor.transform(X_val_fold)

        # Criar uma instância do perceptron para este fold
        input_size = X_train_processed.shape[1]
        hidden_size = 4
        output_size = 1

        perceptron = PerceptronMultilayer(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size
        )

        # Treinar o modelo
        perceptron.fit(X_train_processed, y_train_fold, epochs=1000)

        # Calcular log loss para dados de treinamento e validação
        y_train_pred = perceptron.predict_proba(X_train_processed)
        train_log_loss = log_loss(y_train_fold, y_train_pred)

        y_val_pred = perceptron.predict_proba(X_val_processed)
        val_log_loss = log_loss(y_val_fold, y_val_pred)

        # Calcular acurácia para dados de validação
        accuracy = perceptron.evaluate(X_val_processed, y_val_fold)
        cv_scores.append(accuracy)

        print(f"Fold {fold}:")
        print(f"  Training Log Loss: {train_log_loss:.4f}")
        print(f"  Validation Log Loss: {val_log_loss:.4f}")
        print(f"  Validation Accuracy: {accuracy:.4f}")

    return np.mean(cv_scores)


# Aplicar cross-validation no conjunto de treino
print("Realizando validação cruzada...")
mean_cv_score = cross_validate_perceptron(X_train, y_train)
print(f"Mean cross-validation accuracy: {mean_cv_score:.4f}")

# Após a validação cruzada, treinar o modelo final nos dados de treino completos
print("\nTreinando o modelo final nos dados de treino completos...")
preprocessor = create_preprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

input_size = X_train_processed.shape[1]
hidden_size = 4
output_size = 1

perceptron = PerceptronMultilayer(
    input_size=input_size, hidden_size=hidden_size, output_size=output_size
)

# Treinar o modelo final
perceptron.fit(X_train_processed, y_train, epochs=1000)

# Calcular log loss e acurácia para dados de treinamento e teste
y_train_pred = perceptron.predict_proba(X_train_processed)
train_log_loss = log_loss(y_train, y_train_pred)
train_accuracy = perceptron.evaluate(X_train_processed, y_train)

y_test_pred = perceptron.predict_proba(X_test_processed)
test_log_loss = log_loss(y_test, y_test_pred)
test_accuracy = perceptron.evaluate(X_test_processed, y_test)

print("Resultados finais:")
print(f"  Training Log Loss: {train_log_loss:.4f}")
print(f"  Training Accuracy: {train_accuracy:.4f}")
print(f"  Test Log Loss: {test_log_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}")
