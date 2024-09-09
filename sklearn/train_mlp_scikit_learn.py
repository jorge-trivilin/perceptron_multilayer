import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore 
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.model_selection import train_test_split, KFold # type: ignore
from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.metrics import log_loss, accuracy_score # type: ignore
from joblib import Parallel, delayed # type: ignore

# Função para criar o pré-processador
def create_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Função para treinar o MLP do scikit-learn
def train_mlp(X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold, preprocessor):
    # Pré-processar os dados
    X_train_processed = preprocessor.fit_transform(X_train_fold)
    X_val_processed = preprocessor.transform(X_val_fold)
    
    # Definir o MLP
    mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42)

    # Treinar o modelo
    mlp.fit(X_train_processed, y_train_fold)
    
    # Previsões de treinamento e validação
    y_train_pred_proba = mlp.predict_proba(X_train_processed)[:, 1]
    y_val_pred_proba = mlp.predict_proba(X_val_processed)[:, 1]
    
    # Cálculo do log loss
    train_log_loss = log_loss(y_train_fold, y_train_pred_proba)
    val_log_loss = log_loss(y_val_fold, y_val_pred_proba)
    
    # Cálculo da acurácia
    val_accuracy = accuracy_score(y_val_fold, mlp.predict(X_val_processed))
    
    # Exibir resultados de cada fold
    print(f"Fold {fold}:")
    print(f"  Training Log Loss: {train_log_loss:.4f}")
    print(f"  Validation Log Loss: {val_log_loss:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    
    return train_log_loss, val_log_loss, val_accuracy

# Função para realizar cross-validation
def cross_validate_mlp(X_train, y_train, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preprocessor = create_preprocessor()

    # Executar cross-validation em paralelo
    results = Parallel(n_jobs=-1)(delayed(train_mlp)(
        X_train.iloc[train_index], y_train.iloc[train_index],
        X_train.iloc[val_index], y_train.iloc[val_index],
        fold, preprocessor
    ) for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1))
    
    # Separar resultados
    train_log_losses, val_log_losses, val_accuracies = zip(*results)
    
    # Calcular e exibir as médias
    mean_train_log_loss = np.mean(train_log_losses)
    mean_val_log_loss = np.mean(val_log_losses)
    mean_val_accuracy = np.mean(val_accuracies)
    
    return mean_train_log_loss, mean_val_log_loss, mean_val_accuracy

# Carregar e preparar os dados
df = pd.read_csv("/media/kz/HDD/Development/perceptron_multilayer/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Converter TotalCharges para numérico
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Separar features e target
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Dividir em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir colunas numéricas e categóricas
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                        'Contract', 'PaperlessBilling', 'PaymentMethod']

# Aplicar cross-validation no conjunto de treino
print("Realizando validação cruzada com MLP...")
mean_train_log_loss, mean_val_log_loss, mean_val_accuracy = cross_validate_mlp(X_train, y_train)
print(f"Mean cross-validation Training Log Loss: {mean_train_log_loss:.4f}")
print(f"Mean cross-validation Validation Log Loss: {mean_val_log_loss:.4f}")
print(f"Mean cross-validation accuracy: {mean_val_accuracy:.4f}")

# Treinar o modelo final nos dados de treino completos
print("\nTreinando o modelo final nos dados de treino completos com MLP...")
preprocessor = create_preprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

mlp_final = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42)
mlp_final.fit(X_train_processed, y_train)

# Avaliação final
y_test_pred_proba = mlp_final.predict_proba(X_test_processed)[:, 1]
test_log_loss = log_loss(y_test, y_test_pred_proba)
test_accuracy = accuracy_score(y_test, mlp_final.predict(X_test_processed))

print("Resultados finais com MLP:")
print(f"  Test Log Loss: {test_log_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}")
