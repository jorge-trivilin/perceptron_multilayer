import pandas as pd
import numpy as np
from perceptron_multilayer import PerceptronMultilayer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss, accuracy_score
from joblib import Parallel, delayed

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

# Função para treinar o perceptron
def train_perceptron(X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold, preprocessor):
    X_train_processed = preprocessor.fit_transform(X_train_fold)
    X_val_processed = preprocessor.transform(X_val_fold)
    
    input_size = X_train_processed.shape[1]
    hidden_size = 4
    output_size = 1
    
    perceptron = PerceptronMultilayer(input_size=input_size,
                                      hidden_size=hidden_size,
                                      output_size=output_size)
    
    perceptron.fit(X_train_processed, y_train_fold, epochs=1000)
    
    # Previsões de treinamento e validação
    y_train_pred = perceptron.predict_proba(X_train_processed)
    y_val_pred = perceptron.predict_proba(X_val_processed)
    
    # Cálculo do log loss
    train_log_loss = log_loss(y_train_fold, y_train_pred)
    val_log_loss = log_loss(y_val_fold, y_val_pred)
    
    # Cálculo da acurácia
    val_accuracy = accuracy_score(y_val_fold, y_val_pred.round())
    
    # Exibição dos resultados de cada fold
    print(f"Fold {fold}:")
    print(f"  Training Log Loss: {train_log_loss:.4f}")
    print(f"  Validation Log Loss: {val_log_loss:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    
    return train_log_loss, val_log_loss, val_accuracy

# Função para realizar cross-validation
def cross_validate_perceptron(X_train, y_train, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preprocessor = create_preprocessor()

    # Executar cross-validation em paralelo
    results = Parallel(n_jobs=-1)(delayed(train_perceptron)(
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
print("Realizando validação cruzada...")
mean_train_log_loss, mean_val_log_loss, mean_val_accuracy = cross_validate_perceptron(X_train, y_train)
print(f"Mean cross-validation Training Log Loss: {mean_train_log_loss:.4f}")
print(f"Mean cross-validation Validation Log Loss: {mean_val_log_loss:.4f}")
print(f"Mean cross-validation accuracy: {mean_val_accuracy:.4f}")

# Treinar o modelo final nos dados de treino completos
print("\nTreinando o modelo final nos dados de treino completos...")
preprocessor = create_preprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

input_size = X_train_processed.shape[1]
hidden_size = 4
output_size = 1

perceptron = PerceptronMultilayer(input_size=input_size,
                                  hidden_size=hidden_size,
                                  output_size=output_size)

# Treinar o modelo final
perceptron.fit(X_train_processed, y_train, epochs=1000)

# Avaliação final
y_test_pred = perceptron.predict_proba(X_test_processed)
test_log_loss = log_loss(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred.round())

print("Resultados finais:")
print(f"  Test Log Loss: {test_log_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}")
