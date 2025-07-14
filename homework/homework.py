# flake8: noqa: E501
import os
import json
import pickle
import gzip
import glob
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
)


def inicializar_directorio(path):
    if os.path.exists(path):
        for archivo in glob.glob(f"{path}/*"):
            os.remove(archivo)
        os.rmdir(path)
    os.makedirs(path)


def cargar_csvs():
    df_entrenamiento = pd.read_csv("./files/input/train_data.csv.zip", index_col=False, compression="zip")
    df_prueba = pd.read_csv("./files/input/test_data.csv.zip", index_col=False, compression="zip")
    return df_entrenamiento, df_prueba


def depurar_datos(df):
    df_nuevo = df.copy()
    df_nuevo.rename(columns={'default payment next month': "default"}, inplace=True)
    df_nuevo.drop(columns=["ID"], inplace=True)
    df_nuevo = df_nuevo[df_nuevo["MARRIAGE"] != 0]
    df_nuevo = df_nuevo[df_nuevo["EDUCATION"] != 0]
    df_nuevo["EDUCATION"] = df_nuevo["EDUCATION"].apply(lambda valor: 4 if valor >= 4 else valor)
    return df_nuevo.dropna()


def separar_variables(df):
    caracteristicas = df.drop(columns=["default"])
    objetivo = df["default"]
    return caracteristicas, objetivo


def armar_pipeline(columnas):
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    num_cols = list(set(columnas.columns) - set(cat_cols))

    transformador = ColumnTransformer(
        transformers=[
            ('categoricas', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('numericas', StandardScaler(), num_cols),
        ],
        remainder='passthrough'
    )

    pasos = Pipeline([
        ('transformador', transformador),
        ('pca', PCA()),
        ('kbest', SelectKBest(score_func=f_classif)),
        ('clasificador', SVC(kernel="rbf", random_state=12345, max_iter=-1)),
    ])
    return pasos


def configurar_entrenamiento(pipeline, columnas):
    parametros_grid = {
        "pca__n_components": [20, columnas.shape[1] - 2],
        "kbest__k": [12],
        "clasificador__kernel": ["rbf"],
        "clasificador__gamma": [0.1],
    }

    validacion = StratifiedKFold(n_splits=10)
    evaluador = make_scorer(balanced_accuracy_score)

    busqueda = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros_grid,
        scoring=evaluador,
        cv=validacion,
        n_jobs=-1
    )
    return busqueda


def guardar_modelo_final(ruta, modelo):
    inicializar_directorio("files/models/")
    with gzip.open(ruta, "wb") as archivo:
        pickle.dump(modelo, archivo)


def evaluar_metricas(etiqueta, y_verdadero, y_estimado):
    return {
        "type": "metrics",
        "dataset": etiqueta,
        "precision": precision_score(y_verdadero, y_estimado, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_verdadero, y_estimado),
        "recall": recall_score(y_verdadero, y_estimado, zero_division=0),
        "f1_score": f1_score(y_verdadero, y_estimado, zero_division=0),
    }


def evaluar_matriz_confusion(etiqueta, y_verdadero, y_estimado):
    matriz = confusion_matrix(y_verdadero, y_estimado)
    return {
        "type": "cm_matrix",
        "dataset": etiqueta,
        "true_0": {"predicted_0": int(matriz[0][0]), "predicted_1": int(matriz[0][1])},
        "true_1": {"predicted_0": int(matriz[1][0]), "predicted_1": int(matriz[1][1])},
    }


def ejecutar_pipeline():
    inicializar_directorio("files/output/")

    df_train, df_test = cargar_csvs()
    df_train = depurar_datos(df_train)
    df_test = depurar_datos(df_test)

    x_train, y_train = separar_variables(df_train)
    x_test, y_test = separar_variables(df_test)

    pipeline = armar_pipeline(x_train)
    modelo_final = configurar_entrenamiento(pipeline, x_train)
    modelo_final.fit(x_train, y_train)

    guardar_modelo_final(os.path.join("files/models/", "model.pkl.gz"), modelo_final)

    pred_train = modelo_final.predict(x_train)
    pred_test = modelo_final.predict(x_test)

    resultados = [
        evaluar_metricas("train", y_train, pred_train),
        evaluar_metricas("test", y_test, pred_test),
        evaluar_matriz_confusion("train", y_train, pred_train),
        evaluar_matriz_confusion("test", y_test, pred_test),
    ]

    with open("files/output/metrics.json", "w", encoding="utf-8") as archivo:
        for r in resultados:
            archivo.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ejecutar_pipeline()
