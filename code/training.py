import pandas as pd
import numpy as np
import joblib
import os
from sklearn.impute import KNNImputer
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def prep_data(df):

    # Imputar valores en la columna embarked
    df.loc[df["Embarked"].isnull(),"Embarked"] = "S"

    # Codificar las varibales categoricas
    df.Sex = df.Sex.map({"male":0,"female":1})
    dummy = pd.get_dummies(df.Embarked, prefix = "Embarked")
    df = pd.concat([df, dummy], axis=1)

    # Crear variables
    df["Alone"] = ((df.SibSp.eq(0) & df.Parch.eq(0))*1.0).astype("int64")
    df["Nulos"] = (df["Age"].isnull()*1.0).astype("int64")
    if "Survived" in df.columns:
        return df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked_C","Embarked_Q","Embarked_S","Alone","Nulos","Survived"]]
    else:
        return df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked_C","Embarked_Q","Embarked_S","Alone","Nulos"]]
    
def split_data(df):
    X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked_C","Embarked_Q","Embarked_S","Alone","Nulos"]]
    # Separar target
    y = df["Survived"]

    # Separar train y test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, shuffle = True, random_state = 0)
    data = {"train": {"X": X_train, "y": y_train},"test": {"X": X_test, "y": y_test}}
    return data

def train_model(data, args):
    # Pipeline para imoputar valores, seleccionar variables y estadarizar
    knn = KNNImputer(missing_values = np.nan, **args["knn"])
    fs = SelectKBest(score_func=f_classif, **args["select"])
    sc = StandardScaler()

    # Creacion del modelo
    randfor = RandomForestClassifier(**args["forest"])

    # Entrenar pipeline
    pipe = make_pipeline(knn,sc,fs,randfor).fit(data["train"]["X"],data["train"]["y"])
    return pipe
    
def get_metrics(model,data):
    # Realizar la prediccion
    y_pred = model.predict(data["test"]["X"])

    # Sacar score
    score = accuracy_score(y_pred,data["test"]["y"])
    return {"accuracy":score}

def main():
    # Cargar los datos
    df = pd.read_csv("../data/train.csv")
    prep = prep_data(df)
    data = split_data(prep)
    
    args = {
        "knn":{
            "n_neighbors":5
        },
        "select":{
            "k":9
        },
        "forest":{
            "criterion":"entropy",
            "n_estimators":50,
            "min_samples_split":15
        }
    }
    

    model_name = "random_forest_prueba.pkl"
    dirname = os.path.dirname("training.py")
    filename = os.path.join(dirname, '../models/'+model_name)
    
    model = train_model(data,args)
    metrics = get_metrics(model,data)

    joblib.dump(value = model, filename = filename)
if __name__ == "__main__":
    main()

