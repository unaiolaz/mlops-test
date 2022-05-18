from training import prep_data
import pandas as pd

def test_data():
    df = pd.read_csv("../data/train.csv")
    assert set(["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]).issubset(set(list(df.columns)))
    
def test_prep_data():
    df = pd.read_csv("../data/train.csv")
    assert prep_data(df).Sex.isna().any() == False