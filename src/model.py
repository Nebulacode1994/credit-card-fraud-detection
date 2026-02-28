import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib




base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(base_dir)
data_path = os.path.join(project_dir,"data","creditcard.csv")


df = pd.read_csv(data_path)


x = df.drop('Class', axis = 1)
y = df['Class']

pipe = Pipeline ([
    ( "scaler" , StandardScaler()),
    ("model" , LogisticRegression(
        max_iter = 2000,
        class_weight = 'balanced',
        solver = "liblinear",
        C = 0.5
        ))
    ])


x_train, x_test, y_train, y_test = train_test_split (x,y,test_size = 0.2 , random_state = 42)

pipe.fit(x_train, y_train)

y_proba = pipe.predict_proba(x_test)[:, 1]

for thr in [0.5, 0.7, 0.8, 0.9, 0.95]:
    y_pred_thr = (y_proba >= thr). astype(int)
    print("\nThreshold:", thr)
print(classification_report(y_test, y_pred_thr))

joblib.dump(pipe, "fraud_model.pkl")

print("model saved successfuly")


 