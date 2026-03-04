from flask import Flask, request, jsonify
import joblib
import pandas as pd 
import os 

app = Flask(__name__)

base_dir = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")

project_dir = os.path.dirname(base_dir)


@app.route('/predict', methods = ['POST'])
def predict():
    try:
        
        data = request.json()
        
        df_input = pd.DataFrame(data)
        
        prediction = model.predict(df_input)
        probability = model.predict_proba(df_input)[:, 1]
        
        return jsonify({
            "is_fraud": int(prediction[0]),
            "fraud_probability": float(probability[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug = True, port = 5000)
    
    