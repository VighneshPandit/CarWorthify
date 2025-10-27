from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and columns
model = joblib.load("car_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")


@app.route('/')
def home():
    # Render the HTML frontend
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            #Handle POSTMAN JSON request
            data = request.get_json()

            manufacturer = data.get('manufacturer')
            model_name = data.get('model')
            fuel_type = data.get('fuel_type')
            engine_size = float(data.get('engine_size'))
            year = int(data.get('year'))
            mileage = int(data.get('mileage'))
        else:
            #Handle HTML form submission
            manufacturer = request.form['manufacturer']
            model_name = request.form['model']
            fuel_type = request.form['fuel_type']
            engine_size = float(request.form['engine_size'])
            year = int(request.form['year'])
            mileage = int(request.form['mileage'])

        # Create DataFrame
        input_df = pd.DataFrame([{
            'Manufacturer': manufacturer,
            'Model': model_name,
            'Engine size': engine_size,
            'Fuel type': fuel_type,
            'Year of manufacture': year,
            'Mileage': mileage
        }])

        # One-hot encode & align columns
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Predict
        pred = model.predict(input_encoded)
        pred_exp = np.exp(pred)
        predicted_price = round(pred_exp[0], 2)

        # Return based on request type
        if request.is_json:
            # Return JSON response (for Postman)
            return jsonify({"predicted_price": predicted_price})
        else:
            # Return HTML page with result (for browser)
            return render_template('index.html', prediction_text=f'Predicted Price: $ {predicted_price:,}')

    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 400
        else:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
