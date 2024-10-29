from flask import Flask, request, jsonify
import psycopg2
from model import predict_fraud

app = Flask(__name__)

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="your_dbname",
    user="your_username",
    password="your_password",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = data['features']
    
    result = predict_fraud(input_data)
    
    if result == 1:  # If flagged as fraud
        cursor.execute("INSERT INTO flagged_transactions (transaction_data) VALUES (%s)", (str(input_data),))
        conn.commit()
    
    return jsonify({'fraud': bool(result)})

if __name__ == '__main__':
    app.run(debug=True)
