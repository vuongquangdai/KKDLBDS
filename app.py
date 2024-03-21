from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

### Xây dựng mô hình Linear Regression
data = pd.read_csv('./csdl/data.csv', delimiter=';')

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = data[['dt_san', 'chieu_rong', 'phong_ngu', 'phong_tam', 'duong_vao']]
y = data['muc_gia']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train.values, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    dt_san = float(data['dt_san'])
    chieu_rong = float(data['chieu_rong'])
    phong_ngu = int(data['phong_ngu'])
    phong_tam = int(data['phong_tam'])
    duong_vao = float(data['duong_vao'])

    # Chuẩn bị dữ liệu đầu vào cho mô hình Linear Regression
    input_data = np.array([dt_san, chieu_rong, phong_ngu, phong_tam, duong_vao]).reshape(1, -1)

    # Sử dụng mô hình để dự đoán giá nhà
    predicted_price = model.predict(input_data)

    return jsonify({'predicted_price': round(predicted_price[0],4)})

if __name__ == '__main__':
    app.run(debug=True)