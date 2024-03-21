import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics
# Hiển thị dữ liệu
import matplotlib.pyplot as plt
import seaborn as sns



"""1. Đọc dữ liệu"""
# Đọc dữ liệu từ file CSV
data = pd.read_csv('./csdl/data.csv', delimiter=';')
print(f"{data.describe()}\n\n")
# print(f'{data.info()}\n\n')



"""2. Làm sạch dữ liệu"""
# Kiểm tra dữ liệu null và tính tỷ lệ dữ liệu null cho từng cột
null_percentage = data.isnull().sum() * 100 / data.shape[0]
print("Tỷ lệ dữ liệu null cho từng cột:")
print(f'{null_percentage}\n\n')

# Phát hiện dữ liệu ngoại lai (Outliers) cho các cột có giá trị số
numerical_columns = ['dien_tich','dt_san','chieu_rong','phong_ngu','phong_tam','duong_vao','kc_giao_thong','muc_gia']
fig, axs = plt.subplots(len(numerical_columns), figsize=(10, len(numerical_columns)))
for i, col in enumerate(numerical_columns):
    plt1 = sns.boxplot(data[col], ax=axs[i])
    plt1.set_title(f'Biểu đồ hộp cho {col}')
    
plt.tight_layout()
plt.show()



"""3. Phân tích khám phá (Exploratory Data Analysis – EDA)"""
# Biến đầu ra (muc_gia)
sns.boxplot(data['muc_gia'])
plt.show()
# Vẽ đồ thị phân tán (Scatter Plots) để phân tích mối quan hệ giữa biến đầu ra (muc_gia) với các biến đầu vào
# sns.pairplot(data, x_vars=['dien_tich','dt_san','chieu_rong','phong_ngu','phong_tam','huong_nha','mat_pho','duong_vao','kc_giao_thong','tien_ich','phap_ly'], y_vars='muc_gia', height=11, aspect=1, kind='scatter')
# plt.show()
# Dùng heatmap để xem mối tương quan giữa các biến.
sns.heatmap(data.corr(), cmap="YlGnBu", annot = True)
plt.show()
# Qua Heatmap và Scatter Plots, chúng ta thấy rằng biến 'dt_san' có vẻ tương quan nhất với 'muc_gia'. 
# Vì vậy, tiếp tục và thực hiện hồi quy tuyến tính đơn giản bằng cách sử dụng 'dt_san' làm biến đầu vào.



"""4. Xây dựng mô hình hồi quy"""
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = data[['dt_san', 'chieu_rong', 'phong_ngu', 'phong_tam', 'duong_vao']]
y = data['muc_gia']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Độ chính xác của các mô hình\n")

'''----------------'''

### Xây dựng mô hình hồi quy tuyến tính
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("Linear Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"R2_score: {round(r2_score(y_pred, y_test)*100,2)}%")
print(f"RMSE: {mean_squared_error(y_test,y_pred,squared=False)}\n")


### Xây dựng mô hình Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_dt = decision_tree_model.predict(X_test)

# Đánh giá mô hình
print("Decision Tree Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_dt)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_dt)}")
print(f"R2_score: {round(r2_score(y_test, y_pred_dt)*100,2)}%")
print(f"RMSE: {mean_squared_error(y_test,y_pred_dt,squared=False)}\n")


### Xây dựng mô hình Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_rf = random_forest_model.predict(X_test)

# Đánh giá mô hình
print("Random Forest Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_rf)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf)}")
print(f"R2_score: {round(r2_score(y_test, y_pred_rf)*100,2)}%")
print(f"RMSE: {mean_squared_error(y_test,y_pred_rf,squared=False)}\n")


### Xây dựng mô hình Neural Network Regression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Dự đoán trên tập kiểm tra
y_pred_nn = model.predict(X_test)

# Đánh giá mô hình
print("Neural Network Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_nn)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_nn)}")
print(f"R2_score: {round(r2_score(y_test, y_pred_nn)*100,2)}%")
print(f"RMSE: {mean_squared_error(y_test,y_pred_nn,squared=False)}\n")


### Xây dựng mô hình Lasso
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train,y_train)

# Dự đoán trên tập kiểm tra
y_pred_ls = lasso_model.predict(X_test)

# Đánh giá mô hình
print("Lasso Model Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_ls)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_ls)}")
print(f"R2_score: {round(r2_score(y_test, y_pred_ls)*100,2)}%")
print(f"RMSE: {mean_squared_error(y_test,y_pred_ls,squared=False)}\n")


### Xây dựng mô hình Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

gradient_boosting_model = GradientBoostingRegressor(n_estimators=100,max_depth=5)
gradient_boosting_model.fit(X_train,y_train)

# Dự đoán trên tập kiểm tra
y_pred_gb = gradient_boosting_model.predict(X_test)

# Đánh giá mô hình 
print("Gradient Boosting Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_gb)}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_gb)}")
print(f"R2_score: {round(r2_score(y_test, y_pred_gb)*100,2)}%")
print(f"RMSE: {mean_squared_error(y_test,y_pred_gb,squared=False)}\n")



### Mô hình tốt nhất 
print("Mô hình tốt nhất trong trường hợp này là Linear Regression\n")
print("So sánh giá trị của Linear Regression:")
print("Thực tế - Dự đoán")
for true_price, predicted_price in zip(y_test, y_pred):
    print(f"   {true_price:.2f} - {predicted_price:.2f}")