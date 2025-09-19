import pandas as pd
import matplotlib.pyplot as plt

housing_data = pd.read_csv('./housing.csv')
housing_data.sample(5)
housing_data = housing_data.dropna()
housing_data.loc[housing_data['median_house_value'] == 500001].count
housing_data = housing_data.drop(housing_data.loc[housing_data['median_house_value'] == 500001].index) 
housing_data['ocean_proximity'].unique() 
housing_data = pd.get_dummies(housing_data, columns=['ocean_proximity'])

median = housing_data['median_house_value'].median()

housing_data['above_median'] = (housing_data['median_house_value'] - median) > 0

X = housing_data.drop(['median_house_value', 'above_median'], axis=1)
Y = housing_data['above_median']

print(X.columns)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
Logistic_model = LogisticRegression(solver='liblinear').fit(x_train, y_train)

print("Training_score : ", Logistic_model.score(x_train, y_train))

y_pred = Logistic_model.predict(x_test)

print(pd.DataFrame({'predicated': y_pred, 'actual': y_test}))


from sklearn.metrics import accuracy_score
print('Testing_score : ', accuracy_score(y_test, y_pred))

# 新增代碼：使用XGBoost進行分類並比較
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 訓練XGBoost分類模型
x_train.columns = [str(col).replace("[","").replace("]","").replace("<","").replace(">","") for col in x_train.columns]
x_test.columns = [str(col).replace("[","").replace("]","").replace("<","").replace(">","") for col in x_test.columns]
xgb_clf = XGBClassifier(random_state=42, use_label_encoder=False)
xgb_clf.fit(x_train, y_train)
y_pred_xgb = xgb_clf.predict(x_test)
print('XGBoost Testing Score (Accuracy):', accuracy_score(y_test, y_pred_xgb))

# 比較邏輯迴歸和XGBoost的性能
print("\n--- Logistic Regression Classification Report ---")
print(classification_report(y_test, y_pred))
print("\n--- XGBoost Classification Report ---")
print(classification_report(y_test, y_pred_xgb))

# 比較特徵重要性
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# 邏輯迴歸係數（取絕對值）