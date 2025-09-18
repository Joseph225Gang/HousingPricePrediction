import pandas as pd
import matplotlib.pyplot as plt


houseing_data = pd.read_csv('./housing.csv')
houseing_data.sample(5)
houseing_data = houseing_data.dropna()
print(houseing_data.shape)
houseing_data.loc[houseing_data['median_house_value'] == 500001].count
houseing_data = houseing_data.drop(houseing_data.loc[houseing_data['median_house_value'] == 500001].index) 
print(houseing_data.shape)
houseing_data['ocean_proximity'].unique() 
houseing_data = pd.get_dummies(houseing_data, columns=['ocean_proximity'])
print(houseing_data.shape)
X = houseing_data.drop('median_house_value', axis=1)
Y = houseing_data['median_house_value']
print(X.columns)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
print(x_train.shape, x_test.shape)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
linear_model  = make_pipeline(StandardScaler(), LinearRegression()).fit(x_train, y_train)
print("Training_score : ", linear_model.score(x_train, y_train))

predictors = x_train.columns
# 取出 pipeline 裡的 LinearRegression

linreg = linear_model.named_steps['linearregression']

print(pd.Series(linreg.coef_, predictors).sort_values())

y_pred = linear_model.predict(x_test)

df_pred_actual = pd.DataFrame({'predicated': y_pred, 'actual': y_test})

print(df_pred_actual.head(10))

from sklearn.metrics import r2_score
print('Testing_score :', r2_score(y_test, y_pred))

fig, ax = plt.subplots(figsize=(12,8))

plt.scatter(y_test, y_pred)
plt.show()

df_pred_actual_sample = df_pred_actual.sample(100)
df_pred_actual_sample = df_pred_actual.reset_index()


plt.figure(figsize=(20,10))

plt.plot(df_pred_actual_sample['predicated'], label='Predicated')
plt.plot(df_pred_actual_sample['actual'], label='Actual')

plt.ylabel('median_house_value')

plt.legend()
plt.show()