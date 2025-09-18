import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb 


housing_data = pd.read_csv('./housing.csv')
housing_data.sample(5)
housing_data = housing_data.dropna()
print(housing_data.shape)
housing_data.loc[housing_data['median_house_value'] == 500001].count
housing_data = housing_data.drop(housing_data.loc[housing_data['median_house_value'] == 500001].index) 
print(housing_data.shape)
housing_data['ocean_proximity'].unique() 
housing_data = pd.get_dummies(housing_data, columns=['ocean_proximity'])
print(housing_data.shape)
X = housing_data.drop('median_house_value', axis=1)
Y = housing_data['median_house_value']
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

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)

rf_y_pred = rf.predict(x_test)
print("Random Forest Training_score : ", rf.score(x_test, y_test))


rf2 = RandomForestRegressor(n_estimators=1000,
                             min_samples_split=10,
                             max_depth=14,
                             random_state=42)

rf2.fit(x_train, y_train)
print("Random Forest 2 Training_score : ", rf2.score(x_test, y_test))

xgb_reg = xgb.XGBRegressor(
    objective="reg:squarederror",  # 適合迴歸任務
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

x_train.columns = [str(col).replace("[","").replace("]","").replace("<","").replace(">","") for col in x_train.columns]
x_test.columns = [str(col).replace("[","").replace("]","").replace("<","").replace(">","") for col in x_test.columns]
xgb_reg.fit(x_train, y_train)
y_pred_xgb = xgb_reg.predict(x_test)
print("XGBoost Training_score : ", r2_score(y_test, y_pred_xgb)) 

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