import pandas as pd
import matplotlib.pyplot as plt

houseing_data = pd.read_csv('./housing.csv')
houseing_data.sample(5)
houseing_data = houseing_data.dropna()
houseing_data.loc[houseing_data['median_house_value'] == 500001].count
houseing_data = houseing_data.drop(houseing_data.loc[houseing_data['median_house_value'] == 500001].index) 
houseing_data['ocean_proximity'].unique() 
houseing_data = pd.get_dummies(houseing_data, columns=['ocean_proximity'])

median = houseing_data['median_house_value'].median()

houseing_data['above_median'] = (houseing_data['median_house_value'] - median) > 0

X = houseing_data.drop(['median_house_value', 'above_median'], axis=1)
Y = houseing_data['above_median']

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