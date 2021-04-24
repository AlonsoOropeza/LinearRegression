import pandas as pd # to manage the dataset
import numpy as np # numpy is used to make some operrations with arrays more easily
from sklearn import linear_model 
from sklearn.model_selection import train_test_split # to split into train and test set
from sklearn.metrics import mean_squared_error, r2_score

# Bayesian regression allows a natural mechanism to survive insufficient data or 
# poorly distributed data by formulating linear regression using probability distributors
# rather than point estimates. The output or response ‘y’ is assumed to drawn from a 
# probability distribution rather than estimated as a single value.

df = pd.read_csv('stature_hand_foot.csv') 
df = df.rename(columns = {'idGen':'id'})

dummy = pd.get_dummies(df['gender']) 
df =  pd.concat([df, dummy], axis=1).drop('gender', axis=1) # hot encoding
df = df.reindex(columns= ['id', 'male', 'female', 'height', 'handLen', 'footLen']) 

df['height'] /= 10 # convert mm to cm
df['handLen'] /= 10
df['footLen'] /= 10

options = ['1. height', '2. handLen', '3. footLen']
print('What do you want to predict?')
for option in options:
    print(option)
op = int(input('Enter a number: '))
if op == 1:
    X = df[['male','female','handLen','footLen']]
    y = df['height']
elif op == 2:
    X = df[['male','female','height','footLen']]
    y = df['handLen']
elif op == 3:
    X = df[['male','female','height','handLen']]
    y = df['footLen']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

regr = linear_model.BayesianRidge() 
regr.fit(X_train, y_train) # train
y_pred = regr.predict(X_test) # predict

print('Coefficients: \n', regr.coef_)
print('Mean squared error: %.2f' 
      % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# import matplotlib.pyplot as plt
# plt.scatter(X_test['footLen'], y_test, color='black')
# plt.xlabel('foot')
# plt.ylabel('height')
# plt.show()

if op == 1:
	questions = ['gender','handLen','footLen']
elif op == 2:
    questions = ['gender','height','footLen']
elif op == 3:
	questions = ['gender','height','handLen']

new_sample = []
for quest in questions:
	if(quest == 'gender'):
		new_sample += [1,0] if input(quest + '[male/female] ') == 'male' else [0,1]
	else: new_sample.append(float(input(quest + '[cm] ')))

new_sample = np.array(new_sample).reshape(1, -1) # becuase it is a single sample
prediction = regr.predict(new_sample)
print(prediction)

