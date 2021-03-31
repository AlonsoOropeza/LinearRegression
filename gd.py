import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('stature_hand_foot.csv') 
df = df.rename(columns = {'idGen':'id'})

dummy = pd.get_dummies(df['gender']) 
df =  pd.concat([df, dummy], axis=1).drop('gender', axis=1) # hot encoding
df = df.reindex(columns= ['id', 'male', 'female', 'height', 'handLen', 'footLen']) 

df['height'] /= 10 # convert mm to cm
df['handLen'] /= 10
df['footLen'] /= 10

# plt.scatter(df['handLen'],df['footLen'])
# plt.xlabel('handLen (cm)')
# plt.ylabel('footLen (cm)')
# plt.show()

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

def h(params, sample):
	acum = 0
	for i in range(len(params)):
		acum = acum + params[i]*sample[i]  #evaluates h(x) = a+bx1+cx2+ ... nxn.. 
	return acum