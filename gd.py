import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

df = pd.read_csv('stature_hand_foot.csv') 
df = df.rename(columns = {'idGen':'id'})

dummy = pd.get_dummies(df['gender']) 
df =  pd.concat([df, dummy], axis=1).drop('gender', axis=1) # hot encoding
df = df.reindex(columns= ['id', 'male', 'female', 'height', 'handLen', 'footLen']) 

df['height'] /= 10 # convert mm to cm
df['handLen'] /= 10
df['footLen'] /= 10

plt.scatter(df['handLen'],df['footLen'])
plt.xlabel('handLen (cm)')
plt.ylabel('footLen (cm)')
plt.show()