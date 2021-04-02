import pandas as pd # to manage the dataset
import numpy as np # numpy is used to make some operrations with arrays more easily

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

from sklearn.model_selection import train_test_split # to split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

__errors__= []

def h(params, sample):
	acum = 0
	for i in range(len(params)):
		acum = acum + params[i]*sample[i]  #evaluates h(x) = a+bx1+cx2+ ... nxn.. 
	return acum

def show_errors(params, samples,y):
	global __errors__
	error_acum =0
	for i in range(len(samples)):
		hyp = h(params,samples[i])
		#print( "hyp  %f  y %f " % (hyp,  y[i]))   
		error=hyp-y[i]
		error_acum=+error**2 # this error is the original cost function, (the one used to make updates in GD is the derivated verssion of this formula)
	mean_error_param=error_acum/len(samples)
	__errors__.append(mean_error_param)

def GD(params, samples, y, alfa):
	temp = list(params)
	general_error=0
	for j in range(len(params)):
		acum =0; error_acum=0
		for i in range(len(samples)):
			error = h(params,samples[i]) - y[i]
			acum = acum + error*samples[i][j]  #Sumatory part of the Gradient Descent formula for linear Regression.
		temp[j] = params[j] - alfa*(1/len(samples))*acum  #Subtraction of original parameter value with learning rate included.
	return temp

def scaling(samples):
	acum =0
	samples = np.asarray(samples).T.tolist() 
	for i in range(1,len(samples)):	
		for j in range(len(samples[i])):
			acum=+ samples[i][j]
		avg = acum/(len(samples[i]))
		max_val = max(samples[i])
		#print("avg %f" % avg)
		#print(max_val)
		for j in range(len(samples[i])):
			#print(samples[i][j])
			samples[i][j] = (samples[i][j] - avg)/max_val  #Mean scaling
	return np.asarray(samples).T.tolist()

params = [0,0,0,0]
samples = X_train.values[1:].tolist()
y = y_train[1:].tolist()

alfa =.01  #  learning rate
for i in range(len(samples)):
	if isinstance(samples[i], list):
		samples[i]=  [1]+samples[i]
	else:
		samples[i]=  [1,samples[i]]
#print ("original samples:")
#print (samples)
samples = scaling(samples)
#print ("scaled samples:")
#print (samples)


epochs = 0

while True:  #  run gradient descent until local minima is reached
	oldparams = list(params)
	#print (params)
	params=GD(params, samples,y,alfa)	
	show_errors(params, samples, y)  #only used to show errors, it is not used in calculation
	#print (params)
	epochs = epochs + 1
	if(oldparams == params or epochs == 2000):   #  local minima is found when there is no further improvement
		#print ("samples:")
		#print(samples)
		print ("final params:")
		print (params)
		break

# import matplotlib.pyplot as plt  #use this to generate a graph of the errors/loss so we can see whats going on (diagnostics)
# plt.plot(__errors__)
# plt.show()

test_samples = X_test.values.tolist()
for i in range(len(test_samples)):
	if isinstance(test_samples[i], list):
		test_samples[i]=  [1]+test_samples[i]
	else:
		test_samples[i]=  [1,test_samples[i]]
test_samples = scaling(test_samples) # obtaining test samples
y_pred = []
for i in range(len(test_samples)):
	y_pred.append(h(params,test_samples[i])) # make predictions

y_test = y_test.tolist() # obtaining test results
from sklearn.metrics import mean_squared_error, r2_score
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
		new_sample += [1,0] if input(quest + ' ') == 'male' else [0,1]
	else: new_sample.append(float(input(quest + ' ')))

test_samples = X_test.values.tolist()
test_samples.append(new_sample) # we add the new sample
for i in range(len(test_samples)):
	if isinstance(test_samples[i], list):
		test_samples[i]=  [1]+test_samples[i]
	else:
		test_samples[i]=  [1,test_samples[i]]
test_samples = scaling(test_samples)  
prediction = h(params, test_samples[-1]) # make prediction
print(prediction)