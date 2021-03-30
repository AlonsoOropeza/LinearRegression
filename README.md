# Predicting features of human anatomy using linear regression
## Abstract
Have you ever wonder if the stature of a person determines the lenght of their feet and hands? For Lord of the Rings fans, what about hobbits, they are smaller than humans, nevertheless they have large feet. Is there any relation between these variables or is it mere coincidence? Thankfully, using linear regression, we can make a prediction based on its dependance of other features to finally answer if its true our hypothesis or not. 

## Introduction
As we implied in the abstract, our objective is to determine if there is any relation between the stature of a human and the lenght of their feet and hands.  
We will be using linear regression, which is a machine learning algorithm thats tries to fit data into a linear model. 
![linear-model](https://raw.githubusercontent.com/AlonsoOropeza/LinearRegression/main/linear-model.png)  
But for those of you who just read that line and didn't understand what the heck I am talking about, let me rephrase it.  
Do you remember your highschool math clases, when you saw the equation of a line?  
The famous **y = mx + b**  
Where y is the output, m is the slope of the line, x is the input and b is the intercept.  
Well this is similar, but with minor modifications. Now we can have multiple features, that means multiple pairs of slopes and x's.    
In other words: **prediction = m1x1 + m2x2 + m3x3 + ... + mnxn + bias**    
As you saw, b is now the **bias**, which is the difference between our actual and predicted values. The model is also affected by how "noisy" the data is, the so called **variance** is the model’s sensitivity to fluctuations in the data. Analyize the image below for further explanation.  
![bias-variance](https://raw.githubusercontent.com/AlonsoOropeza/LinearRegression/main/bias-variance.png)  

## Materials and Methods
### Dependencies
In order to make our prediction we have to determine the value of each slope, we can do this using an efficient implementation of linear regression: *gradient descent*.  
Gradient descent update the parameters (slopes) by calculating over and over its values until the predicted value is the same as the real value, the error is less than the acceptance value (alpha) or the number or runs (epochs) reach a limit. In a nutshell, gradient descent does big steps when far way, and does baby steps when close to the optimal value.
![bias-gradient-descent](https://raw.githubusercontent.com/AlonsoOropeza/LinearRegression/main/gradient-descent.png)  
Theta means 

In order to calculate our error, in each epoch we will be using the **mean squared error**
![mean-squared-error](https://raw.githubusercontent.com/AlonsoOropeza/LinearRegression/main/mean-squared-error.png)  


Finally the stature_hand_foot.csv **dataset** has the following variables:  
idGen  (w/in gender)  
gender       (1=M, 2=F)
height   (mm)  
handLen (mm)   
footLen (mm)  
We make a bit of preprocessing before we train the model with the dataframe. 
### How to run it

## Results
## Discussion
## Limitations
Stature, hand length, and foot length among 80 males and 75 females.
## Acknowledgments
## References
S.G. Sani, E.D. Kizilkanat, N. Boyan, et al. (2005).
"Stature Estmation Based on Hand Length and Foot Length," Clinical
Anatomy, Vol. 18, pp. 589-596.
