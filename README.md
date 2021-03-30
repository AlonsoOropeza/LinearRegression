# Predicting features of human anatomy using linear regression
## Abstract
Have you ever wonder if the stature of a person determines the lenght of their feet and hands? For Lord of the Rings fans, what about hobbits, they are smaller than humans, nevertheless they have large feet. Is there any relation between these variables or is it mere coincidence? Thankfully, using linear regression, we can make a prediction based on its dependance of other features to finally answer if its true our hypothesis or not. 

## Introduction
As we implied in the abstract, our objective is to determine if there is any relation between the stature of a human and the lenght of their feet and hands.  
We will be using linear regression, which is a machine learning algorithm thats tries to fit data into a linear model. But for those of you who just read that line and didn't understand what the heck I am talking about, let me rephrase it.  
Do you remember your highschool math clases, when you saw the equation of a line?  
The famous y = mx + b  
Where y is the output, m is the slope of the line, x is the input and b is the intercept? 
Well this is similar, but with minor modifications. Now we can have multiple features, that means multiple pairs of slopes and x's. In other words:  
prediction = m1x1 + m2x2 + m3x3 + ... + mnxn + bias  
As you saw, b is now the bias, which is the difference between our actual and predicted values. The model is also affected by how "noisy" the data is, the so called variance is the model’s sensitivity to fluctuations in the data. See image-1.1 for further explanation.  
![image_1.1](http://url/to/img.png)

## Materials and Methods
Dataset:  stature_hand_foot.csv

Variable Names

idGen  (w/in gender)  
gender       (1=M, 2=F)
height   (mm)  
handLen (mm)   
footLen (mm)  
## Results
## Discussion
## Limitations
Stature, hand length, and foot length among 80 males and 75 females.
## Acknowledgments
## References
S.G. Sani, E.D. Kizilkanat, N. Boyan, et al. (2005).
"Stature Estmation Based on Hand Length and Foot Length," Clinical
Anatomy, Vol. 18, pp. 589-596.