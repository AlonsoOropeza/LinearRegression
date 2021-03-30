# Linear regression to predict features of human anatomy
## Abstract
Have you ever wonder if the stature of a person determines the lenght of their feet and hands? For Lord of the Rings fans, what about hobbits, they are smaller than humans, nevertheless they have large feet. Is there any relation between these variables or is it mere coincidence? Thankfully, using linear regression, we can predict a feature of interest based on its dependance of other features to finally answer if its true our hypothesis. 
## Introduction
As we implied in the abstract, our objective is to determine if there is any relation between the stature of a human and the lenght of their feet and hands.
We will be using linear regression, which is a machine learning algorithm thats tries to fit data into a linear model. But for those of you who just read that line and didn't understand what the heck I am talking about, let me rephrase it. Do you remember your higschool math clases, when you saw the equation of the are not studying 
Linear regression (LR) is a supervised machine learning algorithm that tries to fit the data you give it into a linear model. Before trying to use linear regression you should look for a linear relationship between the variables. If there's no linear relationship between them, this algorithm won't work. [1]


variables based on the stature_hand_foot dataset using linear regression. (Gradient Descent)


Dataset:  stature_hand_foot.csv


Source: S.G. Sani, E.D. Kizilkanat, N. Boyan, et al. (2005).
"Stature Estmation Based on Hand Length and Foot Length," Clinical
Anatomy, Vol. 18, pp. 589-596.


Description: Stature, hand length, and foot length among 80 males and 75 females.
Data simulated to have equal means, SDs, and correlations.


Variable Names

idGen  (w/in gender)  
gender       (1=M, 2=F)
height   (mm)  
handLen (mm)   
footLen (mm)  
