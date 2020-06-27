'''Coronary Heart Diseases prediction using Artificial Neural Network'''

#importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

#importing the Dataset
dataset = pd.read_csv('heart.csv') #Source of dataset : Kaggle
dataset = dataset.drop('education', axis = 1)#dropping the column as it is not helpful in determining the outcome

#Feature Scaling
from sklearn.preprocessing import StandardScaler #there was a large amount of variance among the values
sc_X=StandardScaler()
col_to_scale = ['cigsPerDay']
dataset[col_to_scale]=sc_X.fit_transform(dataset[col_to_scale])

#Dividing the dataset into dependent and independent variables
X = dataset.iloc[: , :-1].values #Selecting everything except the last column
y = dataset.iloc[: , -1].values#Selecting only the last column

#Taking care of the missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')#replacing the missing values by the mean of the columns
imputer=imputer.fit(X[:, [4,13,11,8,3,12]])
X[:, [4,13,11,8,3,12]] = imputer.transform(X[:, [4,13,11,8,3,12]])

#Splitting the dataset into the Training and Test sets
from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) #splitting the dataset into 80-20 ratio

#Building the ANN
ann = tf.keras.models.Sequential() #buiding the ann by 3 layers
ann.add(tf.keras.layers.Dense(units=15, activation='relu'))
ann.add(tf.keras.layers.Dense(units=15, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1 , activation='sigmoid'))

#Applying Callbacks
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor ="val_loss",  mode ="min" , verbose=1 , patience=10) #to prevent the overfitting of the model

#Training the ANN
ann.compile(optimizer='adam' , loss='binary_crossentropy' ,metrics= ['accuracy']) #training and compiling the ann
ann.fit(X_train,Y_train,batch_size=32,epochs=100,validation_data=(x_test,y_test),callbacks = [es])

#Predicting the Test result
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5) #just to remove the probability

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
ac = accuracy_score(y_test,y_pred)
print("Confusion Matrix :")
print(cm)
print("Accuracy Percentage :", ac*100 ,'%')

#Single Prediction and user interface
print("*"*30,"CHD Prediction","*"*30)
print("Hello there , now I'm gonna ask you some questions about your behavioural and medical history .")
print("-*"*20)
print("Please give correct answers !")
print("-*"*20)
sex = int(input("Enter your sex : 1 for male, 0 for female : "))
age = int(input("Enter your age : "))
smoker = int(input("Do You Smoke ? 1 for yes, 0 for no : "))
cigs = int(input("If you smoke , on an avg how many cigs per day ?  0 if a NonSmoker : "))
bpmeds = int(input("Are you currently on BP Medications: 1 for yes, 0 for no : "))
stroke = int(input("In the recent past , did you have any stroke: 1 for yes, 0 for no : "))
hyptns = int(input("Do You have issues like hypertension: 1 for yes, 0 for no : "))
diab = int(input("Do you suffer from Diabetes: 1 for yes, 0 for no : "))
print("-*"*20)
print("Now I am about to ask some questions regarding your current medical condition .")
print("-*"*20)
chol = int(input("Enter your total cholestrol level : "))
sysBP =int(input("Enter your Systolic BP : "))
diaBP = int(input("Enter your Diastolic BP : "))
print("-*"*20)
print("Just a few more !!")
print("-*"*20)
bmi = int(input("Enter your BMI : "))
heartrate = int(input("Enter your Heart Rate : "))
print("-*"*20)
print("And the last one !")
print("-*"*20)
gluc = int(input("Enter your Glucose level: "))
print("-*"*20)
print("Hold on ...processing !")
print("-*"*20)
prediction = ann.predict(([[sex,age,smoker,cigs,bpmeds,stroke,hyptns,diab,chol,sysBP,diaBP,bmi,heartrate,gluc]])>0.5)
#predicting the value
if(prediction==1):
    print("You HAVE a RISK of having a Coronary Heart Disease in the upcoming TEN years.")
    print("Please take care of yourself , and improve your lifestyle !")
    print("We wish you a good health in the upcoming future !")
else:
    print("You are SAFE from any heart diseases for the upcoming TEN years.")
    print("We wish you a good health in your future !")