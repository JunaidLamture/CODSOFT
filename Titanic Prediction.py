# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 11:34:23 2024

@author: S&J
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
df= pd.read_csv("C:\\Datascience\\Titanic-Dataset.csv")
df.head()
df.shape
df.tail()
df.describe()
df.duplicated().sum()
Survived = df[['Survived']].value_counts().reset_index()
data = {'Survived': ['Male - No', 'Male - Yes', 'Female - No', 'Female - Yes'],
        'Counts': [100, 50, 30, 80]}  # replace with actual counts
Survived = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
plt.bar(Survived['Survived'], Survived['Counts'],color=["Black","blue","red","pink"])
plt.xticks(Survived['Survived'])
plt.title('Comparison of Survival')
plt.xlabel('Gender and Survival Status')
plt.ylabel('Number of People')
plt.show()

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()
inputs = df.drop('Survived',axis='columns')
target = df['Survived']
sex=pd.get_dummies(inputs.Sex)
sex.head()
inputs=pd.concat([inputs,sex],axis="columns")
inputs.head()
inputs.drop(["Sex"],axis="columns",inplace=True)
inputs.head()
inputs.isna().sum()
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head()
inputs.info()
inputs.isna().sum()
counts = df.groupby(['Survived', 'Sex']).size().unstack().fillna(0)

# Define the bar width
bar_width = 0.30
index = counts.index

# Plotting
fig, ax = plt.subplots()

# Plot bars for each Sex
bar1 = ax.bar(index - bar_width/2, counts['male'], bar_width, label='male')
bar2 = ax.bar(index + bar_width/2, counts['female'], bar_width, label='female')

# Setting labels and title
ax.set_xlabel('Survived')
ax.set_ylabel('Count')
ax.set_title('Survival Counts by Gender')
ax.set_xticks(index)
ax.set_xticklabels(['Not Survived', 'Survived'])
ax.legend()

# Display the plot
plt.show()
X_train, X_test, y_train, y_test=train_test_split(inputs,target,test_size=0.2)
X_train
X_test
y_train
y_test
inputs.corr()
sns.heatmap(inputs.corr(), annot=True, cmap='coolwarm', fmt=".2f")
model=RandomForestClassifier()
model.fit(X_train,y_train)
model.score(X_test,y_test)
pre=model.predict(X_test)
matrices=r2_score(pre,y_test)
matrices