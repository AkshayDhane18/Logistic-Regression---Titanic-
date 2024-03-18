#!/usr/bin/env python
# coding: utf-8

# ## Titanic Dataset

# In[1]:


import pandas as pd #data processing
import numpy as np #linear algebra

#libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Library for feature scaling
from sklearn.preprocessing import RobustScaler


#Libraries for logistic Regression 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Libraries for evaluation
from sklearn import metrics 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report


# In[3]:


#Loading Dataset
titanic_df= pd.read_csv('titanic_dataset.csv')


# In[4]:


titanic_df.columns


# In[5]:


#Looking at the first few rows in the dataset
titanic_df.head(10)


# ## Exploratory Data Analysis

# In[6]:


titanic_df.shape


# In[7]:


#Information about the dataset
titanic_df.info()


# In[8]:


titanic_df.isnull().sum()


# From above, we see that the dataset is missing a lot of information for the Cabin column. We'll need to deal with that when we go about using **Cabin** column.
#                                                                    
# Other information seems to be complete, except some **Age** entries.

# In[9]:


titanic_df.Survived.value_counts()


# In[10]:


titanic_df.Pclass.value_counts()


# In[11]:


titanic_df.Embarked.value_counts()


# In[12]:


titanic_df_numcol = titanic_df.select_dtypes(exclude=['object']).columns
titanic_df_num= titanic_df[titanic_df_numcol]


# In[13]:


plt.figure(figsize =(10,6))
sns.heatmap(titanic_df_num.corr(), annot=True,cmap = 'Blues')


# In[14]:


plt.figure(figsize =(9,5))
sns.countplot(x= 'Survived',data=titanic_df,palette='Pastel1')


# From above plot it is clear that number of persons Survived is lesss than Demise

# In[15]:


sns.catplot(x ="Survived", col ='Sex', kind ="count", data = titanic_df, palette = 'Pastel2')


# From above plot we see that Male didn't survived as compare to Female

# In[16]:


plt.figure(figsize = (9,4))
sns.catplot(data = titanic_df, x= 'Survived',col ='Pclass',kind='count',palette ='Pastel1')
plt.show()


# Above plot we can see that persons from PClass3 did not survived as compare to PClass2 and PClass1

# In[17]:


plt.figure(figsize=(9,4))
sns.set_style('whitegrid')
sns.catplot(x='Survived', col='Embarked',kind='count',data=titanic_df,palette ='Set3')


# From above plot we can see that majority of people from Embarked-S died as compare Embarked-C and Embarked-Q

# Above we have seen that number of people in PClass3 died more. So lets check is there any relation between **Embarked** and **Pclass**

# In[18]:


plt.figure(figsize=(9,4))
sns.set_style('whitegrid')
sns.catplot(x='Survived', col='Embarked',hue='Pclass',kind='count',data=titanic_df,palette ='Set2')


# Majority of people died from **Embarked-S** was in **PClass3**.

# In[19]:


sns.displot(data=titanic_df, x="Fare",hue='Survived',kind ='kde')


# Here we see that number of persons survived those who bought ticket of high fare.

# In[20]:


plt.figure(figsize = (9,4))
sns.histplot(titanic_df['Age'].dropna(),kde =True, color='blue',bins =30)
plt.show()


# In[21]:


plt.figure(figsize=(9,4))
sns.countplot(x='SibSp',data = titanic_df,palette ='Set3')


# Above we can see that most of the people does not have Sibling or Spouse.

# In[22]:


plt.figure(figsize=(12,8))
sns.countplot(x='Parch',data = titanic_df,palette ='Set1')


# Here we can clearly see that most of the people does not have children accompanying them.

# ## Data Preprocessing

# In[23]:


titanic_df.isnull().sum()


# Here we can see that Age, Cabin, Embarked have missing values.
# 
# Lets hand missing values of Age columns

# In[24]:


sns.set(rc={'figure.figsize':(8, 6)})
sns.heatmap(titanic_df.isnull(),yticklabels= False, cbar=False, cmap='copper')


# In[25]:


plt.figure(figsize =(8,6))
sns.boxplot(x= 'Pclass', y = 'Age', data= titanic_df, palette ='ocean')
plt.title('Age over Pclass')


# In[26]:


def imputing_missing_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass ==1:
            return 38
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[27]:


titanic_df['Age']=titanic_df[['Age','Pclass']].apply(imputing_missing_age,axis=1)


# In[28]:


titanic_df['Age'].isnull().sum()


# In[29]:


titanic_df['Cabin'].isnull().sum()


# In Cabin column we have 687 null values out of 891 values, count of null values is greater than 75%. So we can drop this column

# In[30]:


#Dropping Cabin column
titanic_df.drop('Cabin',axis = 1, inplace = True)


# In[31]:


titanic_df.head(2)


# Cabin column has been removed.

# In[32]:


titanic_df.isnull().sum()


# Here we see that only 2 rows of Embarked is missing, so we can drop those two rows.

# In[33]:


titanic_df.dropna(inplace=True)


# In[34]:


titanic_df.isnull().sum()


# So we have done with missing values.

# In[35]:


titanic_df.info()


# Now, there are only 4 categorical variable named : Name, Sex, Ticket, Embarked.
# 
# Features named Name and Ticket will have no significant meaning for determining target,so we can drop those two features. 
# 
# Where as Sex and Embarked features will have to be encoded for further analysis.

# In[36]:


titanic_df.drop(['Name','Ticket'],axis= 1, inplace = True)


# In[37]:


titanic_df.head()


# Now lets convert categorical variable to numeric values.

# In[38]:


#Encoding Categorical variables

titanic_df['Sex']= titanic_df['Sex'].map({'female':0, 'male':1})
titanic_df['Embarked']= titanic_df['Embarked'].map({'S':0, 'C':1, 'Q':3})


# I have done mapping for converting it to numeric as there were only two variables in Sex and 3 in Embarked 

# In[39]:


titanic_df.head()


# Here Feature named PassengerId is like index, it does not have any significance for building model and will loose its meaning as index when we do feature scaling. So it can be dropped.

# In[40]:


titanic_df.drop('PassengerId',axis= 1, inplace = True)


# In[41]:


titanic_df.head()


# In[42]:


#Feature selection 
X= titanic_df.drop(['Survived'],axis= 1)

#Target variable
y = titanic_df['Survived']


# In[43]:


X.head()


# In[44]:


y.head()


# ## Train-Test Split

# In[45]:


#Splitting the dataset into training and testing sets to evaluate the model's performance

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)


# In[46]:


print(X_train.shape)
print(X_test.shape)


# In[47]:


print(y_train.shape)
print(y_test.shape)


# ## Feature Scaling

# In[48]:


cols = X_train.columns

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train,columns=cols)
X_test = pd.DataFrame(X_test,columns=cols)


# ## Logistic Regression Model

# In[49]:


model= LogisticRegression(max_iter=100)
model.fit(X_train,y_train)


# In[50]:


y_pred = model.predict(X_test)
y_pred


# ## Evaluation

# In[51]:


#Checking accuracy 

accuracy = accuracy_score(y_test,y_pred)
print(f' Accuracy is : {accuracy:.2f}')


# In[52]:


#Confusion martix
confusion_mart= confusion_matrix(y_test,y_pred)
confusion_mart


# In[53]:


#Classfication report

report = classification_report(y_test,y_pred)
print(report)

