#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
fish = pd.read_csv('Fish.csv')
fish.head()


# In[4]:


fish['Species'].unique()


# In[5]:


fish.isnull().sum()


# In[6]:


X = fish.iloc[:, 1:]
y = fish.loc[:, 'Species']


# In[7]:


X


# In[9]:


y


# #  Scaling the input features using MinMaxScaler

# In[11]:


print(X.head())


# In[12]:


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

# Sample DataFrame
data = pd.DataFrame({
    'Fish': ['Bream', 'Salmon', 'Bream', 'Trout'],
    'Weight': [150, 300, 170, 220]
})

# Convert categorical 'Fish' column to numeric
label_encoder = LabelEncoder()
data['Fish'] = label_encoder.fit_transform(data['Fish'])

# Separate features and target
X = data[['Fish']]  # Features
y = data['Weight']  # Target

# Apply MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

print("Scaled Features:")
print(X_scaled)


# In[15]:


scaler = MinMaxScaler()
scaler.fit(X)  
X_scaled = scaler.transform(X)  
print("Scaled Features:")
print(X_scaled)


# In[16]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# #  Label Encoding the target variable using LabelEncoder

# In[17]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y


# #  Splitting into train and test datasets using train_test_split

# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# # Model Building and Training

# In[19]:


from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
logReg.fit(X_train, y_train)


# # Predecting the output

# In[20]:


y_pred = logReg.predict(X_test)


# # Computing the accuracy

# In[21]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# # Confusion matrix

# In[22]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
cf = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cf, annot=True)
plt.xlabel('Prediction')
plt.ylabel('Target')
plt.title('Confusion Matrix')


# # SVC apple_oranges

# In[29]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[30]:


data = pd.read_csv('apples_and_oranges.csv')
print(data)


# In[31]:


import seaborn as sns
sns.scatterplot(x="Weight", y="Size", hue="Class", data=data)


# In[32]:


training_set,test_set = train_test_split(data,test_size=0.2,random_state=1)
print("train:",training_set)
print("test:",test_set)


# In[33]:


x_train = training_set.iloc[:,0:2].values  # data
y_train = training_set.iloc[:,2].values  # target
x_test = test_set.iloc[:,0:2].values  # data
y_test = test_set.iloc[:,2].values  # target
print(x_train,y_train)
print(x_test,y_test)


# In[34]:


classifier = SVC(kernel='rbf',random_state=1,C=1,gamma='auto')
classifier.fit(x_train,y_train)


# In[35]:


y_pred = classifier.predict(x_test)
print(y_pred)


# In[36]:


cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy = float(cm.diagonal().sum())/len(y_test)
print('model accuracy is:',accuracy*100,'%')


# In[ ]:




