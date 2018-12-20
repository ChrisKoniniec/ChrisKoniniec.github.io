---
layout: post
title: Predicting Utility Site Disruptions
nocomments: true
categories: [python, classification, machine learning]
---

Anticipating Utility Site Disruptions
====================

Our business question in this situation is that a Utility company wants to predict their site disruptions in order to minimize the amount of time that their service is down. My boss has come to me and asked me to generate a report every morning in order to preemptively predict these disruptions so that she can send out service technicians to their first location in the morning.

Each service station has a fault severity of 0 (no issues), 1 (minor issues), or 2 (major issue). The other data that are given to us are fairly abstract in nature, but we know each observation is a single instance of a service check on the station.


'''
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
'''

After importing the libraries, we read in the data, which is given to us in 4 different tables. After taking a look at each one, we clean them up enough to combine together

'''
df_list = [event_tb, log_tb, resource_tb, severity_tb, train_tb]

for item in df_list:
    print(item.head())
'''


'''
#Merge the dataframes together
df = reduce((lambda df1, df2: pd.merge(df1, df2, on='id')), df_list)
'''

After taking a look at each variable using value counts, we see that each feature is fairly abstract and it's difficult to see at face value what each feature is.

Since log_feature and volume came in the same dataframe, we'll look at how they relate to one another:

![Graph1](/assets/Project1/Proj1Graph1.png)

![Graph2](/assets/Project1/Proj1Graph2.png)

I'm going to take the log of volume since it's a bit more normalized.

Let's continue to look into the relationships surrounding log_feature

This boxplot will show us how the volumes vary depending on fault severity:

![Graph3](/assets/Project1/Proj1Graph3.png)


Since these features are fairly abstract, lets take a look at some more graphs to get a sense of how the variables are related to one another.

![Graph4](/assets/Project1/Proj1Graph4.png)

This graph shows the frequency of fault severity per resource type and severity type.

We can see that resource type 5 only has 2, or 'critical' fault severity. We can also see that resource types 2, 3, and 9 have higher chance of 0 fault severity.

![Graph5](/assets/Project1/Proj1Graph5.png)

We can see some relationships within each event type or resource type, but no real overarching trends. This is why we have models to pick up on the subtler relationships between variables.

Lets prepare our data for the model by getting dummies on our object columns. After that, we will group our data by ID number, since we want to look at the fault severity at each ID.


'''
data = data.groupby('id', sort = False).sum()
'''

We separate out our features and target, and run a train_test_split to randomly sample out our data and decide to use the gradient boosting classifier to predict.



'''
#The Gradient Boosting Classifier fits our data the best
gbc_model = gbc.fit(x_train, y_train)

gbc_accuracy = accuracy_score(y_train, gbc.predict(x_train))
print(gbc_accuracy)
'''

'''
#setting up prediction probabilities for each row
y_pred_proba = gbc.predict_proba(x_test)
y_pred = gbc.predict(x_test)
'''

After predicting the probability, we want to create an output that is easily readable by our supervisor so that they can send the service techs to the correct locations.

'''
#Make a DataFrame of the predictions
result = pd.DataFrame({
    "id": x_test.index,
    "Predicted fault_severity": y_pred,
    "prediction_probability_0": y_pred_proba[:, 0],
    "prediction_probability_1": y_pred_proba[:, 1],
    "prediction_probability_2": y_pred_proba[:, 2]
}, columns =['id', 'Predicted fault_severity', "prediction_probability_0",
            "prediction_probability_1", "prediction_probability_2"])

result.head(10)
'''

Next, we sort the values by most likely to have a severe fault.

'''
result.sort_values(['prediction_probability_2'], ascending= False, inplace= True)
result.head(20)
'''

After we get this table, all we need to do is write it to a csv and send it to our boss so the service techs can do their thing!


Our predictions with the basic model are decent, but as you can see in the confusion matrix below, there is only a 58% chance that we will predict a fault severity of 2 correctly.

After some hyperparameter tuning, we can increase this to 63% chance that a 2 is predicted correctly, and nearly 80% chance there is at least a minor issue with the site. Given that the assignment is to give insight into a technician's first assignment of the day, 63% is a considerable increase to chance for very low risk.


![Graph6](/assets/Project1/ConfMatrix1.png)

![Graph7](/assets/Project1/ConfMatrix2.png)

Graphs made by using matplotlib, seaborn, and yellowbrick.

Link to Jupyter notebook may be coming soon here, or can be sent upon request.
