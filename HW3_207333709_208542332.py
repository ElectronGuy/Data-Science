# Ziv Baruch 207333709
# Guy Arbiv 208542332

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

df = pd.read_csv("./csv/votersdata.csv")
RSEED = 123
pd.set_option('display.max_columns', 7)

# Q2A
# Crosstabing with vote for each categorial feature:
df_stacked_sex = pd.crosstab(df['sex'], df['vote'])
print(df_stacked_sex)
df_stacked_passtime = pd.crosstab(df['passtime'], df['vote'])
df_stacked_status = pd.crosstab(df['status'], df['vote'])
df_stacked_sex.plot.bar(stacked=True, color = ['black','red'])
df_stacked_passtime.plot.bar(stacked=True, color = ['grey','blue'])
df_stacked_status.plot.bar(stacked=True, color = ['pink','green'])
# Q2B
# Crosstabing with vote for each numerical feature:
df.boxplot(column=['volunteering'], by='vote', grid=False)
df.boxplot(column=['salary'], by='vote', grid=False)
df.boxplot(column=['age'], by='vote', grid=False)

# Q3
df.isnull().sum() # Checking nulls in the data frame.
# Sex:
le_sex = LabelEncoder() # Transforming to numerical
le_sex.fit(df['sex'])
df['new_sex'] = le_sex.transform(df['sex'])

# Age: Treating outliers:
plt.hist(df.age, bins=5) # Normal distribution.
Q1_age = df['age'].quantile(0.25)
Q3_age = df['age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
lowerThreshold_age = Q1_age - (IQR_age * 1.5)
df['age'] = df['age'].mask(df['age'] < lowerThreshold_age, np.nan)
df['age'] = df['age'].fillna(value=df.age.mean()) # All NA values turned to mean.
df.boxplot(column=['age'], grid=False)
df.isnull().sum() # Checking nulls in the data frame.

#Normalizing:
age_normalized = stats.zscore(df['age'])
df['age_z'] = age_normalized

# Salary: Treating outliers:

Q1_sal = df['salary'].quantile(0.25)
Q3_sal = df['salary'].quantile(0.75)
IQR_sal = Q3_sal - Q1_sal
upperThreshold_salary = Q3_sal + (IQR_sal * 1.5)
df['salary'] = df['salary'].mask(df['salary'] > upperThreshold_salary, np.nan)
df.boxplot(column=['salary'], grid=True)
df['salary'] = df['salary'].fillna(value=df.salary.median()) # NA salary values replaced to median.
df.boxplot(column=['salary'], grid=True)
# Normalizing:
salary_normalized = stats.zscore(df['salary'])
df['salary_z'] = salary_normalized

#Volunteering:
volunteering_normalized = stats.zscore(df['volunteering'])
df['new_volunteering'] = volunteering_normalized

#Passtime:
df['passtime'].describe() # Most are fishing, I'll change two missing values to top.
df['passtime'].value_counts() # Fishing most common
df['passtime'] = df['passtime'].replace(to_replace=np.nan, value='fishing')
# Transform to numeric:
le_passtime = LabelEncoder()
le_passtime.fit(df['passtime'])
df['new_passtime'] = le_passtime.transform(df['passtime'])
df['new_passtime'].isna().sum() # No NA's.

#Status:
le_status = LabelEncoder()
le_status.fit(df['status'])
df['new_status'] = le_status.transform(df['status'])

#Vote:
le_vote = LabelEncoder()
le_vote.fit(df['vote'])
df['new_vote'] = le_vote.transform(df['vote'])

# Reearranging data in a new dataframe (df2):
df2 = df.drop(['sex','salary','volunteering','age', 'passtime', 'status', 'vote'], axis=1)

# Q4 + 5
X = df2.drop(columns = ['new_vote'])
y = df2['new_vote']
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RSEED)
model = DecisionTreeClassifier(random_state=RSEED)
model.fit(X_train, y_train)
plt.figure(figsize=(3, 5), dpi=250)
plot_tree(model, filled=True, feature_names=X.columns, class_names= le_vote.inverse_transform(model.classes_))
plt.show()

# Q6
y_pred_test = model.predict(X_test)
cm_test = pd.crosstab(y_test, y_pred_test, colnames=['pred'], margins=True)
print(cm_test) # Confusion matrix, 1 = Repub, 0 = Democ

print("Test set results:") # Screening results:
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_test))
print("Precision:", metrics.precision_score(y_test, y_pred_test))
print("recall:", metrics.recall_score(y_test, y_pred_test))
#Q7

y_pred_train = model.predict(X_train)
cm_train = pd.crosstab(y_train, y_pred_train, colnames=['pred'], margins=True)
print (cm_train)
print("Train set results:")
print("Accuracy:", metrics.accuracy_score(y_train, y_pred_train))
print("Precision:", metrics.precision_score(y_train, y_pred_train))
print("recall:", metrics.recall_score(y_train, y_pred_train))

# Although the train set shows best stats possible it is not overfitted, the test preds are good.

#Q8

new_model = DecisionTreeClassifier(max_depth=5, min_samples_split=40, random_state=RSEED)
new_model.fit(X_train, y_train)
plt.figure(figsize=(3, 5), dpi=250)
plot_tree(new_model, filled=True,fontsize=2, feature_names=X.columns, class_names= le_vote.inverse_transform(model.classes_))
plt.show()
#Q8A The tree's depth 5
#Q8B it has 8 leaves.
#Q8C Status.
#Q8D Yes, passtime.
#Q8E They are the same :
y_pred = new_model.predict(X.loc[67:67, :])
print('Value by target:', df.loc[67:67, :]['vote'])
print('The predicted value is:', y_pred, 'While 1 = Republican and 0 = Democrat')

#Q9

y_pred_test = new_model.predict(X_test)
new_cm_test = pd.crosstab(y_test, y_pred_test, colnames=['pred'], margins=True)
print(new_cm_test) # Confusion matrix, 1 = Repub, 0 = Democ

print("New test set results:") # Screening results:
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_test))
print("Precision:", metrics.precision_score(y_test, y_pred_test))
print("recall:", metrics.recall_score(y_test, y_pred_test))

y_pred_train = new_model.predict(X_train)
new_cm_train = pd.crosstab(y_train, y_pred_train, colnames=['pred'], margins=True)
print (new_cm_train)
print("New train set results:")
print("Accuracy:", metrics.accuracy_score(y_train, y_pred_train))
print("Precision:", metrics.precision_score(y_train, y_pred_train))
print("recall:", metrics.recall_score(y_train, y_pred_train))

#Q10
# We could see that all the results are pretty high, the Test and the Train ones. They are close to each
#other as well so we can say the model is very succesful and the predictions are good.

X = df2.drop(columns = ['new_status'])
y = df2['new_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RSEED)
second_model = DecisionTreeClassifier(random_state=RSEED)
second_model.fit(X_train, y_train)
plt.figure(figsize=(3, 5), dpi=250)
plot_tree(second_model, filled=True, feature_names=X.columns, class_names= le_status.inverse_transform(second_model.classes_))
plt.show()
#Test's reults:
y_pred_test = second_model.predict(X_test)
second_cm_test = pd.crosstab(y_test, y_pred_test, colnames=['pred'], margins=True)
print(second_cm_test) # Second confusion matrix.
#Accuracy: Accuracy is 0.78
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_test))

# Train's result:
y_pred_train = second_model.predict(X_train)
second_new_cm_test = pd.crosstab(y_train, y_pred_train, colnames=['pred'], margins=True)
print(second_new_cm_test) # Second confusion matrix.
#Accuracy: Accuracy is 1
print("Accuracy:", metrics.accuracy_score(y_train, y_pred_train))

#Q10 B
# The train's accuracy is 1 as expected without limitations, and the accuacy of the test is 78%. the model won't predict
# that good. When limiting the model to 5 max depth and 40 max sample leaves we can see that it would stabalize on around
# 56% precent both test and train, which means it isn't that good..
# Im putting the model with the limitations in quotes below.

# LIMITATIONS:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RSEED)
# second_model = DecisionTreeClassifier(random_state=RSEED, min_impurity_decrease=40, max_depth=5)
# second_model.fit(X_train, y_train)
# plt.figure(figsize=(3, 5), dpi=250)
# plot_tree(second_model, filled=True, feature_names=X.columns, class_names= le_status.inverse_transform(second_model.classes_))
# plt.show()
# #Test's reults:
# y_pred_test = second_model.predict(X_test)
# second_cm_test = pd.crosstab(y_test, y_pred_test, colnames=['pred'], margins=True)
# print(second_cm_test) # Second confusion matrix.
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred_test))
#
# # Train's result:
# y_pred_train = second_model.predict(X_train)
# second_new_cm_test = pd.crosstab(y_train, y_pred_train, colnames=['pred'], margins=True)
# print(second_new_cm_test) # Second confusion matrix.
# print("Accuracy:", metrics.accuracy_score(y_train, y_pred_train))


y_test_pred = second_model.predict(X_test)
cm3 = pd.crosstab(y_test, y_test_pred, colnames=['pred'], margins=True)

print('Precision:', (cm3.iloc[2, 2]) / (cm3.iloc[3, 2]))
