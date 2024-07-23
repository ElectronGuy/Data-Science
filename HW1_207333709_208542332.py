# Ziv Baruch 207333709
# Guy Arbiv 208542332

import pandas as pd
import numpy as np

# Q1a
v1 = np.arange(1,100,10) # Creating a vector (array) with arange function
print(v1)
#Q1b
v2 = np.arange(1,100,100/8)  # Creating a vector (array) with arange function evenly spaced 100/8
#Q1c
print (v2.reshape(4,2)) # Reshaping v2 into a matrix.

#Q2a
df = pd.read_csv('./csv/customerData.csv') # Reading CSV file.
print (df.describe)
print (df.shape)
#Q2b
# Numerical - custid, income, num_vehicles, age
# Categorial - sex, is_emplyoyed, martial_stat, health_ins, housing_type, recent_move, state_of_res

#Q2c
# custid cannot help us with studying new information because it doesn't reflect anything. It's a generic number
# given to customers to define them, just like people's ID, it's unique and doesn't contain any information.

#Q3
new = df.iloc[0::10, 0::2]
print(new)

#Q4a
print (df.shape)
#Q4b
print (df.size)
#Q4c
#The difference is that the first one splits the columns and the rows and the second one is the result of rows
# multiplied by the columns. I could calculate the size (11000) with shape (1000,11) by multiplying them with one another.
# tup = df.shape
# list1 = []
# for a in tup:
#     list1.append(a)
# print (list1[0]*list1[1])

#Q5
df_age = df.loc[(df['age'] >= 38) & (df['age'] <= 50)] # loc age range between 38 and 50.
print (df_age)

#Q6a
df_numerical_columns = df.loc[(df['age'] >= 50), ('custid', 'income', 'num_vehicles', 'age')]
print (df_numerical_columns) # locing for age over 50 and columns (by name).
#Q6b
df_new_columns = df[df['age'] > 50] # Creating a new df for age over 50
df_new_columns = df_new_columns.iloc[:,[0,3,8,9]] # ilocing the new df for the wanted columns (numerical)
print (df_new_columns)

#Q7
df_first_head = df_new_columns.loc[:, ['age']].head(100)
print (df_first_head)
if isinstance(df_first_head, pd.DataFrame): # Checking if it is indeed data frame type.
    print (2)
else:
    print (3)

#Q8
print (df.loc[((df['marital_stat'] == 'Divorced/Separated') | (df['marital_stat'] == 'Married')) & (df['age'] < 18)]['custid'])

#Q9a
print('mean: ',df.loc[(df['income'] > 16000) & (df['state_of_res'] == 'Washington')]['age'].mean())
#Q9b
print('Max: ',df.loc[(df['income'] > 16000) & (df['state_of_res'] == 'Washington')]['age'].max())
#Q9c
print('Min income: ',df.loc[(df['income'] > 16000) & (df['state_of_res'] == 'Washington')]['income'].min())
#Q9d
print(len(df[(df['income'] > 16000) & (df['state_of_res'] == 'Washington')]))

#Q10a
df.groupby(['sex'])['housing_type'].describe()
# Most frequent female housing type is Rented.(200)
#Q10b
# Most frequent male housing type is Homeowner with mortage/loan(256).