# Ziv Baruch 207333709
# Guy Arbiv 208542332
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv('./csv/clubmed_HW2.csv') # Reading the file
# Q1A

plt.xlabel('Age')
plt.ylabel('Amount')
plt.hist(df.age, color='b')
plt.show()

# Q1B

plt.hist(df.age, bins=5, color='b')
plt.hist(df.age, bins=15, color='b')

## We can see that when we split the histogram to more bins the data is more specific. We could see that when
## we used two bins (Q1B) it would be much harder to understand the data. We can see the age difference better.

# Q2
df.club_member.value_counts().plot(kind='bar', title='Membership', xlabel='Is a Member', ylabel='Count', color='grey')

# Q3 - We chose Room_Price.

df['log_roomprice'] = df['room_price'].mask(df['room_price'] < 1, 1)
df['log_roomprice'] = np.log10(df['log_roomprice'])
plt.hist(df.log_roomprice)
# The transformation made the data "normal" as possible so that the statistical analysis results from this data become more valid.
# The transformation reduces skewness from the originla data and it did help.

# Q4A

df_status_sex_bars2 = df.groupby(['sex', 'status'])[['status']].count()
df_status_sex_bars2.rename(columns={"status": "amount"}, inplace=True) # Renaming
df_status_sex_bars2 = df_status_sex_bars2.reset_index('status') # Index by status
df_status_sex_bars2 = df_status_sex_bars2.pivot(columns='status', values='amount')
df_status_sex_bars2 = df_status_sex_bars2.div(df_status_sex_bars2.sum(axis=1), axis=0) # Making the table a proportional one
df_status_sex_bars2.plot(kind='bar', stacked=True, color=['black', 'red', 'blue']) # Plotting as a stacked colored bar chart.

# Q4B (Same notes as last question)

df_status_sex_bars = df.groupby(['status', 'sex'])[['sex']].count()
df_status_sex_bars.rename(columns={"sex": "amount"}, inplace=True)
df_status_sex_bars = df_status_sex_bars.reset_index('sex')
df_status_sex_bars = df_status_sex_bars.pivot(columns='sex', values='amount')
df_status_sex_bars = df_status_sex_bars.div(df_status_sex_bars.sum(axis=1), axis=0)
df_status_sex_bars.plot(kind='bar', stacked=True, color=['grey', 'turquoise'])

# Questions:
# 1 - The highest number of men are in Couple status, the amount is not the biggest for sure because the table is a
# proportional one. Might be 70% men out of 100 against 20% men out of 2500.

# 2 - Couple
# 3 - 70%.
# 4 - 50%.

# Q4D

df_roomservice_nights_bars = df.groupby(['nights', 'roomservice'])[['roomservice']].count()
df_roomservice_nights_bars.rename(columns={"roomservice": "amount"}, inplace=True)
df_roomservice_nights_bars = df_roomservice_nights_bars.reset_index('roomservice')
df_roomservice_nights_bars = df_roomservice_nights_bars.pivot(columns='roomservice', values='amount')
df_roomservice_nights_bars = df_roomservice_nights_bars.div(df_roomservice_nights_bars.sum(axis=1), axis=0)
df_roomservice_nights_bars.plot(title='Room service orders', kind='bar', figsize=(15, 8), stacked=True, ylabel='amount',
                                color=['grey', 'turquoise', 'green', 'blue', 'red', 'black'])
# You could learn how many room services people order in opposed to the amount of nights they stay.

# Q4E

df3 = df.groupby(['sex', 'club_member'])[['club_member']].count()
print(df3)
df4 = df.groupby(['sex', 'status'])[['status']].count()
print(df4)

# There's a stronger relation to the club member category in opposed to the sex category. We could see that males tend
# to have a club member whilst females have less. When we look at the status we can see thr same numebrs more or less.

# Q5

plt.scatter(df.age,df.minibar, color='blue') # Creating the bar chart.
plt.title ('Scatter plot - Minibar and age') # Title given
plt.ylabel ('Minibar') # Giving labels
plt.xlabel ('Age')
plt.show()

#Q6A
room_price_notna = df[df[['room_price']].notna()]['room_price'] # Removing unknown elements.
print (room_price_notna.describe(include='all'))
Q1 = room_price_notna.quantile(0.25) # first quantile
Q3 = room_price_notna.quantile(0.75) # third quantile
IQR = Q3-Q1
print (IQR)

#Q6B
median = room_price_notna.median()
print (median)
room_price_notna[room_price_notna <= room_price_notna.median()].count()
# It does not because there are a lot of observations that are smaller or equal to the median.

#Q6C
plt.hist(room_price_notna, bins = 10)
mn = room_price_notna.mean() # Calculating mean
std = room_price_notna.std() # Calculating std
plt.axvline(x=mn, color='red') # mean line
plt.axvline(x=mn+std, color = 'black') # mean + std line
plt.axvline(x=mn-std, color = 'grey') # mean - std line.

#Q6D
plt.hist(room_price_notna, bins = 10)
# We could see that the distribution is close to normal, narrow with a right tail.

#Q6E
df_ranking = df.boxplot(column = ['age'], by = 'ranking', grid = False)
plt.show()
# Rank number 2
age_Q1 = df.age.quantile(0.25) # Calculating quantiles.
age_Q3 = df.age.quantile(0.75)
age_IQR = age_Q3 - age_Q1 # Calculating IQR
lowerThreshold = age_Q1 - age_IQR*1.5  # Lower threshold
upperThreshold = age_Q3 + age_IQR*1.5 # Upper threshold
plt.axhline(y=lowerThreshold, color = 'green')
plt.axhline (y=upperThreshold, color = 'black')
plt.show()

#Q6F

df_visits5years = df.boxplot(column = ['age'], by = 'visits5years', grid = True)
# Rank number 7, age 59, checked in the boxplot.

#Q6G

df_room_price_visits = df.boxplot(column = ['room_price'], by = 'visits5years', grid = True)
# We could say that they spent the least money from all the other categories.

#Q6H

df_ranking_total_expenditure = df.groupby(['ranking'])[['total_expenditure']].mean()
print (df_ranking_total_expenditure)
plt.bar(df_ranking_total_expenditure.index, df_ranking_total_expenditure.total_expenditure, color = 'blue', width = 0.4)
plt.show()
# There is no obvious tendentious relation between the rank the hosts has given and the total amount of money they've spent.

#Q7

print(df['visits2016'].describe(include = 'all'))
df_visits2016 = df['visits2016'] # Creating a new series
print(df_visits2016)
print(df_visits2016.shape)
print(df_visits2016.isna().sum())
df_visits2016 = df_visits2016.replace(to_replace=np.nan, value='Joined later') # Replacing nan to Joined later (meanwhile year 2016(, making it categorial)
df_visits2016 = df_visits2016.replace(to_replace=0, value ='Didnt visit') # Replacing 0 times to didnt visit (Categorial)
df_visits2016 = df_visits2016.mask(df['visits2016'] > 0, 'Visited', inplace=False) # Masking all visits to categorial 'Visited'
print(df_visits2016.head(60)) # Just checking myself..;)
df['2016Visits_new'] = df_visits2016

# Q8A

total_expenditure_new = df['total_expenditure']
total_expenditure_new = total_expenditure_new.mask(df['total_expenditure'] < 0, np.nan) # Removing negative numbers with nan
total_expenditure_new = total_expenditure_new.replace(to_replace=np.nan, value=df.total_expenditure.mean()) # Changing nan to mean
total_expenditure_new.describe()
exp_Q1 = total_expenditure_new.quantile(0.25) # Calculating quantiles for bins.
exp_Q2 = total_expenditure_new.quantile(0.5)
exp_Q3 = total_expenditure_new.quantile(0.75)
exp_Q4 = total_expenditure_new.quantile(1)
labels = ['Low', 'Mid','Mid-High', 'High'] # Labels making
expenditure_grouped = pd.cut(total_expenditure_new, bins=[0, exp_Q1, exp_Q2, exp_Q3, exp_Q4], labels=labels)
# Cuting by bins and labels so we would have categorial values seperated by quantiles.
print (expenditure_grouped)
df['total_expenditure_new'] = expenditure_grouped

#Q8B

df.room_price.describe()
print(df.room_price.describe())
print(df.room_price.median())
print(df.isna().sum())
df_room_price_new1 = df['room_price'].replace(to_replace=np.nan, value= df.room_price.median())
df_room_price_new2 = df['room_price'].replace(to_replace=np.nan, value= df.room_price.mean())
print(df.room_price.describe())
print(df_room_price_new1.describe())
print(df_room_price_new2.describe())

# We should correct the values by changing them to the mean
#The reason is that there is a right tail when we look at the graph, when we change the values with mean it wouldn't change.

#Q8C

tot_expend_Z = df['total_expenditure']
tot_expend_Z = tot_expend_Z.mask(df['total_expenditure'] < 0, np.nan) # Same as 8A
tot_expend_Z = tot_expend_Z.replace(to_replace=np.nan, value=df.total_expenditure.mean()) # Same as 8A
tot_expend_Z = stats.zscore(tot_expend_Z) # Normalizing the series.
bins = np.unique([-3,-2,-1,0,1,2,3]) # Creating bins
labels = ['Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5', 'Cat6'] # Creating labels.
tot_expend_grouped_Z = pd.cut(tot_expend_Z, bins=bins, labels = labels) # Cutting the series by bins and labels
nan_amount = tot_expend_grouped_Z.isna().sum()
# Counting the amount of guests that who doesn't show in the new division.
print ('The number of nan is: ' ,nan_amount)
# Two guests aren't included.

# Q9

minibar_Z = df['minibar'] # Copying the series so we wouldn't change the original.
minibar_Z = stats.zscore(minibar_Z) # Normalizing the series
print ('STD before normalization: ', df['minibar'].std())
print ('STD after normalization: ', minibar_Z.std())
minibar_Z.value_counts(bins=[-1,1]) # First option to count.
count = 0
for i in minibar_Z:
    if -1 < i < 1:
        count+=1
print (count) # Second option to count.


