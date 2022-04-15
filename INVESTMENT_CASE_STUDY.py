#!/usr/bin/env python
# coding: utf-8

# # INVESTMENT CASE STUDY
# 

# Case Study Brief

# You are working for Spark Funds, an asset management company. Spark Funds wants to make investments in a few companies. The CEO of Spark Funds wants to understand the global trends in investments so that they can take the investment decisions effectively.
# 
# Spark Funds has two minor following constraints for investments:
# 
# . It wants to invest between 5 to 15 million USD per round of investment
# 
# . It wants to invest only in English-speaking countries because of the ease of communication with the companies it would invest   in.

# # Data cleaning

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[6]:


c = pd.read_excel('companies.xlsx')


# In[7]:


r = pd.read_excel('rounds2.xlsx')


# In[11]:


c.head(5)


# In[12]:


r.head()


# In[13]:


# checking the unique values
len(c.permalink.unique())


# In[14]:


c.shape


# In[15]:


len(r.company_permalink.unique())


# In[16]:


r.shape


# There are more companies_permalink than permalink, this may be case-sensetive, hence converting all the company link names to lower case.

# In[17]:


c.permalink = c.permalink.str.lower()


# In[18]:


r.company_permalink = r.company_permalink.str.lower()


# In[19]:


len(r.company_permalink.unique())


# In[20]:


len(c.permalink.unique())


# There are same number of unique companies in both the DataFrames.

# In[21]:


#checking if these two are same
c[~c.permalink.isin(r.company_permalink)]


# All the companies in c.permalink are present in r.company_permalink.

# # Missing Value Treatment

# Now we check for missing value treatment.
# 
# Check the number of missing values in both the dataframes.

# In[22]:


# checking the missing value in c and r
c.isnull().sum()


# In[23]:


r.isnull().sum()


# Since there are no missing values in company permalink, merging those two DataFrames.

# In[25]:


#renaming the company_permalink column in r to permalink
r.rename(columns = {'company_permalink':'permalink'}, inplace = True)


# In[26]:


r.head()


# In[51]:


#merging c and r DataFrames on permalink
df = pd.merge(c,r, how = 'inner', on = 'permalink')
df.head()


# In[52]:


#checking for duplicated rows
df[df.duplicated()]


# In[53]:


df.info()


# In[54]:


#checking for fraction of missing values in master DataFrame columns
df.isnull().sum()


# In[43]:


df.index


# In[55]:


round(100*(df.isnull().sum()/len(df.index)),2)


# Clearly, the column funding_round_code is useless (with about 73% missing values). Also, for the business objectives given, the columns homepage_url, founded_at, state_code, region and city need not be used.
# 
# Thus, we drop these columns.

# In[56]:


# dropping the columns
df = df.drop(['funding_round_code','homepage_url','founded_at', 'state_code', 'region', 'city'], axis = 1 )
df.head()


# In[57]:


round(100*(df.isnull().sum()/len(df.index)),2)


# Note that the column raised_amount_usd is an important column, since that is the number we want to analyse (compare, means, sum etc.). That needs to be carefully treated.
# 
# Also, the column country_code will be used for country-wise analysis, and category_list will be used to merge the dataframe with the main categories.
# 
# Checking with missing values in raised_amount_usd.

# In[58]:


df.raised_amount_usd.describe()


# The mean is somewhere around USD 10 million, while the median is only about USD 1m. The min and max values are also miles apart.
# 
# In general, since there is a huge spread in the funding amounts, it will be inappropriate to impute it with a metric such as median or mean. Also, since we have quite a large number of observations, it is wiser to just drop the rows.
# 
# We thus remove the rows having NaNs in raised_amount_usd which acts as a target variable.

# In[59]:


df = df[~np.isnan(df.raised_amount_usd)]


# In[62]:


round(100*(df.isnull().sum()/len(df.index)),2)


# In[63]:


df.country_code.value_counts()


# We see that the most number of investments have happened in USA. We can also see the fractions.

# In[66]:


100*(df.country_code.value_counts()/len(df.index))


# Now, we can either delete the rows having country_code missing (about 6% rows), or we can impute them by USA. Since the number 6 is quite small, and we have a decent amount of data, it may be better to just remove the rows.

# In[67]:


#deleating rows with NaN country code
df = df[~df.country_code.isnull()]


# In[68]:


round(100*(df.isnull().sum()/len(df.index)),2)


# Here the fraction of missing values in the remaining dataframe has also reduced now - only 0.65% in category_list. We thus remove those as well.
# 
# We could have simply let the missing values in the dataset and continued the analysis. But in this case, since we will use that column later for merging with the 'main_categories', removing the missing values will be quite convenient (and again - we have enough data).

# In[69]:


df = df[~df.category_list.isnull()]


# In[70]:


round(100*(df.isnull().sum()/len(df.index)),2)


# In[71]:


# after missing value treatment, approx 77% observations are retained
100*(len(df.index) / len(r.index))


# # Analysis

# Here we'll conduct three types of analyses - funding type, country analysis, and sector analysis.

# ## Funding Type Analysis

# Here we compare the funding amounts across the funding types. Also, we need to impose the constraint that the investment amount should be between 5 and 15 million USD. 
# We will choose the funding type such that the average investment amount falls in this range.

# In[72]:


# filtering the df so it only contains the four specified funding types
df = df[(df.funding_round_type == 'venture') |
        (df.funding_round_type == 'angel') | 
        (df.funding_round_type == 'seed') |
       (df.funding_round_type == 'private_equity')]


# In[73]:


df.shape


# Here we have to compute a representative value of the funding amount for each type of invesstment.Here we can either choose the mean or the median and having a look at the distribution of raised_amount_usd to get a sense of the distribution of data.

# In[75]:


# distribution of raised_amount_usd is shown using boxplot.
plt.figure(figsize = [14,10])
sns.boxplot(y = 'raised_amount_usd', data = df)
plt.yscale('log')
plt.show()


# In[76]:


df.raised_amount_usd.describe()


# Figuring out that there's a significant difference between the mean and the median - USD 9.5m and USD 2m.

# In[77]:


#Comparing the raised_amount_usd with the different type of funding using box-plot
plt.figure(figsize = [15,10])
sns.boxplot(x = 'funding_round_type', y = 'raised_amount_usd', data = df)
plt.yscale('log')
plt.show()


# In[79]:


df.pivot_table(values = 'raised_amount_usd', columns = 'funding_round_type', aggfunc = [np.median, np.mean])


# From the above box-plots it is indicating that there are many outliers in the raised_amount_usd across each funding type ,which takes the mean away from the medain as we can seen from the pivot table.Hence median can be the representative value of the funding amount across each funding amount.

# In[80]:


# compare the median investment amount across the types
df.groupby('funding_round_type')['raised_amount_usd'].median().sort_values(ascending=False)


# The median investment amount for type 'private_equity' is approx. USD 20M, which is beyond Spark Funds' range of 5-15M. The median of 'venture' type is about USD 5M, which is suitable for them. The average amounts of angel and seed types are lower than their range.
# 
# Thus, 'venture' type investment will be most suited to them.

# # Country Analysis

# Now we compare the total investment amounts across countries.Filtering the data only for the 'venture' type investments and then compare the 'total investment' across countries.

# In[81]:


#filtering df for only venture type
df = df[df.funding_round_type=='venture']


# In[82]:


#grouping by country codes and comparing total amount
df.groupby('country_code')['raised_amount_usd'].sum().sort_values(ascending = False)


# In[83]:


#extracting top 9 countries from the above list
df.groupby('country_code')['raised_amount_usd'].sum().sort_values(ascending = False)[0:9]


# Among the top 9 countries, USA, GBR and IND are the top three English speaking countries.Hence we filter the dataframe so that it contains only the top 3 countries.

# In[85]:


df = df[(df.country_code== 'USA') | (df.country_code == 'GBR') | (df.country_code =='IND')]


# In[86]:


# boxplot to see distributions of funding amount across countries
plt.figure(figsize = [15,10])
sns.boxplot(y = df.raised_amount_usd, x = df.country_code)
plt.yscale('log')
plt.show()


# # Sector Analysis

# First, we need to extract the main sector using the column category_list.

# In[87]:


df.category_list


# The category_list column contains values such as 'Apps|Cable|Distribution|Software' - in this, 'Apps' is the 'main category' of the company, which we need to use.
# 
# Let's extract the main categories in a new column.

# In[90]:


df['category_list'] = df['category_list'].apply(lambda x :x.split('|')[0])


# Hence we extracted only main category list.

# In[91]:


df.head()


# In[92]:


# importing mapping DataFrame
m = pd.read_csv('mapping.csv')
m


# In[94]:


#checking for NaN values
m.isnull().sum()


# In[97]:


#removing NaNs
m = m[~pd.isnull(m.category_list)]


# In[99]:


m.isnull().sum()


# In[100]:


# now before merging 'm' with 'df' let us first convert category list to lower frames to both DataFrames.
m['category_list'] = m['category_list'].str.lower()
df['category_list'] = df['category_list'].str.lower()


# In[101]:


df.head()


# In[102]:


m.head()


# In[106]:


# merging file of df and m
mdf = pd.merge(df,m, how ='inner', on = 'category_list')
mdf.head(5)


# In[109]:


#Replacing 1 with corresponding column names for columns [9 to end]
for x in mdf.columns[9::]:
    mdf[x] = mdf[x].apply(lambda y: x if y ==1 else '')
mdf.head()    


# In[110]:


#Creating new column named Sector.
mdf['Sector']=''
for x in mdf.columns[9:-1]:
    mdf['Sector']=mdf['Sector']+mdf[x]
mdf.head(5)


# In[111]:


#Dropping the columns with sector names
mdf=mdf.drop(list(mdf.columns[9:-1]),axis=1)
mdf


# The dataframe now contains only venture type investments in countries USA, IND and GBR, and we have mapped each company to one of the eight main sectors (named 'Sector' in the dataframe).
# 
# We can now compute the sector-wise number and the amount of investment in the three countries.

# In[112]:


# first, let's also filter for investment range between 5 and 15m USD

mdf  = mdf[(mdf['raised_amount_usd'] >= 5000000) & (mdf['raised_amount_usd'] <= 15000000) ]
mdf


# In[113]:


mdf.groupby(['country_code','Sector'])['raised_amount_usd'].agg(['count','sum'])


# This will be much more easy to understand using a plot.

# In[114]:


# plotting sector-wise count and sum of investments in the three countries
plt.figure(figsize=[15,10])

sns.barplot(y=mdf.Sector,x=mdf.raised_amount_usd,hue=mdf.country_code,estimator=np.sum)
plt.title('Total invested amount in USD')
plt.show()


# In[115]:


plt.figure(figsize=[15,10])
sns.countplot(y=mdf.Sector,hue=mdf.country_code)
plt.title('Total number of investements')
plt.show()


# Thus, the top country in terms of the number of investments (and the total amount invested) is the USA.
# 
# The sectors 'Others', 'Social, Finance, Analytics and Advertising' and 'Cleantech/Semiconductors' are the most heavily invested ones.

# In[ ]:




