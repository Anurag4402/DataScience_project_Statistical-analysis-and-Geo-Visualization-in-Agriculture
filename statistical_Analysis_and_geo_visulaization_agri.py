#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pycountry


# In[2]:


conda install scikitlearn


# In[4]:


import pandas as pd


# In[6]:


df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/data-science-in-agriculture-basic-statistical-analysis-and-geo-visualisation/estat_aact_eaa01_defaultview_en.csv')


# In[7]:


df


# In[8]:


df.columns


# In[9]:


col = df.columns[6:-1]
col


# In[8]:


df = df[col]
df


# In[10]:


df.info()


# In[11]:


df.loc[:, 'geo'] = df['geo'].astype('category')
df.info()


# In[12]:


df['geo'] = df['geo'].cat.add_categories(["GB", "GR"]) #Add GB and GR in geo coloum


# In[13]:


pd.options.mode.chained_assignment = None  # swich of the warnings
mask = df['geo'] == 'UK' # Binary mask
df.loc[mask, 'geo'] = "GB" # Change the values for mask
df


# In[14]:


mask = df['geo'] == 'EL'
df.loc[mask, 'geo'] = "GR"
df


# In[15]:


import pycountry


# In[16]:


list_alpha_2 = [i.alpha_2 for i in list(pycountry.countries)]  # create a list of country codes
print("Country codes", list_alpha_2)

def country_flag(df):
    '''
    df: Series
    return: Full name of country or "Invalide code"
    '''
    if (df['geo'] in list_alpha_2):
        return pycountry.countries.get(alpha_2=df['geo']).name
    else:
        print(df['geo'])
        return 'Invalid Code'

df['country_name']=df.apply(country_flag, axis = 1)
df


# In[19]:


mask = df['country_name'] !='Invalid code'
df = df[mask]
df


# In[20]:


df.info()


# In[21]:


df.describe()


# In[22]:


df.describe(include=['category'])


# In[23]:


df['country_name'].value_counts()


# In[24]:


df['country_name'].value_counts()


# In[25]:


pt_country = pd.pivot_table(df, values= 'OBS_VALUE', index= ['TIME_PERIOD'], columns=['country_name'], aggfunc='sum', margins=True)
pt_country


# In[26]:


pt_country.describe()


# In[27]:


pt = pd.pivot_table(df, values= 'OBS_VALUE', index= ['country_name'], columns=['TIME_PERIOD'], aggfunc='sum', margins=True)
pt


# In[28]:


pt.describe()


# In[29]:


pt.iloc[-1][:-1].plot.bar()


# In[34]:


pt['All'][:-1].plot.bar(x='country_name', y='val', rot=90)


# In[35]:


pt.loc['Sweden'][:-1].plot()


# In[32]:


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(len(pt.columns)-1)  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots() # Create subplots
rects1 = ax.bar(x - width/2, pt.loc['Germany'][:-1], width, label='Germany') # parameters of bars
rects2 = ax.bar(x + width/2, pt.loc['France'][:-1], width, label='France')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('OBS_VALUE')
ax.set_xlabel('Years')
ax.set_xticks(x)
plt.xticks(rotation = 90)
ax.set_xticklabels(pt.columns[:-1])
ax.legend()

fig.tight_layout()

plt.show()


# In[36]:


import seaborn as sns #tocreate regrrssion plot
d = pd.DataFrame(pt.loc['Sweden'][:-1])
print(d)
sns.regplot(x=d.index.astype(int), y="Sweden", data=d,)


# In[38]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = np.reshape(d.index, (-1, 1)) # transform X values
y = np.reshape(d.values, (-1, 1)) # transform Y values
model.fit(X, y)


# In[39]:


X_pred= np.append(X, [2021, 2022, 2023])
X_pred = np.reshape(X_pred, (-1, 1))  ## Replace with your actual X values
# calculate trend
trend = model.predict(X_pred)# Replace with your actual y values
                             ## Replace with your actual model

plt.plot(X_pred, trend, "-", X, y, ".")


# In[45]:





# In[ ]:




