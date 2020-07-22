#------------------------------------------------------------------------------------------------------------#
### 1.0 LOAD LIBRARY'S ----
import os, sys
import pandas as pd
import numpy as np
import dateutil


# Workflow
# 1.Create a 4 column DataFrame with 10 rows, the first column being a date 
# field and the rest numbers.
# 2.Fill the first column with the first day of each month for 3 years 
# (for example: 1/1/2018, 2/1/2018).
# 3.Fill the next 2 columns with random numbers.
# 4.Fill the 4th column with the difference of the first 2 data columns 
# (for example: Col3 - Col2).
# 5.Break the DataFrame into 3 different DataFrames based on the dates 
# (for example: 2018, 2019, 2020)


### dates dataframe
df_data1 = pd.DataFrame({'date': ['2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01'
                                ,'2019-01-01', '2019-02-01', '2019-03-01'
                                ,'2020-01-01', '2020-02-01', '2020-03-01']
                        })
### random numbers dataframe
num = np.random.randint(5, 30, size=(10, 2))
df_data2 = pd.DataFrame(num, columns=['random_numbers_1', 'random_numbers_2'])

### merged dataframe
df_data = pd.concat([df_data1.reset_index(drop=True), df_data2], axis=1)

### difference column
df_data['difference'] = df_data['random_numbers_2'] - df_data['random_numbers_1']

### add column to date type
df_data["date"] = df_data["date"].apply(lambda x: dateutil.parser.parse(x))

### split by year   
for m in df_data["date"].dt.to_period("Y").unique():
    temp = 'df_data_{}'.format(m)    
    vars()[temp] = df_data[df_data["date"].dt.to_period("Y")==m]   