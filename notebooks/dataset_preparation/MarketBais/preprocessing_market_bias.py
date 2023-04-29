import pandas as pd
import csv
import numpy as np
import time
from datetime import datetime

df= pd.read_csv('C:/Users/adity/Desktop/SJSU_work/Sem2/ADM/PROJECT/group-recommender-master_adi/drgr/df_electronics.csv')
df=df.drop(['model_attr','category','brand','year','user_attr','split'],axis=1)
df['rating']=df['rating'].astype(int)
df = df[['user_id', 'item_id','rating','timestamp']]
#Convert timestamp in seconds
for i in range(len(df['timestamp'])):
  date_string = df['timestamp'][i]
  date_obj = datetime.strptime(date_string, '%Y-%m-%d')
  timestamp = int(date_obj.timestamp())
  df['timestamp'][i]=timestamp

df_20k = df[1:20001]
# store in csv
df_20k.to_csv('market_preprocessing_20k.csv')
df_20k['user_id'].to_csv('item_data.csv', index=False, header=True)
df_20k['user_id'].to_csv('users_data.csv', index=False, header=True)

# store in .dat
output_file = 'market_preprocessing_20k.dat'

# Open the output file for writing
with open(output_file, 'w') as f:
    # Iterate over each row of the dataframe
    for index, row in df_20k.iterrows():
        # Write the data to the output file in the desired format
        f.write('{}::{}::{}::{}\n'.format(row['user_id'], row['item_id'], row['rating'], row['timestamp']))
print("done")