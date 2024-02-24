import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = '/Users/geoffreyfadera/Documents/aai_500_group_project/VM.csv'

file_path = '/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/VM.csv'

filtered_file_path = file_path.split('.')[0] + '_filtered.csv'
df = pd.read_csv(file_path)

# Delete the column
df.drop('Time', axis=1, inplace=True)

df.to_csv(file_path, index=False)

df = pd.read_csv(file_path)

# # Sample data
data = df['Execution Time']

# Calculate Q1, Q3, and IQR
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - data

# Determine bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"Outliers: {outliers}")

filtered_df = data[(data >= lower_bound) | (data <= upper_bound)]

# # Save the filtered DataFrame back to a new CSV file, without the unwanted rows
filtered_df.to_csv(filtered_file_path, index=False)


def clean_df(file_path):
   filtered_file_path = file_path.split('.')[0] + '_filtered.csv'
   df = pd.read_csv(file_path)

   # Delete the column
   df.drop('Time', axis=1, inplace=True)

   #df.to_csv(file_path, index=False) 
   #no need to modify the input file so anyone can re-run the same script over and over

   #df = pd.read_csv(file_path) #no need to re-read the input file.
   #no need to modify the input file so anyone can re-run the same script over and over

   # # Sample data
   data = df['Execution Time']
   
   print("original data max: ", np.max(data))
   print("original data min: ", np.min(data))

   # Calculate Q1, Q3, and IQR
   Q1 = np.percentile(data, 25)
   Q3 = np.percentile(data, 75)
   IQR = Q3 - Q1 #correction
   
   print("Q1, Q3, IQR : ", Q1, Q3, IQR)

   # Determine bounds for outliers
   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR
   
   print('lower bound: ', lower_bound)
   print('upper bound: ', upper_bound)

   # Identify outliers
   #outliers = data[(data < lower_bound) & (data > upper_bound)] #correction
   #print(f"Outliers: {outliers}")
   
   filtered_df = data[data >= lower_bound]
   filtered_df = filtered_df[filtered_df <= upper_bound]

   #filtered_df = data[(data >= lower_bound) | (data <= upper_bound)]
   #this is not giving the desired filtered data
   
   print("filtered data max: ", np.max(filtered_df))
   print("filtered data min: ", np.min(filtered_df))

   # # Save the filtered DataFrame back to a new CSV file, without the unwanted rows
   filtered_df.to_csv(filtered_file_path, index=False)
   return filtered_df

VM = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/VM.csv')
Mac1 = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/MacBookPro1.csv')
Mac2 = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/MacBookPro2.csv')
RasPi = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/RasberryPi.csv')

plt.xlim(0, np.maximum(np.maximum(np.maximum(np.max(VM), np.max(Mac1)),np.max(Mac2)),np.max(RasPi)))
#np.max(Mac2)
#np.max(RasPi)
#np.max([VM, Mac1, Mac2, RasPi])
#plt.xlim(0, np.max)
sns.kdeplot(VM, color='r')
sns.kdeplot(Mac1, color='g')
sns.kdeplot(Mac2, color='y')
sns.kdeplot(RasPi, color='c')

sns.histplot(Mac2)

#unified data

sizeOfData = 300

clock = np.concatenate([np.concatenate([1.4*np.ones(sizeOfData), 2.5*np.ones(sizeOfData)]),1.8*np.ones(sizeOfData)])

core = np.concatenate([np.concatenate([4*np.ones(sizeOfData), 2*np.ones(sizeOfData)]),4*np.ones(sizeOfData)])

ram = np.concatenate([np.concatenate([8*np.ones(sizeOfData), 8*np.ones(sizeOfData)]),4*np.ones(sizeOfData)])

exeTime = np.concatenate([np.concatenate([Mac1[0:sizeOfData],Mac2[0:sizeOfData]]), RasPi[0:sizeOfData]])

combined = {'clock' : clock,
        'core' : core,
        'ram' : ram,
        'time' : exeTime} 

combined = pd.DataFrame(combined)
combined.head()

#GLM 
import statsmodels.api as sm
import statsmodels.formula.api as smf

model= smf.glm(formula = 'time ~ clock + core + ram', family = sm.families.Gaussian(), data = combined).fit()
model.summary()

model2= smf.glm(formula = 'time ~ clock + core + ram + clock:core', family = sm.families.Gaussian(), data = combined).fit()
model2.summary()

#other exploratory analysis: time-series

x = np.linspace(0,1,len(VM))
plt.ylim(0, 1.1)
plt.plot(x, VM)
#Variability is almost non-existent relative to others

x = np.linspace(0,1,len(Mac1))
plt.ylim(0, 1.1)
plt.plot(x, Mac1)

x = np.linspace(0,1,len(Mac2))
plt.ylim(0, 1.1)
plt.plot(x, Mac2)

x = np.linspace(0,1,len(RasPi))
plt.ylim(0, 1.5)
plt.plot(x, RasPi)

### trial Mac2 half length

Mac2_try = Mac2[Mac2 <=0.37]

sns.kdeplot(Mac2, color='y')
sns.kdeplot(Mac2_try, color='r')





