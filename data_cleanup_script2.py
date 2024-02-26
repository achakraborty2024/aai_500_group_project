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


def clean_df(file_path, removeTime = True):
   filtered_file_path = file_path.split('.')[0] + '_filtered.csv'
   df = pd.read_csv(file_path)

   # Delete the column
   if(removeTime == True):
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
   
   #filtered_df = data[data >= lower_bound]
   #filtered_df = filtered_df[filtered_df <= upper_bound]
   
   #filtered_df = df[df['Execution Time'] >= lower_bound]
   
   print("hellow1: ", len(df))
   df.drop(df.loc[df['Execution Time']<=lower_bound].index, inplace=True)
   df.drop(df.loc[df['Execution Time']>=upper_bound].index, inplace=True)
   print("hellow2: " , len(df))
   
   if(removeTime == True):
       filtered_df = df['Execution Time']
       print("filtered data max: ", np.max(filtered_df))
       print("filtered data min: ", np.min(filtered_df))
       
   else:
       filtered_df = df
       print("filtered data max: ", np.max(filtered_df['Execution Time']))
       print("filtered data min: ", np.min(filtered_df['Execution Time']))
   
   #filtered_df = filtered_df.drop(filtered_df.loc[filtered_df['Execution Time'] >= upper_bound].index, inplace = True)

   #filtered_df = data[(data >= lower_bound) | (data <= upper_bound)]
   #this is not giving the desired filtered data
   
   filtered_df = filtered_df.reset_index(drop = True)

   # # Save the filtered DataFrame back to a new CSV file, without the unwanted rows
   filtered_df.to_csv(filtered_file_path, index=False)
   
   return filtered_df

VM = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/VM.csv')
Mac1 = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/MacBookPro1.csv')
Mac2 = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/MacBookPro2.csv')
RasPi = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/RasberryPi.csv')


Mac1_series = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/MacBookPro1.csv', False)
Mac1_series.head()
Mac1_series.tail()

Mac2_series = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/MacBookPro2.csv', False)
Mac2_series.head()

RasPi_series = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/RasberryPi.csv', False)
RasPi_series.head()
RasPi_series.tail()

VM_series = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/VM.csv', False)
VM_series.head()
VM_series.tail()
#Mac2_series["Time"][0]
#print(Mac2_series["Time"][0][14],
#Mac2_series["Time"][0][15],
##Mac2_series["Time"][0][16],
#Mac2_series["Time"][0][17])
#len(Mac2_series)
#int(Mac1_series["Time"][43][12])
Mac1_series = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/MacBookPro1.csv', False)
Mac1_series2 = Mac1_series
Mac1_time = []
#print("LENGTH: ", len(Mac1_series))
for i in range (0, len(Mac1_series2)):
    Mac1_time += [ -17258 + 60*60*int(Mac1_series["Time"][i][12])  + 60*(10*int(Mac1_series["Time"][i][14]) + int(Mac1_series["Time"][i][15])) + 10*int(Mac1_series["Time"][i][17]) + int(Mac1_series["Time"][i][18])]

Mac1_series2["Time"] = Mac1_time 
Mac1_series2.head()

Mac1_series2.set_index('Time', inplace = True)
Mac1_series2.head()


Mac1_series2.dropna(inplace = True)
Mac1_series2.head()
Mac1_series2.plot()


from statsmodels.tsa.seasonal import seasonal_decompose
series1 = pd.Series(Mac1_series2['Execution Time'])
series1.head()
results1 = seasonal_decompose(series1, model='additive', period = 100)

plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(series1, label='Original')
plt.legend(loc='best')
plt.subplot(312)
plt.plot(results1.trend, label='Trend')
bottom, top = plt.ylim() 
mean = (bottom + top)/2
newbottom = mean - 0.05
newtop = mean + 0.05
plt.ylim(newbottom, newtop)
plt.legend(loc='best')
plt.subplot(313)
plt.plot(results1.seasonal, label='Seasonality')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Mac2_series = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/MacBookPro2.csv', False)
Mac2_series2 = Mac2_series
Mac2_time = []
for i in range (0, len(Mac2_series)):
    Mac2_time += [-357 + 60*(10*int(Mac2_series["Time"][i][14]) + int(Mac2_series["Time"][i][15])) + 10*int(Mac2_series["Time"][i][17]) + int(Mac2_series["Time"][i][18])]


Mac2_series2["Time"] = Mac2_time 
Mac2_series2.head()
Mac2_series.head()

Mac2_series2.set_index('Time', inplace = True)
Mac2_series2.head()

Mac2_series2.dropna(inplace = True)
Mac2_series2.plot()

from statsmodels.tsa.seasonal import seasonal_decompose
series = pd.Series(Mac2_series2['Execution Time'])
series.head()
results = seasonal_decompose(series, model='additive', period = 100)

print("sample mean: ", np.mean(Mac2_series2['Execution Time']))
print("sample std: ", np.std(Mac2_series2['Execution Time']))

plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(series, label='Original')
plt.legend(loc='best')

#plot the trend within one standard deviation away from the sample mean
plt.subplot(312)
plt.plot(results.trend, label='Trend')
bottom, top = plt.ylim() 
mean = (bottom + top)/2
std = np.std(Mac2_series2['Execution Time'])
print("std: ", std)
newbottom = mean - 1*std
newtop = mean + 1*std
plt.ylim(newbottom, newtop)
plt.legend(loc='best')



plt.subplot(313)
plt.plot(results.seasonal, label='Seasonality')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
RasPi_series = clean_df('/Users/geoffreyfadera/Documents/USD Stuff/AAI-500/aai_500_group_project/RasberryPi.csv', False)
RasPi_series2 = RasPi_series
RasPi_time = []
for i in range (0, len(RasPi_series)):
    RasPi_time += [ -23343 + 60*60*int(RasPi_series["Time"][i][12])  + 60*(10*int(RasPi_series["Time"][i][14]) + int(RasPi_series["Time"][i][15])) + 10*int(RasPi_series["Time"][i][17]) + int(RasPi_series["Time"][i][18])]


RasPi_series2["Time"] = RasPi_time 
#RasPi_series2.head()
#RasPi_series.head()

RasPi_series2.set_index('Time', inplace = True)
#RasPi_series2.head()

RasPi_series2.dropna(inplace = True)
#RasPi_series2.plot()


from statsmodels.tsa.seasonal import seasonal_decompose
seriesR = pd.Series(RasPi_series2['Execution Time'])
seriesR.head()
resultsR = seasonal_decompose(seriesR, model='additive', period = 100)

plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(seriesR, label='Original')
plt.legend(loc='best')

#plot the trend within one standard deviation away from the sample mean
plt.subplot(312)
plt.plot(resultsR.trend, label='Trend')
bottom, top = plt.ylim() 
mean = (bottom + top)/2
std = np.std(RasPi_series2['Execution Time'])
print("std: ", std)
newbottom = mean - 1*std
newtop = mean + 1*std
plt.ylim(newbottom, newtop)
plt.legend(loc='best')


plt.subplot(313)
plt.plot(resultsR.seasonal, label='Seasonality')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

VM_series2 = VM_series
VM_time = []
for i in range (0, len(VM_series)):
    VM_time += [-991 + 60*(10*int(VM_series["Time"][i][14]) + int(VM_series["Time"][i][15])) + 10*int(VM_series["Time"][i][17]) + int(VM_series["Time"][i][18])]


VM_series2["Time"] = VM_time 
VM_series2.head()
VM_series.head()

VM_series2.set_index('Time', inplace = True)
VM_series2.head()

VM_series2.dropna(inplace = True)
VM_series2.plot()

from statsmodels.tsa.seasonal import seasonal_decompose
seriesV = pd.Series(VM_series2['Execution Time'])
seriesV.head()
resultsV = seasonal_decompose(seriesV, model='additive', period = 200)

plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(seriesV, label='Original')
plt.legend(loc='best')
plt.subplot(312)
plt.plot(resultsV.trend, label='Trend')
bottom, top = plt.ylim() 
mean = (bottom + top)/2
newbottom = mean - 0.05
newtop = mean + 0.05
plt.ylim(newbottom, newtop)
plt.legend(loc='best')
plt.subplot(313)
plt.plot(resultsV.seasonal, label='Seasonality')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#plt.xlim(0, np.maximum(np.maximum(np.maximum(np.max(VM), np.max(Mac1)),np.max(Mac2)),np.max(RasPi)))

plt.xlim(0,0.6)
#np.max(Mac2)
#np.max(RasPi)
#np.max([VM, Mac1, Mac2, RasPi])
#plt.xlim(0, np.max)
sns.kdeplot(VM, color='r', label="Virtual Machine")
sns.kdeplot(Mac1, color='g', label = "MacBookPro 1")
plt.legend()
plt.xlabel('Execution time in seconds')


sns.kdeplot(Mac2, color='y', label = "MacBookPro 2")
sns.kdeplot(RasPi, color='c', label = "Raspberry Pi")
plt.legend()
plt.xlabel('Execution time in seconds')

sns.histplot(Mac2)

#unified data 
clock = []
sizeOfData = 800

clock = np.concatenate([np.concatenate([1.4*np.ones(sizeOfData), 2.5*np.ones(sizeOfData)]),1.8*np.ones(sizeOfData)])

core = np.concatenate([np.concatenate([4*np.ones(sizeOfData), 2*np.ones(sizeOfData)]),4*np.ones(sizeOfData)])

ram = np.concatenate([np.concatenate([8*np.ones(sizeOfData), 8*np.ones(sizeOfData)]),4*np.ones(sizeOfData)])
ram2 = np.concatenate([np.concatenate([3*np.ones(sizeOfData), 3*np.ones(sizeOfData)]),2*np.ones(sizeOfData)])

#for RAM, 8GB and 4GB need to be log transformed because the available RAM sizes are all powers of 2.
#so 2GB should be entered as 1, 4GB should be entered as 2, 8GB should be entered as 3, etc

exeTime = np.concatenate([np.concatenate([Mac1[0:sizeOfData],Mac2[0:sizeOfData]]), RasPi[0:sizeOfData]])

print(clock)
print(core)

print(ram)

print(exeTime)


combined = {'clock' : clock,
        'core' : core,
        'ram' : ram,
        'time' : exeTime} 

combined2 = {'clock' : clock,
        'core' : core,
        'ram' : ram2,
        'time' : exeTime} 

combined = pd.DataFrame(combined)
combined2 = pd.DataFrame(combined2)

#compare execution time with ram size constant

y




#GLM 
import statsmodels.api as sm
import statsmodels.formula.api as smf

model= smf.glm(formula = 'time ~ clock + core + ram', family = sm.families.Gaussian(), data = combined).fit()
print(model.summary())

model2= smf.glm(formula = 'time ~ clock + core + ram', family = sm.families.Gaussian(), data = combined2).fit()
print(model2.summary())

#other exploratory analysis: time-series

sns.kdeplot(combined['time'], color='r')

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





