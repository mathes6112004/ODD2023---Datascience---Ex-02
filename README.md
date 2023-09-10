# Ex02-Outlier
You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

(i) Using IQR detect weight outliers and print them

(ii) Using IQR, detect height outliers and print them

## Aim:
TO detect and remove the outliers in the given data set and save the final data.

## EXPLANATION:
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

## ALGORITHM:
STEP 1 Read the given Data

STEP 2 Get the information about the data

STEP 3 Detect the Outliers using IQR method and Z score

STEP 4 Remove the outliers

## CODE AND OUTPUT
```
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/52863607-9403-4d68-b6d1-b4ecc898d0ee)

```
from google.colab import files
uploaded = files.upload()
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/f13a32b6-98da-4efa-8376-309f0f8a404f)

```
df = pd.read_csv("bhp.csv")
q1 = df['price_per_sqft'].quantile(0.25)
q2 = df['price_per_sqft'].quantile(0.5)
q3 = df['price_per_sqft'].quantile(0.75)
iqr = q3-q1
iqr
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/7a2273b0-b82d-471b-81b6-4180c23ba2a6)

```
low = q1-1.5*iqr
low
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/5f0fd443-b383-42c1-8317-042ef101b6f8)

```
high = q3+1.5*iqr
high
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/31c43390-13ad-4bae-8702-c8a6410c833b)

```
df = df[((df['price_per_sqft']>=low) & (df['price_per_sqft']<=high))]
df
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/3989396c-953c-4aa3-b6b3-e86bee7531a6)

```
z = np.abs(stats.zscore(df['price_per_sqft']))
z
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/1aefd877-e4b5-4301-831d-c7d69b4e9ece)

```
df1 = df[z<3]
df1
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/623f4d88-f4ca-42a3-8703-eb53c782a1b8)

```
from google.colab import files
uploaded = files.upload()
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/e0135abc-3cf7-49c2-ae1e-68e4dccda3f2)

```
df = pd.read_csv("height_weight.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/1b06ad91-b99c-40af-aacf-6840c589e940)

```
low = q1 - 1.5*iqr
low
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/cb38d668-c592-407a-8211-e8efa20f1c71)

```
high = q3+1.5*iqr
high
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/a461b70d-2de3-45fe-b0a4-0f89cf4d9d65)

```
df = df[((df['height'] >=low) & (df['height']<= high))]
df
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/a25f4a99-4b7e-4464-903a-f88159b2feb8)

```
z = np.abs(stats.zscore(df['height']))
z
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/7226af9a-1735-4ef8-b37d-b7b31980c739)

```
df1 = df[z<3]
df1
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/cc06c6ff-e4bb-4c1f-a7ce-9a57009d069c)

```
df = pd.read_csv("height_weight.csv")
q1 = df['weight'].quantile(0.25)
q2 = df['weight'].quantile(0.5)
q3 = df['weight'].quantile(0.75)
iqr = q3-q1
iqr
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/76aad33e-48ff-4cd7-a397-ad8ad0058cf5)

```
low = q1 - 1.5*iqr
low
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/968e03c9-dd25-4d7c-b0aa-a501e623ba80)

```
high = q3 + 1.5*iqr
high
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/36b2c610-b542-4237-abf3-7ca393bb21fe)

```
df1 = df[((df['weight'] >=low) & (df['weight']<= high))]
df1
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/971353a7-6646-474e-aac7-c1b5a77282e0)

```
z = np.abs(stats.zscore(df1['weight']))
z
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/4fdc96bd-cd1b-4bce-a2d8-446c09895bae)

```
df2 = df1[z<3]
df2
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/3ca5ddd2-6b99-4b31-a856-4796d5d39e03)

```
from google.colab import files
uploaded = files.upload()
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/e2a30f94-50bb-4484-a73c-94800feb6df6)

```
df = pd.read_csv("heights.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/22710613-4eb6-46be-8e3a-98325e34938a)

```
low = q1 - 1.5*iqr
low
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/12560539-2134-4e2a-a25f-9288935a8b21)

```
high = q3 + 1.5*iqr
high
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/347d027a-9c53-4488-aaab-68744e98bfce)

```
df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/5edaf9be-6e7b-436f-a3eb-d4a6782e988b)

```
z = np.abs(stats.zscore(df['height']))
z
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/813dc0d0-9778-4e29-83d3-61f70acdc041)

```
df1 = df[z<3]
df1
```
![image](https://github.com/mathes6112004/ODD2023---Datascience---Ex-02/assets/119477782/de4770d5-9637-40da-a34a-1fff5b4663ac)

## RESULT:
The given datasets are read and outliers are detected and are removed using IQR and z-score methods.


