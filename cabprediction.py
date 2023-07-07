from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import r2_score
import pandas as pd
df=pd.read_parquet('yellow_tripdata_2023-01.parquet')
# print(df.head())


# passenger_count: The number of passengers in the taxi.
# trip_distance: The distance traveled during the taxi ride in miles.
# PULocationID: The identifier for the zone where the taxi ride started (pickup location).
# DOLocationID: The identifier for the zone where the taxi ride ended (drop-off location).
# mta_tax: The tax amount imposed by the Metropolitan Taxicab Authority.
# tolls_amount: The amount of tolls paid during the trip.
# improvement_surcharge: Additional surcharges or fees applied to the fare amount.
# congestion_surcharge: A surcharge applied to taxi fares in areas with high traffic congestion.
# airport_fee: An additional fee specific to airport trips.
#extra


# fare_amount:output 

df=df.drop(['tpep_pickup_datetime','tpep_dropoff_datetime','VendorID','store_and_fwd_flag','RatecodeID','payment_type'],axis=1)

# print(df.info())

import matplotlib.pyplot as plt
plt.scatter(df['passenger_count'],df['fare_amount'])
# plt.show()

# print(df.describe())
df=df.dropna()
# print(df.head())
# print(df.columns)
df.to_csv('newdata.csv')
x=df[['passenger_count', 'trip_distance','extra', 'mta_tax', 'tip_amount', 'tolls_amount','improvement_surcharge', 'congestion_surcharge','airport_fee']]
y=df['fare_amount']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


linear=LinearRegression()
linear.fit(x_train,y_train)
y_linear=linear.predict(x_test)
r2linear=r2_score(y_test,y_linear)
# print("r2 score for linear regression ",r2linear)

pc=float(input("enter number of passenger : "))
td=float(input("enter total trip distance "))
e=float(input("enter the extra amount for example waiting charge if any "))
mta=float(input("enter metropolian tax "))
tip=float(input("enter amount of tip "))
toll=float(input("enter amount of toll tax "))
imp=float(input("enter the amount of improvement charge "))
csurch=float(input('congestion surcharge '))
air=float(input("enter the amount of airport fee if any "))

newdata=pd.DataFrame({'passenger_count' :[pc],'trip_distance':[td],'extra':[e], 'mta_tax':[mta], 'tip_amount':[tip], 'tolls_amount':[toll],'improvement_surcharge':[imp], 'congestion_surcharge':[csurch],'airport_fee':[air]})

newpred=linear.predict(newdata)
print("predicted fare amount is ", newpred[0])

