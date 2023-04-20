from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

#Gather data

boston_dataset = load_boston()
data = pd.DataFrame(data = boston_dataset.data, columns = boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis = 1)
features.head()

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns = ['PRICE'])
target.shape

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

ZILLOW_MEDIAN_PRICE = 583.3 #La mediana de precios de propiedades en Boston actualmente segun Zillow.com
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)

property_stats = features.mean().values.reshape(1,11)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

#Challenge: calculate the MSE and RMSE using sklearn

mse = mean_squared_error(target, fitted_vals)
rmse = mean_squared_error(target, fitted_vals, squared = False)
print(mse, rmse)

def get_log_estimate(nr_rooms,
                    students_per_classroom,
                    next_to_river = False,
                    high_confidence = True):
    #Configure property
    
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
    
    #Make prediction
    log_estimate = regr.predict(property_stats)
    
    #Calc range
    if high_confidence:
        #Do X
        upper_bound = log_estimate + 2*rmse
        lower_bound = log_estimate - 2*rmse
        interval = 95
    else:
        #Do y
        upper_bound = log_estimate + rmse
        lower_bound = log_estimate - rmse
        interval = 68
    
    return log_estimate[0][0], upper_bound[0][0], lower_bound[0][0], interval


def get_dollar_estimate(rm, ptratio, chas = False, large_range = True):
    
    """Estimate the price of a property in Boston.
    
    Keyword arguments:
    
    rm = Number of rooms
    ptratio = Number of students per classroom
    chas = if True the property is next to the river. Default is False
    large_range = True for a 95% prediction interval, False for a 68% interval.
    """
    
    log_estimate, upper_bound, lower_bound, interval = get_log_estimate(rm, ptratio, 
                                                                        next_to_river = chas, 
                                                                        high_confidence = large_range)
    
    if rm<1 or ptratio<1:
        print('That is unrealistic. Try again.')
        return
    
    
    #Convert to today's dollars
    dollar_est = np.e**log_estimate * 1000 * SCALE_FACTOR
    dollar_hi = np.e**upper_bound * 1000 * SCALE_FACTOR
    dollar_low = np.e**lower_bound * 1000 * SCALE_FACTOR


    #Round the dollar values to nearest thousand
    rounded_est = np.around(dollar_est, -3)
    rounded_hi = np.around(dollar_hi, -3)
    rounded_low = np.around(dollar_low, -3)

    print(f'The estimated property value is {rounded_est}.')
    print(f'At {interval}% confidence the valuation range is:')
    print(f'USD {rounded_low} at the lower end to USD {rounded_hi} at the high end.')