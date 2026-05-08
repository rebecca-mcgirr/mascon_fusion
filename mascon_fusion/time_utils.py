
"""
Module: time_utils.py

Description:
Time-conversion helpers. Provides conversions between
days-since-reference, decimal-year and calendar date.

Functions:
- get_decyear
- decyear_from_daysince
- decyear_to_daysince
- YearFractiontoDatetime

Author: R McGirr 2026-03
"""

import datetime
import numpy as np

def get_decyear(time_array, year=2002, month=1, day=1):
    ref_date = datetime.date(year, month, day)
    decyear = decyear_from_daysince(time_array, ref_date)
    return decyear

def decyear_from_daysince(days_since, ref_date):
    """
    Calculate the decimal year accurately
    days_since: array or float, time in days since a reference date
    ref_date: datetime object, the reference date
    """
    # test whether days_since is an array or a float
    if isinstance(days_since, np.ndarray):
        # create an array of timedelta objects
        delta = np.array([datetime.timedelta(days=int(i)) for i in days_since])
        # create an array of datetime objects
        dates = np.array([ref_date + i for i in delta])
        # get total umber of days in each year from datetime
        days_in_year = np.array([datetime.datetime(date.year, 12, 31).timetuple().tm_yday for date in dates])
        # calculate the year fraction
        day_of_year = np.array([date.timetuple().tm_yday for date in dates])
        # create years array
        years = np.array([date.year for date in dates])
    else:
        # create a timedelta object
        delta = datetime.timedelta(days=int(days_since))
        # create a datetime object
        dates = ref_date + delta
        # get total number of days in the year from datetime
        days_in_year = datetime.datetime(dates.year, 12, 31).timetuple().tm_yday
        # calculate the day of the year
        day_of_year = dates.timetuple().tm_yday
        # create years variable
        years = dates.year

    return years + (day_of_year - 1) / days_in_year

def decyear_to_daysince(decyear, ref_date):
    '''
    Calculate the number of days since a reference date
    decyear: array or float, decimal year
    ref_date: datetime object, the reference date
    '''
    # test whether decyear is an array or a float
    if isinstance(decyear, np.ndarray):
        daysince = np.zeros_like(decyear)
        for i in range(len(decyear)):
            date = YearFractiontoDatetime(decyear[i])
            diff = date - ref_date
            daysince[i] = diff.days
    else:
        date = YearFractiontoDatetime(decyear)
        diff = date - ref_date
        daysince = diff.days
    return daysince

def YearFractiontoDatetime(yearFraction):
    '''
    Convert a decimal year to a datetime date object
    '''
    year = int(yearFraction)
    day = (yearFraction - year) * int(datetime.date(year,12,31).strftime("%j"))
    return datetime.date(year,1,1) + datetime.timedelta(day)

def YMDtoYearFraction(yr,month,day):
    '''
    Convert a datetime date obect to a decimal year
    '''
    date = datetime.date(yr,month,day)
    return date.year + int(date.strftime("%j"))/int(datetime.date(date.year,12,31).strftime("%j"))