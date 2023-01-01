from datetime import timedelta

def num_days_between(date1, date2, inclusive=True, exclude=[]):
    '''
    Counts the number of days between two dates
    @params
        date1: datetime object
        date2: datetime object, should be more recent than date1
        inclusive: Is date2 included within the list?
        exclude: list of dates (datetime objects) to exclude from count
    '''
    # Iterate over the days between the two dates
    current_date = date1
    num_days = 0 # Initialize a counter for the number of days
    while (current_date <= date2) if inclusive else (current_date < date2):
        if current_date not in exclude:
            num_days += 1 # If the current day is not in the exclude list, increment the counter
        current_date += timedelta(days=1) # Move to the next day
    return num_days # Return the number of days


def get_all_dates_between_dates(date1, date2, inclusive=True):
    '''
    Returns a list of all dates between date1 and date2
    @params
        date1: datetime object
        date2: datetime object, should be more recent than date1
        inclusive: Is date2 included within the list?
    '''
    dates = []
    current_date = date1
    while (current_date <= date2) if inclusive else (current_date < date2):
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates