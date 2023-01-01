import mead_datetime as mdt
from datetime import datetime

def test_num_days_between():
    date1, date2 = datetime(2022, 12, 1), datetime(2022, 12, 7)
    assert mdt.num_days_between(date1, date2) == 7
    assert mdt.num_days_between(date1, date2, inclusive=False) == 6
    assert mdt.num_days_between(date2, date1) == 0
    assert mdt.num_days_between(date1, date2, exclude=[datetime(2022, 12, 2)]) == 6
    print('num_days_between tests passed')

def test_get_all_dates_between_dates():
    date1, date2 = datetime(2022, 12, 1), datetime(2022, 12, 3)
    assert mdt.get_all_dates_between_dates(date1, date2) == [datetime(2022, 12, 1), datetime(2022, 12, 2), datetime(2022, 12, 3)]
    assert mdt.get_all_dates_between_dates(date1, date2, inclusive=False) == [datetime(2022, 12, 1), datetime(2022, 12, 2)]
    assert mdt.get_all_dates_between_dates(date2, date1) == [] 
    print('get_all_dates_between_dates tests passed')

test_num_days_between()
test_get_all_dates_between_dates()