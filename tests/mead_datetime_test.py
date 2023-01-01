# Standard imports
import unittest
from datetime import datetime

# My imports
import mead_datetime as mdt

class test_datetime(unittest.TestCase):

    def test_num_days_between(self):
        date1, date2 = datetime(2022, 12, 1), datetime(2022, 12, 7)
        assert mdt.num_days_between(date1, date2) == 7, 'Should be 7'
        assert mdt.num_days_between(date1, date2, inclusive=False) == 6, 'Should be 6'
        assert mdt.num_days_between(date2, date1) == 0, 'Should be 0'
        assert mdt.num_days_between(date1, date2, exclude=[datetime(2022, 12, 2)]) == 6, 'Should be 6'
        print('num_days_between tests passed')

    def test_get_all_dates_between_dates(self):
        date1, date2 = datetime(2022, 12, 1), datetime(2022, 12, 3)
        assert mdt.get_all_dates_between_dates(date1, date2) == [datetime(2022, 12, 1), datetime(2022, 12, 2), datetime(2022, 12, 3)], 'Should be a list of three dates'
        assert mdt.get_all_dates_between_dates(date1, date2, inclusive=False) == [datetime(2022, 12, 1), datetime(2022, 12, 2)], 'Should be a list of two dates'
        assert mdt.get_all_dates_between_dates(date2, date1) == [], 'Should be an empty list'
        print('get_all_dates_between_dates tests passed')

if __name__ == '__main__':
    unittest.main()