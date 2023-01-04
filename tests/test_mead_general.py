# Standard imports
import unittest
import numpy as np

# My imports
import mead_general as mead

class test_general(unittest.TestCase):

    ### High level ###

    def test_none_or_string(self):
        assert mead.none_or_string('None') == None, 'Should be python None'
        assert mead.none_or_string('Mead') == 'Mead', 'Should return string'

    ### ###

    ### Basic ###

    def test_periodic_integer(self):
        assert mead.periodic_integer(1, 10) == 1
        assert mead.periodic_integer(11, 10) == 1
        assert mead.periodic_integer(12, 5) == 2
        assert mead.periodic_integer(0, 10) == 0
        assert mead.periodic_integer(10, 10) == 0

    def test_periodic_float(self):
        assert mead.periodic_float(1., 2.) == 1.
        assert mead.periodic_float(2., 2.) == 0.
        assert mead.periodic_float(1.5, 1.) == 0.5
        assert mead.periodic_float(0., 1.2) == 0.

    def test_opposite_side(self):
        assert mead.opposite_side('left') == 'right'
        assert mead.opposite_side('right') == 'left'

    def test_number_name(self):
        assert mead.number_name(10) == 'ten'
        assert mead.number_name(100) == 'hundred'
        assert mead.number_name(1e6) == 'million'

    # def test_file_length(fname):
    #     assert mead.file_length(fname)

    def test_mrange(self):
        assert list(mead.mrange(3, b=None, step=None)) == [0, 1, 2, 3]
        assert list(mead.mrange(1, 3)) == [1, 2, 3]
        assert list(mead.mrange(1, 10, step=2)) == [1, 3, 5, 7, 9]
        assert list(mead.mrange(3, 1)) == []
        assert list(mead.mrange(1, 1)) == [1]
        assert list(mead.mrange(1)) == [0, 1]

    # def test_is_float_close_to_integer(self):
    #     assert mead.is_float_close_to_integer(x)

    def test_bin_edges_for_integers(self):
        assert mead.bin_edges_for_integers(1, 5) == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        assert mead.bin_edges_for_integers([1, 1, 4, 3]) == [0.5, 1.5, 2.5, 3.5, 4.5]

    ### ###

    ### Collections ###

    def test_key_from_value(self):
        dict = {'hat': 1, 'glove': 2}
        assert mead.key_from_value(dict, 2) == 'glove'

    def test_count_entries_of_nested_list(self):
        assert mead.count_entries_of_nested_list([[1, 2], [3], [4, [5, 6]]]) == 6
        assert mead.count_entries_of_nested_list([[[1, 2]]]) == 2
        assert mead.count_entries_of_nested_list([[]]) == 0
        assert mead.count_entries_of_nested_list(['gobble']) == 1

    def test_create_unique_list(self):
        assert mead.create_unique_list(['spam', 'spam', 'ham']) == ['spam', 'ham']
        assert mead.create_unique_list(['spam', 'ham']) == ['spam', 'ham']

    def test_remove_list_from_list(self):
        assert mead.remove_list_from_list([1, 3], [1, 2, 3, 4]) == [2, 4]
        assert mead.remove_list_from_list([], [1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
        assert mead.remove_list_from_list([1, 2, 3], [1, 2, 3]) == []
        assert mead.remove_list_from_list([1], [1, 1, 2]) == [1, 2]

    def test_remove_multiple_elements_from_list(self):
        assert mead.remove_multiple_elements_from_list(['ham', 'spam', 'ham'], [2]) == ['ham', 'spam']
        assert mead.remove_multiple_elements_from_list(['ham', 'spam', 'ham'], [0]) == ['spam', 'ham']
        assert mead.remove_multiple_elements_from_list(['ham', 'spam', 'ham'], [0, 2]) == ['spam']

    def test_second_largest(self):
        assert mead.second_largest([1, 2, 3, 4]) == 3
        assert mead.second_largest([1., 100., 1000.]) == 100.
        #assert mead.second_largest([1, 2, 3., 3.]) == 3. # TODO: Does not work, should probably return 2.

    ### ###

    ### Numpy ###

    def test_arange(self):
        assert (mead.arange(1, 3) == np.array([1, 2, 3])).all()
        assert (mead.arange(5, 10) == np.array([5, 6, 7, 8, 9, 10])).all()
        assert (mead.arange(0, 4) == np.array([0, 1, 2, 3, 4])).all()

    ### ###

if __name__ == '__main__':
    unittest.main()
    