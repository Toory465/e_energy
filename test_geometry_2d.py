from unittest import TestCase
from geometry_2d import *


class Test(TestCase):

    def test_get_line_info_with_wrong_inputs(self):
        point1 = 0
        point2 = 3, 7
        self.assertRaises(InputInfoException, get_line_info, point1, point2)
        point1 = 3, 5
        point2 = 4
        self.assertRaises(InputInfoException, get_line_info, point1, point2)

    def test_get_line_for_vertical_case(self):
        point1 = 3, 6
        point2 = 3, 7
        self.assertRaises(VerticalLineException, get_line_info, point1, point2)

    def test_get_line_for_non_vertical_case(self):
        #  test for horizontal case
        point1 = 4, 6
        point2 = 8, 6
        self.assertEqual(get_line_info(point1, point2), (0.0, 6.0))
        # and other cases
        point1 = 4, 6
        point2 = 8, 2
        self.assertEqual(get_line_info(point1, point2), (-1.0, 10.0))

    # Line intersection unit tests
    def test_get_two_lines_intersection_with_wrong_line_info(self):
        line1 = 1
        line2 = 2, 7
        self.assertRaises(InputInfoException, get_line_info, line1, line2)
        line1 = 3, 5
        line2 = 1
        self.assertRaises(InputInfoException, get_line_info, line1, line2)

    def test_get_two_lines_intersection_parallel_case(self):
        line1 = 4, 6
        line2 = 4, 7
        self.assertRaises(ParallelLineException, get_two_lines_intersection, line1, line2)

    def test_get_two_lines_intersection(self):
        line1 = 2, 6
        line2 = 4, 7
        self.assertEqual(get_two_lines_intersection(line1, line2), (-0.5, 5.0))
