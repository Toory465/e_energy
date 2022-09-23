from unittest import TestCase
from geometry_3d import *


class Test(TestCase):
    def test_plane_instantiation(self):
        normal = 3, 5
        point = 0, 2, 1
        self.assertRaises(InputInfoException, Plane, normal, point)
        normal = 1, 3, 5
        point = 0, 2
        self.assertRaises(InputInfoException, Plane, normal, point)

    def test_get_equation_parameters(self):
        normal = 1, 3, 5
        point = 0, 2, 1
        plane = Plane(normal, point)
        self.assertEqual(plane.get_equation_parameters(), (1, 3, 5, -11))

    def test_get_distance_with_wrong_input(self):
        normal = 1, 3, 5
        point = 0, 2, 1
        plane = Plane(normal, point)
        given_point = 1, 2
        self.assertRaises(InputInfoException, plane.get_distance, given_point)

    def test_get_distance(self):
        normal = 1, 3, 5
        point = 0, 2, 1
        plane = Plane(normal, point)
        given_point = 1, 10, 1
        self.assertAlmostEqual(plane.get_distance(given_point), 7.9444, 4)

