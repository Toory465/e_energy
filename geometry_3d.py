from custom_exceptions import InputInfoException
import numpy as np


class Plane:
    def __init__(self, normal, point):
        """
        Class constructor.
        :param normal: A normal vector that is a perpendicular to the plane that is defined as a tuple (a, b, c).
        :param point: A given point coordinate as a tuple (x, y, z) that lies in the plane
        """
        try:
            self.normal = np.array([normal[0], normal[1], normal[2]])
            self.point = np.array([point[0], point[1], point[2]])
        except:
            raise InputInfoException('The first and second parameters should be passed as tuples with three members.')

    def get_equation_parameters(self):
        """
        This function returns the parameters a, b, c, d of the plane with equation ax + by + cz + d = 0
        of the current instance.
        The equation of the plane can be calculated from the following formula: a(x - x_p) + b(y - y_p) + c(z - z_p) = 0.
        So, we have: ax + by + cz - (<a, b, c> . <x_p, y_p, z_p>) = 0.
        :return: a, b, c, d as the equation parameters of the plane.
        """
        a, b, c = self.normal
        d = -self.normal.dot(self.point)
        return a, b, c, d

    def get_distance(self, point):
        """
        This function returns the distance of the given point to the current plane's instance.
        :param point: the coordinate of the given point as a tuple (x, y, z) in a 3D space.
        This function checks the input parameter and throw an exception if the input is not valid.
        :return: distance.
        """
        try:
            point = np.array([point[0], point[1], point[2]])
        except:
            raise InputInfoException('The point should be passed as tuples with three members.')

        # According to Math, the distance of point p with coordinates (x_0, y_0, z_0) and the given plane
        # with equation Ax + By + Cz + D = 0 is calculated by |Ax_0 + By_0+ Cz_0 - D|/sqrt(A^2 + B^2 + C^2).
        a, b, c, d = self.get_equation_parameters()
        ABC = np.array([a, b, c])
        distance = np.abs(ABC.dot(point) - d) / np.sqrt(np.sum(np.power(ABC, 2)))
        return distance











