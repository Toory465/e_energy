from custom_exceptions import InputInfoException, VerticalLineException, ParallelLineException


def get_line_info(point1, point2):
    """
    This method returns slope and interceptor of the line passing through given points in the form y = ax + b.
    If the line is a vertical line, the code throw an exception since the assumption of the question is that
    The equation is always in the form of y = ax + b which doesn't hold for a vertical line with equation of X = a.
    :param point1: First point as tuple that describe two coordinates of (x1, y1)
    :param point2: Second point as tuple that describe two coordinates of (x2, y2)
    :return: slope and interceptor as a tuple.
    """
    try:
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
    except:
        raise InputInfoException('The first and second parameters should be passed as tuples.')

    if x1 == x2:
        raise VerticalLineException('We have a vertical line. The assumption of having a line'
                                    f'with form of y = ax + b dose not hold.\nThe equation is: x = {x1}')
    else:
        a = (y2 - y1) / (x2 - x1)
        b = y1 - (a * x1)

    return a, b


def get_two_lines_intersection(line1, line2):
    """
    This function returns the intersection point of two lines in 2D.
    :param line1: slope and interception of the first line in a tuple format.
    :param line2: slope and interception of the second line in a tuple format.
    :return: The coordinate of the intersection point as a tuple (x, y).
    """
    try:
        slope1 = line1[0]
        intercept1 = line1[1]
        slope2 = line2[0]
        intercept2 = line2[1]
    except:
        raise InputInfoException('The first and second parameters should be passed as tuples.')

    if slope1 == slope2:
        raise ParallelLineException("Two line are parallel and they don't have intersection.")
    else:
        x = (intercept2 - intercept1) / (slope1 - slope2)
        y = slope1 * x + intercept1
    return x, y
