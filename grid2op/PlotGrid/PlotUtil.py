import numpy as np

def format_value_unit(value, unit):
    return "{} {}".format(value, unit)

def middle_from_points(x1, y1, x2, y2):
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5

def vec_from_points(x1, y1, x2, y2):
    return x2 - x1, y2 - y1

def mag_from_points(x1, y1, x2, y2):
    x, y = vec_from_points(x1, y1, x2, y2)
    return np.linalg.norm([x, y])

def norm_from_points(x1, y1, x2, y2):
    x, y, = vec_from_points(x1, y1, x2, y2)
    return norm_from_vec(x, y)

def norm_from_vec(x, y):
    n = np.linalg.norm([x, y])
    return x / n, y / n

def orth_from_points(x1, y1, x2, y2):
    x, y = vec_from_points(x1, y1, x2, y2)
    return -y, x

def orth_norm_from_points(x1, y1, x2, y2):
    x, y = vec_from_points(x1, y1, x2, y2)
    xn, yn = norm_from_vec(x, y)
    return -yn, xn
