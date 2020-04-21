import numpy as np

class PlotUtil:
    @staticmethod
    def format_value_unit(value, unit):
        return "{} {}".format(value, unit)

    @staticmethod
    def middle_from_points(x1, y1, x2, y2):
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    @staticmethod
    def vec_from_points(x1, y1, x2, y2):
        return x2 - x1, y2 - y1

    @staticmethod
    def mag_from_points(x1, y1, x2, y2):
        x, y = PlotUtil.vec_from_points(x1, y1, x2, y2)
        return np.linalg.norm([x, y])

    @staticmethod
    def norm_from_points(x1, y1, x2, y2):
        x, y, = PlotUtil.vec_from_points(x1, y1, x2, y2)
        return PlotUtil.norm_from_vec(x, y)

    @staticmethod
    def norm_from_vec(x, y):
        n = np.linalg.norm([x, y])
        return x / n, y / n

    @staticmethod
    def orth_from_points(x1, y1, x2, y2):
        x, y = PlotUtil.vec_from_points(x1, y1, x2, y2)
        return -y, x

    @staticmethod
    def orth_norm_from_points(x1, y1, x2, y2):
        x, y = PlotUtil.vec_from_points(x1, y1, x2, y2)
        xn, yn = PlotUtil.norm_from_vec(x, y)
        return -yn, xn
