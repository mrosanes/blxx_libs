import numpy as np
from scipy.optimize import leastsq


class BaseFit(object):
    """
    Base class to fit data.
    """

    def fit(self, *args, **kwargs):
        """
        :param args: Arguments needed to do the fit. They depend of the fit
        type.
        :param kwargs:
        :return:
        """
        raise NotImplementedError('You must implement the method.')


class GaussianFit(BaseFit):
    """
    Class to fit the data to a gaussian + slope curve.
    """

    def gaussian(self, x, v=[0.0, 0.0, 1.0, 0.0, 1.0]):
        """
        Funtion to implement a guassian funtion plus a line:
        y = A + B*x + C*exp(-((x - D) / E)**2 / 2)

        :param x: numpy array or list with the data
        :param v: offset(A), slope(B), height(C), center(D), sigma(E)
        :return: numpy array
        """

        return v[0] + v[1]*np.array(x) + v[2]*np.exp(-(((x - v[3])/v[4])**2)/2)

    def fit(self, x_data, y_data):
        """
        :param x_data: numpy array or list with the X data.
        :param y_data: numpy array or list with the Y data
        :return: offset, slope, height, center, sigma, FWHM
        """

        # init_values are the initial values taken to start the fit
        # A fraction of the Y data range is taken as initial gaussian height and
        # X data range center is taken as initial gaussian center
        y_min = min(y_data)
        y_max = max(y_data)
        x_min = min(x_data)
        x_max = max(x_data)

        offset = y_min
        slope = 0
        height = 0.6 * (y_max-y_min)
        center = (x_min + x_max)/2
        sigma = 1

        init_values = [offset, slope, height, center, sigma]

        # errfunc returns the difference between data and fitted function
        # at the given x for the given parameters v
        errfunc = lambda v, x, y: (self.gaussian(x, v) - y)

        # leastsq search the v parameters that minimizes the sum of squared
        # errfunc values
        results, success = leastsq(errfunc, init_values, args=(x_data, y_data))

        noffset = results[0]
        nslope = results[1]
        nheight = results[2]
        ncenter = results[3]
        nsigma = abs(results[4])
        fwhm = 2*np.sqrt(2*np.log(2))*nsigma
        return noffset, nslope, nheight, ncenter, nsigma, fwhm

