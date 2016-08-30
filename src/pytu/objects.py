#
# Lacerda@Macarrao - 30/Ago/2016
#
import numpy as np
import argparse as ap
import scipy.stats as st
from scipy.ndimage.filters import gaussian_filter1d


class CustomArgumentParser(ap.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(CustomArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg


class tupperware_none:
    def __init__(self):
        pass

    def __getattr__(self, attr):
        r = self.__dict__.get(attr, None)
        return r


class tupperware:
    pass


class runstats(object):
    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y
        self.x_inv = y
        self.y_inv = x
        self.xbin = None
        self.debug = kwargs.get('debug', False)
        self._gsmooth = kwargs.get('smooth', None)
        self.sigma = kwargs.get('sigma', None)
        self._tendency = kwargs.get('tendency', None)
        self._inverse = kwargs.get('inverse', None)
        self.rstats(**kwargs)
        self.Rs, self.Rs_pval = st.spearmanr(x, y)
        self.Rp, self.Rp_pval = st.pearsonr(x, y)

        if self._inverse is not None:
            self.rstats_yx(**kwargs)

        if kwargs.get('OLS', False):
            self.OLS_bisector()

        if kwargs.get('poly1d', False):
            self.poly1d()

    def rstats(self, **kwargs):
        nx = len(self.x)
        nBox = kwargs.get('nBox', nx * kwargs.get('frac', 0.1))
        if nx > nBox:
            aux = calc_running_stats(self.x, self.y, **kwargs)
            self.xbin = aux[0]
            self.xbinCenter = aux[1]
            self.xMedian = aux[2]
            self.xMean = aux[3]
            self.xStd = aux[4]
            self.yMedian = aux[5]
            self.yMean = aux[6]
            self.yStd = aux[7]
            self.nInBin = aux[8]
            self.xPrc = aux[9]
            self.yPrc = aux[10]
            self.xPrcS = None
            self.yPrcS = None

            if self._tendency is True:
                aux = self.tendency(self.x, self.y, **kwargs)
                self.xT = aux[0]
                self.yT = aux[1]
                self.xbin = aux[2]
                self.spline = aux[3]
        else:
            self.xbin = self.x
            self.xbinCenter = self.x
            self.xMedian = self.x
            self.xMean = self.x
            self.xStd = self.x
            self.yMedian = self.y
            self.yMean = self.y
            self.yStd = self.y
            self.nInBin = np.ones_like(self.x, dtype=np.int)
            self.xPrc = None
            self.yPrc = None
            self.xPrcS = None
            self.yPrcS = None

        if self._gsmooth is True:
            aux = self.gaussian_smooth(**kwargs)
            self.xS = aux[0]
            self.yS = aux[1]
            self.xPrcS = aux[2]
            self.yPrcS = aux[3]

    def rstats_yx(self, **kwargs):
        ny = len(self.y)
        nBox = kwargs.get('nBox', ny * kwargs.get('frac', 0.1))
        if ny > nBox:
            aux = calc_running_stats(self.y, self.x, **kwargs)
            self.inv_xbin = aux[0]
            self.inv_xbinCenter = aux[0]
            self.inv_xMedian = aux[1]
            self.inv_xMean = aux[2]
            self.inv_xStd = aux[3]
            self.inv_yMedian = aux[4]
            self.inv_yMean = aux[5]
            self.inv_yStd = aux[6]
            self.inv_nInBin = aux[7]
            self.inv_xPrc = aux[8]
            self.inv_yPrc = aux[9]
            self.inv_xPrcS = []
            self.inv_yPrcS = []
            if self._tendency is True:
                aux = self.tendency(self.y, self.x, **kwargs)
                self.inv_xT = aux[0]
                self.inv_yT = aux[1]
                self.inv_xbin = aux[2]
                self.inv_spline = aux[3]
        else:
            self.inv_xbinCenter = self.x
            self.inv_xMedian = self.x
            self.inv_xMean = self.x
            self.inv_xStd = self.x
            self.inv_yMedian = self.y
            self.inv_yMean = self.y
            self.inv_yStd = self.y
            self.inv_nInBin = np.ones_like(self.y, dtype=np.int)
            self.inv_xPrc = -1
            self.inv_yPrc = -1
            self.inv_xPrcS = -1
            self.inv_yPrcS = -1

        if self._gsmooth is True:
            aux = self.gaussian_smooth(**kwargs)
            self.inv_xS = aux[0]
            self.inv_yS = aux[1]
            self.inv_xPrcS = aux[2]
            self.inv_yPrcS = aux[3]

    def gaussian_smooth(self, **kwargs):
        xPrcS = []
        yPrcS = []
        if self.sigma is None:
            self.sigma = self.y.std()
        self.sigma = kwargs.get('sigma', self.sigma)
        xM = np.ma.masked_array(self.xMedian)
        yM = np.ma.masked_array(self.yMedian)
        m_gs = np.isnan(xM) | np.isnan(yM)
        # self.xS = gaussian_filter1d(xM[~m_gs], self.sigma)
        xS = self.xMedian[~m_gs]
        yS = gaussian_filter1d(yM[~m_gs], self.sigma)
        # print '>X>X>X>', len(self.xMedian[~m_gs]), len(self.xS)
        if kwargs.get('gs_prc', None) is not None:
            for i in xrange(len(self.xPrc)):
                xM = np.ma.masked_array(self.xPrc[i])
                yM = np.ma.masked_array(self.yPrc[i])
                m_gs = np.isnan(xM) | np.isnan(yM)
                # self.xS = gaussian_filter1d(xM[~m_gs], self.sigma)
                xPrcS.append(self.xPrc[i][~m_gs])
                yPrcS.append(gaussian_filter1d(yM[~m_gs], self.sigma))
        return xS, yS, xPrcS, yPrcS

    def OLS_bisector(self):
        a, b, sa, sb = OLS_bisector(self.x, self.y)
        self.OLS_slope = a
        self.OLS_intercept = b
        self.OLS_slope_sigma = sa
        self.OLS_intercept_sigma = sb
        a, b, sa, sb = OLS_bisector(self.xS, self.yS)
        self.OLS_median_slope = a
        self.OLS_median_intercept = b
        self.OLS_median_slope_sigma = sa
        self.OLS_median_intercept_sigma = sb

    def poly1d(self):
        p = np.polyfit(self.x, self.y, 1)
        slope, intercept = p
        self.poly1d_slope = slope
        self.poly1d_intercept = intercept
        p = np.polyfit(self.xS, self.yS, 1)
        slope, intercept = p
        self.poly1d_median_slope = slope
        self.poly1d_median_intercept = intercept

    def tendency(self, x, y, xbin=None, **kwargs):
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(self.x, self.y)
        if isinstance(x, np.ma.core.MaskedArray) or isinstance(y, np.ma.core.MaskedArray):
            xm, ym = ma_mask_xyz(x=x, y=y)
            x = xm.compressed()
            y = ym.compressed()
        if xbin is None:
            nx = len(x)
            ind_xs = np.argsort(x)
            xS = x[ind_xs]
            nx = len(x)
            frac = kwargs.get('frac', 0.1)
            minimal_bin_points = kwargs.get('min_np', nx * frac)
            i = 0
            xbin = []
            xbin.append(xS[0])
            while i < nx:
                to_i = i + minimal_bin_points
                delta = (nx - to_i)
                miss_frac = 1. * delta / nx
                if to_i < nx and miss_frac >= frac:
                    xbin.append(xS[to_i])
                else:
                    to_i = nx
                    xbin.append(xS[-1])
                i = to_i
        xT = xbin
        yT = spline(xT)
        return xT, yT, xbin, spline
