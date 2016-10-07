#
# Lacerda@UFSC - 30/Ago/2016
#
import numpy as np


def calc_running_stats(x, y, **kwargs):
    '''
    Statistics of x & y with a minimal (floor limit) number of points in x
    Note: we have overlapping boxes.. so running stats..

    XXX Lacerda@IAA - masked array mess with the method
    '''

    debug = kwargs.get('debug', False)
    if isinstance(x, np.ma.core.MaskedArray) or isinstance(y, np.ma.core.MaskedArray):
        xm, ym = ma_mask_xyz(x=x, y=y)
        x = xm.compressed()
        y = ym.compressed()
    ind_xs = np.argsort(x)
    xS = x[ind_xs]
    nx = len(x)
    frac = kwargs.get('frac', 0.1)
    minimal_bin_points = kwargs.get('min_np', nx * frac)
    i = 0
    xbin = kwargs.get('xbin', [])
    debug_var(debug, xbin=xbin)
    if xbin == []:
        xbin.append(xS[0])
        min_next_i = int(np.ceil(minimal_bin_points))
        next_i = min_next_i
        while i < nx:
            to_i = i + next_i
            delta = (nx - to_i)
            miss_frac = 1. * delta / nx
            # print to_i, int(to_i), xS[to_i], xS[int(to_i)]
            if to_i < nx:
                if (xS[to_i] != xbin[-1]) and (miss_frac >= frac):
                    xbin.append(xS[to_i])
                    next_i = min_next_i
                else:
                    next_i += 1
            else:
                # last bin will be the xS.max()
                to_i = nx
                xbin.append(xS[-1])
            i = to_i
    # Def x-bins
    xbin = np.asarray(xbin)
    nxbin = len(xbin)
    debug_var(debug,
              minimal_bin_points=minimal_bin_points,
              xbin=xbin,
              n_xbin=nxbin)
    # Reset in-bin stats arrays
    xbinCenter_out = []
    xbin_out = []
    xMedian_out = []
    xMean_out = []
    xStd_out = []
    yMedian_out = []
    yMean_out = []
    yStd_out = []
    xPrc_out = []
    yPrc_out = []
    nInBin_out = []
    ixBin = 0
    while ixBin < (nxbin - 1):
        left = xbin[ixBin]
        xbin_out.append(left)
        right = xbin[ixBin + 1]
        isInBin = np.bitwise_and(np.greater_equal(x, left), np.less(x, right))
        xx , yy = x[isInBin] , y[isInBin]
        center = (right + left) / 2.
        xbin_out.append(right)
        xbinCenter_out.append(center)
        Np = isInBin.astype(np.int).sum()
        nInBin_out.append(Np)
        if Np >= 2:
            xMedian_out.append(np.median(xx))
            xMean_out.append(xx.mean())
            xStd_out.append(xx.std())
            yMedian_out.append(np.median(yy))
            yMean_out.append(yy.mean())
            yStd_out.append(yy.std())
            xPrc_out.append(np.percentile(xx, [5, 16, 84, 95]))
            yPrc_out.append(np.percentile(yy, [5, 16, 84, 95]))
        else:
            if len(xMedian_out) > 0:
                xMedian_out.append(xMedian_out[-1])
                xMean_out.append(xMean_out[-1])
                xStd_out.append(xStd_out[-1])
                yMedian_out.append(yMedian_out[-1])
                yMean_out.append(yMean_out[-1])
                yStd_out.append(yStd_out[-1])
            else:
                xMedian_out.append(0.)
                xMean_out.append(0.)
                xStd_out.append(0.)
                yMedian_out.append(0.)
                yMean_out.append(0.)
                yStd_out.append(0.)
            if len(xPrc_out) > 0:
                xPrc_out.append(xPrc_out[-1])
                yPrc_out.append(xPrc_out[-1])
            else:
                xPrc_out.append(np.asarray([0., 0., 0., 0.]))
                yPrc_out.append(np.asarray([0., 0., 0., 0.]))
        ixBin += 1
    debug_var(debug,
              xbinCenter_out=np.array(xbinCenter_out),
              xMedian_out=np.array(xMedian_out),
              nInBin_out=nInBin_out,
    )
    return xbin, \
        np.array(xbinCenter_out), np.array(xMedian_out), np.array(xMean_out), np.array(xStd_out), \
        np.array(yMedian_out), np.array(yMean_out), np.array(yStd_out), np.array(nInBin_out), \
        np.array(xPrc_out).T, np.array(yPrc_out).T


def PCA(arr, reduced=False, arr_mean=False, arr_std=False, sort=True):
    '''
    ARR array must have shape (measurements, variables)

    If you are looking for PCA of spectra each wavelength is a variable.

    reduced = True:
        each var = (var - var.mean()) / var.std()
    '''
    from scipy.linalg import eigh
    arr__mv = arr
    N_measurements, N_vars = arr__mv.shape
    if not arr_mean or not arrMean.any():
        arr_mean__v = arr.mean(axis=0)
    else:
        arr_mean__v = arrMean
    if not reduced:
        arr_std__v = None
        deviance_mean__mv = arr__mv - arr_mean__v
    else:
        if not arr_std or not arr_std.any():
            arr_std__v = arr.std(axis=0)
        else:
            arr_std__v = arr_std
        deviance_mean__mv = np.asarray([diff / arr_std__v for diff in (arr__mv - arr_mean__v)])
    covariance_matrix__vv = (deviance_mean__mv.T).dot(deviance_mean__mv) / (N_vars - 1)
    eigen_values__e, eigen_vectors__ve = eigh(covariance_matrix__vv)
    eigen_values_sorted__e = eigen_values__e
    eigen_vectors_sorted__ve = eigen_vectors__ve
    if sort:
        S = np.argsort(eigen_values__e)[::-1]
        eigen_values_sorted__e = eigen_values__e[S]
        eigen_vectors_sorted__ve = eigen_vectors__ve[:, S]
    return deviance_mean__mv, arr_mean__v, arr_std__v, covariance_matrix__vv, eigen_values_sorted__e, eigen_vectors_sorted__ve

def debug_var(debug_mode=False, **kwargs):
    pref = kwargs.pop('pref', '>>>')
    if debug_mode:
        for kw, vw in kwargs.iteritems():
            if isinstance(vw, dict):
                print '%s' % pref, kw
                for k, v in vw.iteritems():
                    print '\t%s' % pref, k, ':\t', v
            else:
                print '%s' % pref, '%s:\t' % kw, vw


def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level


def OLS_bisector(x, y, **kwargs):
    xdev = x - x.mean()
    ydev = y - y.mean()
    Sxx = (xdev ** 2.0).sum()
    Syy = (ydev ** 2.0).sum()
    Sxy = (xdev * ydev).sum()
    b1 = Sxy / Sxx
    b2 = Syy / Sxy
    var1 = 1. / Sxx ** 2.
    var1 *= (xdev ** 2.0 * (ydev - b1 * xdev) ** 2.0).sum()
    var2 = 1. / Sxy ** 2.
    var2 *= (ydev ** 2.0 * (ydev - b2 * xdev) ** 2.0).sum()
    cov12 = 1. / (b1 * Sxx ** 2.0)
    cov12 *= (xdev * ydev * (ydev - b1 * ydev) * (ydev - b2 * ydev)).sum()
    bb1 = (1 + b1 ** 2.)
    bb2 = (1 + b2 ** 2.)
    b3 = 1. / (b1 + b2) * (b1 * b2 - 1 + (bb1 * bb2) ** .5)
    var = b3 ** 2.0 / ((b1 + b2) ** 2.0 * bb1 * bb2)
    var *= (bb2 ** 2.0 * var1 + 2. * bb1 * bb2 * cov12 + bb1 ** 2. * var2)
    slope = b3
    intercept = y.mean() - slope * x.mean()
    var_slope = var
    try:
        n = x.count()
    except AttributeError:
        n = len(x)
    gamma1 = b3 / ((b1 + b2) * (bb1 * bb2) ** 0.5)
    gamma13 = gamma1 * bb2
    gamma23 = gamma1 * bb1
    var_intercept = 1. / n ** 2.0
    var_intercept *= ((ydev - b3 * xdev - n * x.mean() * (gamma13 / Sxx * xdev * (ydev - b1 * xdev) + gamma23 / Sxy * ydev * (ydev - b2 * xdev))) ** 2.0).sum()
    sigma_slope = var_slope ** 0.5
    sigma_intercept = var_intercept ** 0.5
    debug_var(kwargs.get('debug', False), slope=slope,
              intercept=intercept, sigma_slope=sigma_slope,
              sigma_intercept=sigma_intercept)
    return slope, intercept, sigma_slope, sigma_intercept


def ma_mask_xyz(x, y=None, z=None, mask=None):
    m = np.bitwise_or(np.isnan(x), np.isinf(x))
    if mask is not None:
        m = np.bitwise_or(m, np.asarray(mask))
    if isinstance(x, np.ma.core.MaskedArray):
        m = np.bitwise_or(m, x.mask)
    if y is not None:
        m_y = np.bitwise_or(np.isnan(y), np.isinf(y))
        m = np.bitwise_or(m, m_y)
        if isinstance(y, np.ma.core.MaskedArray):
            m = np.bitwise_or(m, y.mask)
        if z is not None:
            m_z = np.bitwise_or(np.isnan(z), np.isinf(z))
            m = np.bitwise_or(m, m_z)
            if isinstance(z, np.ma.core.MaskedArray):
                m = np.bitwise_or(m, z.mask)
            xm = np.ma.masked_array(x, mask=m, dtype='float')
            ym = np.ma.masked_array(y, mask=m, dtype='float')
            zm = np.ma.masked_array(z, mask=m, dtype='float')
            return xm, ym, zm
        else:
            xm = np.ma.masked_array(x, mask=m, dtype='float')
            ym = np.ma.masked_array(y, mask=m, dtype='float')
            return xm, ym
    xm = np.ma.masked_array(x, mask=m, dtype='float')
    return xm


def gaussSmooth_YofX(x, y, FWHM):
    '''
    Sloppy function to return the gaussian-smoothed version of an y(x) relation.
    Cid@Lagoa - 07/June/2014
    '''
    sig = FWHM / np.sqrt(8. * np.log(2.))
    xS, yS = np.zeros_like(x), np.zeros_like(x)
    w__ij = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        # for j in np.arange(len(x)):
        #     w__ij[i,j] = np.exp( -0.5 * ((x[j] - x[i]) / sig)**2  )
        w__ij[i, :] = np.exp(-0.5 * ((x - x[i]) / sig) ** 2)
        w__ij[i, :] = w__ij[i, :] / w__ij[i, :].sum()
        xS[i] = (w__ij[i, :] * x).sum()
        yS[i] = (w__ij[i, :] * y).sum()
    return xS, yS


def calcYofXStats_EqNumberBins(x, y, nPerBin = 25):
    '''
    This gives the statistics of y(x) for x-bins of variable width, but all containing
    the same number of points.
    We 1st sort x, and the y accordingly. Then we compute the median, mean and std
    of x & y in contiguous x-bins in x defined to have nPerBin points each

    Cid@Lagoa - 05/June/2014
    '''
    ind_sx = np.argsort(x)
    xS, yS = x[ind_sx], y[ind_sx]
    Nbins = len(x) - nPerBin + 1
    xMedian, xMean, xStd = np.zeros(Nbins), np.zeros(Nbins), np.zeros(Nbins)
    yMedian, yMean, yStd = np.zeros(Nbins), np.zeros(Nbins), np.zeros(Nbins)
    nInBin = np.zeros(Nbins)
    for ixBin in np.arange(0, Nbins):
        xx, yy = xS[ixBin:ixBin + nPerBin], yS[ixBin:ixBin + nPerBin]
        xMedian[ixBin], xMean[ixBin], xStd[ixBin] = np.median(xx), xx.mean(), xx.std()
        yMedian[ixBin], yMean[ixBin], yStd[ixBin] = np.median(yy), yy.mean(), yy.std()
        nInBin[ixBin] = len(xx)
    return xMedian, xMean, xStd, yMedian, yMean, yStd, nInBin


def create_dx(x):
    dx = np.empty_like(x)
    # dl/2 from right neighbor
    dx[1:] = (x[1:] - x[:-1]) / 2.
    # dl/2 from left neighbor
    dx[:-1] += dx[1:]
    dx[0] = 2. * dx[0]
    dx[-1] = 2. * dx[-1]
    # dx[-1] = x[-1]
    return dx


def linearInterpol(x1, x2, y1, y2, x):
    '''
    Let S be the matrix:
        S = |x x1 x2|
            |y y1 y2|
    Now we do:
        DET(S) = 0,
    to find the linear equation between the points (x1, y1) and (x2, y2).
    Hence we find the general equation Ax + By + C = 0 where:
        A = (y1 - y2)
        B = (x2 - x1)
        C = x1y2 - x2y1
    '''
    return (x2 * y1 - x1 * y2 - x * (y1 - y2)) / (x2 - x1)


def calc_agebins(ages, age=None):
    # Define ranges for age-bins
    # TODO: This age-bin-edges thing could be made more elegant & general.
    aCen__t = ages
    aLow__t = np.empty_like(ages)
    aUpp__t = np.empty_like(ages)
    aLow__t[0] = 0.
    aLow__t[1:] = (aCen__t[1:] + aCen__t[:-1]) / 2.
    aUpp__t[:-1] = aLow__t[1:]
    aUpp__t[-1] = aCen__t[-1]
    # Find index of age-bin corresponding to the last bin fully within < tSF
    age_index = -1
    if age is not None:
        age_index = np.where(aLow__t < age)[0][-1]
    return aCen__t, aLow__t, aUpp__t, age_index
