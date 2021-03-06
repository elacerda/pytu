#
# Lacerda@UFSC - 30/Ago/2016
#
import numpy as np


def pickle(filename, data, protocol=None):
    import pickle

    if protocol is None:
        protocol = 2

    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol)


def unpickle(filename):
    import pickle

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def send_gmail(sender='dhubax@gmail.com', receivers=[], message=None, login=None, password=None, password_file=None):
    from smtplib import SMTP
    from subprocess import PIPE, Popen
    if login is None:
        login = sender
    if password is None:
        if not (password_file is None):
            o = Popen('cat %s' % password_file, shell=True, stdout=PIPE).stdout
            password = o.read().strip()
    try:
        server = SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(login, password)
        D = server.sendmail(sender, receivers, message)
        print('successfully sent the mail')
        return D
    except:
        print('failed to send mail')


def create_equal_N_bins(x, frac=0.1, x_sorted=None, min_np=None):
    xS = x_sorted
    if xS is None:
        ind_xs = np.argsort(x)
        xS = x[ind_xs]
    xbin = [xS[0]]
    N = len(xS)
    if min_np is None:
        min_np = frac * N
    i = 0
    min_next_i = int(np.ceil(min_np))
    next_i = min_next_i
    while i < N:
        to_i = i + next_i
        delta = (N - to_i)
        miss_frac = 1. * delta / N
        #print(to_i, int(to_i), xS[to_i], xS[int(to_i)])
        if to_i < N:
            if (xS[to_i] != xbin[-1]) and (miss_frac >= frac):
                xbin.append(xS[to_i])
                next_i = min_next_i
            else:
                next_i += 1
        else:
            #### last bin will be the xS.max()
            to_i = N
            xbin.append(xS[-1])
        i = to_i
    return np.asarray(xbin)


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
    i = 0
    xbin = kwargs.get('xbin', [])
    if xbin == []:
        xbin = create_equal_N_bins(x=x, x_sorted=xS,
                                   frac=kwargs.get('frac', 0.1),
                                   min_np=kwargs.get('min_np', None))
    nxbin = len(xbin)
    debug_var(debug, xbin=xbin, n_xbin=nxbin)
    # Reset in-bin stats arrays
    xbinCenter_out, xbin_out, nInBin_out = [], [], []
    xMedian_out, xMean_out, xStd_out, xPrc_out = [], [], [], []
    yMedian_out, yMean_out, yStd_out, yPrc_out = [], [], [], []
    ixBin = 0
    while ixBin < (nxbin - 1):
        left = xbin[ixBin]
        xbin_out.append(left)
        right = xbin[ixBin + 1]
        # isInBin = np.bitwise_and(np.greater_equal(x, left), np.less(x, right))
        isInBin = (x >= left) & (x < right)
        Np = isInBin.sum()
        nInBin_out.append(Np)
        xx, yy = x[isInBin], y[isInBin]
        # print(ixBin, Np, xx, yy)
        center = (right + left) / 2.
        xbin_out.append(right)
        xbinCenter_out.append(center)
        if Np > 0:
            xMedian_out.append(np.median(xx))
            xMean_out.append(xx.mean())
            xStd_out.append(xx.std())
            yMedian_out.append(np.median(yy))
            yMean_out.append(yy.mean())
            yStd_out.append(yy.std())
            if Np > 1:
                xPrc_out.append(np.percentile(xx, [5, 16, 84, 95]))
                yPrc_out.append(np.percentile(yy, [5, 16, 84, 95]))
            else:  # Np == 1
                if len(xPrc_out) > 0:
                    xPrc_out.append(xPrc_out[-1])
                    yPrc_out.append(xPrc_out[-1])
                else:
                    xPrc = np.median(xx)
                    yPrc = np.median(yy)
                    xPrc_out.append(np.asarray([xPrc, xPrc, xPrc, xPrc]))
                    yPrc_out.append(np.asarray([yPrc, yPrc, yPrc, yPrc]))
        else:  # Np == 0
            if len(xMedian_out) > 0:  # if there is some value, repeat it.
                xMedian_out.append(xMedian_out[-1])
                xMean_out.append(xMean_out[-1])
                xStd_out.append(xStd_out[-1])
                yMedian_out.append(yMedian_out[-1])
                yMean_out.append(yMean_out[-1])
                yStd_out.append(yStd_out[-1])
            else:  # This should be None
                xMedian_out.append(np.nan)
                xMean_out.append(np.nan)
                xStd_out.append(np.nan)
                yMedian_out.append(np.nan)
                yMean_out.append(np.nan)
                yStd_out.append(np.nan)
                # xMedian_out.append(0.)
                # xMean_out.append(0.)
                # xStd_out.append(0.)
                # yMedian_out.append(0.)
                # yMean_out.append(0.)
                # yStd_out.append(0.)
            if len(xPrc_out) > 0:
                xPrc_out.append(xPrc_out[-1])
                yPrc_out.append(xPrc_out[-1])
            else:
                xPrc_out.append(np.asarray([np.nan, np.nan, np.nan, np.nan]))
                yPrc_out.append(np.asarray([np.nan, np.nan, np.nan, np.nan]))
        ixBin += 1
    debug_var(
        debug,
        xbinCenter_out = np.array(xbinCenter_out),
        xMedian_out = np.array(xMedian_out),
        yMedian_out = np.array(yMedian_out),
        nInBin_out = nInBin_out,
    )
    if kwargs.get('return_ma', False):
        return xbin, \
            np.ma.masked_array(xbinCenter_out, mask=np.isnan(xbinCenter_out)), \
            np.ma.masked_array(xMedian_out, mask=np.isnan(xMedian_out)), \
            np.ma.masked_array(xMean_out, mask=np.isnan(xMean_out)), \
            np.ma.masked_array(xStd_out, mask=np.isnan(xStd_out)), \
            np.ma.masked_array(yMedian_out, mask=np.isnan(yMedian_out)), \
            np.ma.masked_array(yMean_out, mask=np.isnan(yMean_out)), \
            np.ma.masked_array(yStd_out, mask=np.isnan(yStd_out)), \
            np.ma.masked_array(nInBin_out, mask=np.isnan(nInBin_out)), \
            np.ma.masked_array(xPrc_out, mask=np.isnan(xPrc_out)).T, \
            np.ma.masked_array(yPrc_out, mask=np.isnan(yPrc_out)).T
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
    verbose_level = kwargs.pop('verbose_level', 0)
    pref = '\t' * verbose_level + pref
    if debug_mode:
        for kw, vw in kwargs.items():
            if isinstance(vw, dict):
                print('%s') % pref, kw
                for k, v in vw.items():
                    print('\t%s' % pref, k, ':\t', v)
            else:
                print('%s' % pref, '%s:\t' % kw, vw)


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
            xm = np.ma.masked_array(x, mask=m, dtype='float', copy=True)
            ym = np.ma.masked_array(y, mask=m, dtype='float', copy=True)
            zm = np.ma.masked_array(z, mask=m, dtype='float', copy=True)
            return xm, ym, zm
        else:
            xm = np.ma.masked_array(x, mask=m, dtype='float', copy=True)
            ym = np.ma.masked_array(y, mask=m, dtype='float', copy=True)
            return xm, ym
    xm = np.ma.masked_array(x, mask=m, dtype='float', copy=True)
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


def analytic_linreg(x, y, dy, print_output=True):
    sx = (x/dy**2).sum()
    sx2 = (x**2/dy**2).sum()
    s1 = (1./dy**2).sum()
    sy = (y/dy**2).sum()
    sxy = (x*y/dy**2).sum()
    q = sx2*s1-sx**2
    slope = (s1*sxy-sx*sy)/q
    intercept = (sx2*sy-sx*sxy)/q
    p = [slope, intercept]
    e_slope = (s1/q)**0.5
    e_intercept = (sx2/q)**0.5
    ep = [e_slope, e_intercept]
    sigma = (y - np.polyval(p, x)).std()
    if print_output:
        print(r's:{:.5f}+/-{:.5f} i:{:.5f}+/-{:.5f} stddev={:.5f}'.format(p[0], ep[0], p[1], ep[1], sigma))
    return [slope, intercept], [e_slope, e_intercept], sigma


def my_WLS(x, y, sy, ddof=1, return_all=True):
    tmp = np.ones(x.size)
    Nred = (x.size-ddof)
    I = np.diag(tmp)
    X = np.vstack([tmp, x]).T
    W = np.diag(1/sy**2)
    M = np.linalg.inv(X.T.dot(W).dot(X))
    A = M.dot(X.T).dot(W).dot(y)
    if return_all:
        H = X.dot(M).dot(X.T).dot(W)
        resids = (I - H).dot(y)
        sigma = np.sqrt((resid**2).sum()/Nred)
        rchisq = resid.T.dot(W).dot(resid)/Nred
        return A, sigma, resids, rchisq
    return A


def my_ODR(x, y, sx, sy, beta0, model, return_output=False, print_output=True):
    model = odr.Model(model)
    first_guess = beta0
    output = odr.ODR(odr.RealData(x, y, sx=ex, sy=ey), model, beta0=first_guess).run()
    p, ep = output.beta.tolist(), output.sd_beta.tolist()
    sigma_odr = ((output.delta**2 + output.eps**2)**.5).std(ddof=2)
    sigma_y_odr = sigma_odr/(np.cos(np.arctan(p[0])))  # projection of the sigma_odr in the y_axis
    if print_output:
        print(r's:{:.5f}+/-{:.5f} i:{:.5f}+/-{:.5f} stddev_odr={:.5f} stddev_y={:.5f}'.format(p[0], ep[0], p[1], ep[1], sigma_odr, sigma_y_odr))
    if return_output:
        return p, ep, sigma_odr, sigma_y_odr, output
    else:
        return p, ep, sigma_odr, sigma_y_odr


def ODR_MC(x, y, sx, sy, beta0):
    p, _, _, _ = my_ODR(x, y, sx, sy, beta0, np.polyval, False, False)
    return p


def analytic_linreg_MC(x, y, sx, sy, beta0):
    p, _, _ = analytic_linreg(x, y, ey, False)
    return p


def my_WLS_MC(x, y, sx, sy, beta0):
    p = my_WLS(x, y, sy, ddof=2, return_all=False)
    return p


def linreg_MonteCarlo(func_linreg, x, y, sx, sy, p, tries=100, beta0=None):
    _p = []
    for i in range(tries):
        xdist = np.random.normal(x, sx)
        ydist = np.random.normal(np.polyval(p, x), sy)
        _p.append(func_linreg(xdist, ydist, sx, sy, p))
    _p = np.asarray(_p)
    print(_p.mean(axis=0), _p.std(axis=0), _p.max(axis=0), _p.min(axis=0))
