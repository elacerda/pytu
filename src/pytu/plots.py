#
# Lacerda@UFSC - 30/Ago/2016
#
import numpy as np
from matplotlib import pyplot as plt
from .functions import find_confidence_interval, debug_var


def add_subplot_axes(ax, rect, axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height], axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)

    return subax


def density_contour(xdata, ydata, binsx, binsy, ax=None, levels_confidence=[0.68, 0.95, 0.99], range=None, **contour_kwargs):
    """ Create a density contour plot.
    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    binsx : int
        Number of bins along x dimension
    binsy : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """
    import scipy.optimize as so
    # nbins_x = len(binsx) - 1
    # nbins_y = len(binsy) - 1
    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=[binsx, binsy], range=range, normed=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1])
    y_bin_sizes = (yedges[1:] - yedges[:-1])
    pdf = (H * (x_bin_sizes * y_bin_sizes))
    levels = [so.brentq(find_confidence_interval, 0., 1., args=(pdf, lvl)) for lvl in levels_confidence]
    # one_sigma = so.brentq(find_confidence_interval, 0., 1., args = (pdf, 0.68))
    # two_sigma = so.brentq(find_confidence_interval, 0., 1., args = (pdf, 0.95))
    # three_sigma = so.brentq(find_confidence_interval, 0., 1., args = (pdf, 0.99))
    # levels = [one_sigma, two_sigma, three_sigma]
    X, Y = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
    Z = pdf.T
    if ax is None:
        contour = plt.contour(X, Y, Z, levels=levels[::-1], origin="lower", **contour_kwargs)
    else:
        # contour = ax.contour(X, Y, Z, levels=levels, origin="lower", **contour_kwargs)
        contour = ax.contour(X, Y, Z, levels=levels[::-1], origin="lower", **contour_kwargs)
    return contour


def plot_spearmanr_ax(ax, x, y, pos_x, pos_y, fontsize):
    from scipy.stats import spearmanr
    rhoSpearman, pvalSpearman = spearmanr(x, y)
    txt = '<y/x>:%.3f - (y/x) median:%.3f - $\sigma(y/x)$:%.3f - Rs: %.2f' % (np.mean(y / x), np.ma.median((y / x)), np.ma.std(y / x), rhoSpearman)
    plot_text_ax(ax, txt, pos_x, pos_y, fontsize, 'top', 'left')


def plot_OLSbisector_ax(ax, x, y, **kwargs):
    pos_x = kwargs.get('pos_x', 0.99)
    pos_y = kwargs.get('pos_y', 0.00)
    fontsize = kwargs.get('fontsize', kwargs.get('fs', 10))
    color = kwargs.get('color', kwargs.get('c', 'r'))
    rms = kwargs.get('rms', True)
    txt = kwargs.get('text', True)
    kwargs_plot = dict(c=color, ls='-', lw=1.5, label='')
    kwargs_plot.update(kwargs.get('kwargs_plot', {}))
    label = kwargs_plot['label']
    x_rms = kwargs.get('x_rms', x)
    y_rms = kwargs.get('y_rms', y)
    OLS = kwargs.get('OLS', None)
    va = kwargs.get('verticalalignment', kwargs.get('va', 'bottom'))
    ha = kwargs.get('horizontalalignment', kwargs.get('ha', 'right'))
    plotOLS = kwargs.get('plotOLS', True)
    if OLS is None:
        a, b, sigma_a, sigma_b = OLS_bisector(x, y)
    else:
        a = x
        b = y
        sigma_a = None
        sigma_b = sigma_a
    Yrms_str = ''
    if rms:
        Yrms = (y_rms - (a * x_rms + b)).std()
        Yrms_str = r'(rms:%.3f)' % Yrms
    if plotOLS:
        ax.plot(ax.get_xlim(), a * np.asarray(ax.get_xlim()) + b, **kwargs_plot)
    if b > 0:
        txt_y = r'$y_{OLS}$ = %.2f$x$ + %.2f %s' % (a, b, Yrms_str)
    else:
        txt_y = r'$y_{OLS}$ = %.2f$x$ - %.2f %s' % (a, b * -1., Yrms_str)
    debug_var(True, y_OLS=txt_y)
    if txt:
        txt_y = '%s (%.3f, %.3f, %.3f)' % (label, a, b, Yrms)
        plot_text_ax(ax, txt_y, pos_x, pos_y, fontsize, va, ha, color=color)
    else:
        print txt_y
    return a, b, sigma_a, sigma_b


def plot_text_ax(ax, txt, xpos=0.99, ypos=0.01, fontsize=10, va='bottom', ha='right', color='k', **kwargs):
    xpos = kwargs.get('pos_x', xpos)
    ypos = kwargs.get('pos_y', ypos)
    fontsize = kwargs.get('fontsize', kwargs.get('fs', fontsize))
    va = kwargs.get('verticalalignment', kwargs.get('va', va))
    ha = kwargs.get('horizontalalignment', kwargs.get('ha', ha))
    color = kwargs.get('color', kwargs.get('c', color))
    alpha = kwargs.get('alpha', 1.)
    textbox = dict(boxstyle='round', facecolor='wheat', alpha=0.)
    transform = kwargs.get('transform', True)
    rot = kwargs.get('rotation', 0)
    if transform is True:
        ax.text(xpos, ypos, txt, fontsize=fontsize, color=color,
                transform=ax.transAxes,
                verticalalignment=va, horizontalalignment=ha,
                bbox=textbox, alpha=alpha, rotation=rot)
    else:
        ax.text(xpos, ypos, txt, fontsize=fontsize, color=color,
                verticalalignment=va, horizontalalignment=ha,
                bbox=textbox, alpha=alpha, rotation=rot)


def plot_histo_ax(ax, x, **kwargs):
    c = kwargs.get('color', kwargs.get('c', 'b'))
    first = kwargs.get('first', False)
    va = kwargs.get('verticalalignment', kwargs.get('va', 'top'))
    ha = kwargs.get('verticalalignment', kwargs.get('ha', 'right'))
    fs = kwargs.get('fontsize', kwargs.get('fs', 14))
    pos_x = kwargs.get('pos_x', 0.98)
    bins = kwargs.get('bins', 30)
    range = kwargs.get('range', None)
    kwargs_histo = kwargs.get('kwargs_histo', dict(bins=bins, range=range,
                              color=c, align='mid', alpha=0.6,
                              histtype='stepfilled', normed=True))
    ax.hist(x, **kwargs_histo)
    pos_y = [0.96, 0.88, 0.80, 0.72, 0.64]
    if first:
        txt = [r'$<x>$: %.2f' % np.mean(x), r'med($x$): %.2f' % np.median(x),
               r'$\sigma(x)$: %.2f' % np.std(x), r'max$(x)$: %.2f' % np.max(x),
               r'min$(x)$: %.2f' % np.min(x)]
    else:
        txt = ['%.2f' % np.mean(x), '%.2f' % np.median(x), '%.2f' % np.std(x),
               '%.2f' % np.max(x), '%.2f' % np.min(x)]
    for i, pos in enumerate(pos_y):
        plot_text_ax(ax, txt[i], **dict(pos_x=pos_x, pos_y=pos, fs=fs, va=va, ha=ha, c=c))
    return ax


def next_row_col(row, col, N_rows, N_cols):
    if col == (N_cols - 1):
        col = 0
        row += 1
        if (row == N_rows):
            row = 0
    else:
        col += 1
    return row, col


def plot_percentiles_ax(ax, x, y, **kwargs):
    median_kwargs = kwargs.pop('median_kwargs', kwargs)
    ax.plot(x, np.median(y), **median_kwargs)
    ax.plot(x, np.percentile(y, 5), **kwargs)
    ax.plot(x, np.percentile(y, 16), **kwargs)
    ax.plot(x, np.percentile(y, 84), **kwargs)
    ax.plot(x, np.percentile(y, 95), **kwargs)
