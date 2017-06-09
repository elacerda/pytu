#
# Lacerda@UFSC - 30/Ago/2016
#
import numpy as np
import matplotlib as mpl
from .lines import Lines
from .functions import debug_var
from matplotlib import pyplot as plt
from .functions import find_confidence_interval


def stats_med12sigma(x, y, bin_edges, prc=[5, 16, 50, 84, 95]):
    bin_edges = np.array(bin_edges)
    bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    N_R = len(bin_center)
    yMean = np.ma.masked_all(bin_center.shape, dtype='float')
    npts = np.zeros(bin_center.shape, dtype='int')
    prc_stats = []
    for iR in range(N_R):
        left = bin_edges[iR]
        right = bin_edges[iR + 1]
        msk = np.bitwise_and(np.greater(x, left), np.less_equal(x, right))
        if msk.astype(int).sum():
            yTmp = y[msk]
            yMean[iR] = np.mean(yTmp)
            npts[iR] = len(yTmp)
            if prc is not None:
                prc_stats.append(np.percentile(y[msk], prc))
            else:
                prc_stats.append(None)
        else:
            if prc is not None:
                prc_stats.append([np.nan for i in range(len(prc))])
            else:
                prc_stats.append(None)
        # print iR, prc_stats[-1], np.asarray(prc_stats).shape, np.asarray(prc_stats).T.shape
    return yMean, np.asarray(prc_stats).T, bin_center, npts


def cmap_discrete(colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)], n_bins=None, cmap_name='myMap'):
    if n_bins == None:
        n_bins = len(colors)
    cm = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cm


def plot_scatter_histo(x, y, xlim, ylim, xbins=30, ybins=30, xlabel='', ylabel='',
                       c=None, cmap=None, figure=None, axScatter=None, axHistx=None, axHisty=None,
                       scatter=True, histo=True, s=1, histtype='barstacked'):
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()  # no labels
    if axScatter is None:
        f = figure
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]
        axScatter = f.add_axes(rect_scatter)
        axHistx = f.add_axes(rect_histx)
        axHisty = f.add_axes(rect_histy)
        axHistx.xaxis.set_major_formatter(nullfmt)  # no labels
        axHisty.yaxis.set_major_formatter(nullfmt)  # no labels
    if scatter:
        if isinstance(x, list):
            for X, Y, C in zip(x, y, c):
                axScatter.scatter(X, Y, c=C, s=s, cmap=cmap, marker='o', edgecolor='none')
        else:
            print 'alow'
            axScatter.scatter(x, y, c=c, s=s, cmap=cmap, marker='o', edgecolor='none')
    axScatter.set_xlim(xlim)
    axScatter.set_ylim(ylim)
    axScatter.set_xlabel(xlabel)
    axScatter.set_ylabel(ylabel)
    if histo:
        axHistx.hist(x, bins=xbins, range=xlim, color=c, histtype=histtype)
        axHisty.hist(y, bins=ybins, range=ylim, orientation='horizontal', color=c, histtype=histtype)
    plt.setp(axHisty.xaxis.get_majorticklabels(), rotation=270)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    return axScatter, axHistx, axHisty


def plotWHAN(ax, N2Ha, WHa, z=None, cmap='viridis', mask=None, labels=True, N=False, cb_label=r'R [HLR]', vmax=None, vmin=None, dcontour=True):
    from .functions import ma_mask_xyz
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if mask is None:
        mask = np.zeros_like(N2Ha, dtype=np.bool_)
    extent = [-1.5, 0.5, -0.5, 3.]
    if z is None:
        bins = [30, 30]
        xm, ym = ma_mask_xyz(N2Ha, np.ma.log10(WHa), mask=mask)
        if dcontour:
            density_contour(xm.compressed(), ym.compressed(), bins[0], bins[1], ax, range=[extent[0:2], extent[2:4]], colors=['b', 'y', 'r'])
        sc = ax.scatter(xm, ym, marker='o', c='0.5', s=10, edgecolor='none', alpha=0.4)
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
    else:
        xm, ym, z = ma_mask_xyz(N2Ha, np.ma.log10(WHa), z, mask=mask)
        sc = ax.scatter(xm, ym, c=z, cmap=cmap, vmin=vmin, vmax=vmax, marker='o', s=10, edgecolor='none')
        ax.set_xlim(extent[0:   2])
        ax.set_ylim(extent[2:4])
        ax.set_aspect('equal', 'box')
        the_divider = make_axes_locatable(ax)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(sc, cax=color_axis)
        # cb = plt.colorbar(sc, ax=ax, ticks=[0, .5, 1, 1.5, 2, 2.5, 3], pad=0)
        cb.set_label(cb_label)
    if labels:
        xlabel = r'$\log [NII]/H\alpha$'
        ylabel = r'$\log WH\alpha$'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    if not N:
        N = xm.count()
    c = ''
    if (xm.compressed() < extent[0]).any():
        c += 'x-'
    if (xm.compressed() > extent[1]).any():
        c += 'x+'
    if (ym.compressed() < extent[2]).any():
        c += 'y-'
    if (ym.compressed() > extent[3]).any():
        c += 'y+'
    # plt.axis(extent)
    plt.axis(extent)
    plot_text_ax(ax, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
    ax.plot((-0.4, -0.4), (np.log10(3), 3), 'k-')
    ax.plot((-0.4, 0.5), np.ma.log10([6, 6]), 'k-')
    ax.axhline(y=np.log10(3), c='k')
    p = [np.log10(0.5/5.0), np.log10(0.5)]
    xini = (np.log10(3.) - p[1]) / p[0]
    ax.plot((xini, 0.), np.polyval(p, [xini, 0.]), 'k:')
    ax.plot((0, 0.5), np.log10([0.5, 0.5]), 'k:')
    ax.text(-1.4, 0.75, 'SF')
    ax.text(0.07, 0.9, 'sAGN')
    ax.text(0.05, 0.55, 'wAGN')
    ax.text(0.25, 0.0, 'RG')
    ax.text(-0.8, 0, 'PG')
    return ax


def plotBPT(ax, N2Ha, O3Hb, z=None, cmap='viridis', mask=None, labels=True, N=False, cb_label=r'R [HLR]', vmax=None, vmin=None, dcontour=True):
    from .functions import ma_mask_xyz
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if mask is None:
        mask = np.zeros_like(O3Hb, dtype=np.bool_)
    extent = [-1.5, 1, -1.5, 1.5]
    if z is None:
        bins = [30, 30]
        xm, ym = ma_mask_xyz(N2Ha, O3Hb, mask=mask)
        if dcontour:
            density_contour(xm.compressed(), ym.compressed(), bins[0], bins[1], ax, range=[extent[0:2], extent[2:4]], colors=['b', 'y', 'r'])
        sc = ax.scatter(xm, ym, marker='o', c='0.5', s=10, edgecolor='none', alpha=0.4)
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
    else:
        xm, ym, z = ma_mask_xyz(N2Ha, O3Hb, z, mask=mask)
        sc = ax.scatter(xm, ym, c=z, cmap=cmap, vmin=vmin, vmax=vmax, marker='o', s=10, edgecolor='none')
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])
        ax.set_aspect('equal', 'box')
        the_divider = make_axes_locatable(ax)
        color_axis = the_divider.append_axes('right', size='5%', pad=0)
        cb = plt.colorbar(sc, cax=color_axis)
        # cb = plt.colorbar(sc, ax=ax, ticks=[0, .5, 1, 1.5, 2, 2.5, 3], pad=0)
        cb.set_label(cb_label)
    if labels:
        ax.set_xlabel(r'$\log\ [NII]/H\alpha$')
        ax.set_ylabel(r'$\log\ [OIII]/H\beta$')
    L = Lines()
    if not N:
        N = xm.count()
    c = ''
    if (xm.compressed() < extent[0]).any():
        c += 'x-'
    if (xm.compressed() > extent[1]).any():
        c += 'x+'
    if (ym.compressed() < extent[2]).any():
        c += 'y-'
    if (ym.compressed() > extent[3]).any():
        c += 'y+'
    plot_text_ax(ax, '%d %s' % (N, c), 0.01, 0.99, 20, 'top', 'left', 'k')
    plot_text_ax(ax, 'S06', 0.30, 0.02, 20, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'K03', 0.53, 0.02, 20, 'bottom', 'left', 'k')
    plot_text_ax(ax, 'K01', 0.85, 0.02, 20, 'bottom', 'right', 'k')
    plot_text_ax(ax, 'CF10', 0.92, 0.98, 20, 'top', 'right', 'k', rotation=38)  # 44.62)
    ax.plot(L.x['S06'], L.y['S06'], 'k-', label='S06')
    ax.plot(L.x['K03'], L.y['K03'], 'k-', label='K03')
    ax.plot(L.x['K01'], L.y['K01'], 'k-', label='K01')
    ax.plot(L.x['CF10'], L.y['CF10'], 'k-', label='CF10')
    L.fixCF10('S06')
    return ax


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


def plot_spearmanr_ax(ax, x, y, pos_x=0.01, pos_y=0.99, fontsize=10, verticalalignment='top', horizontalalignment='left', more_stats=False):
    from scipy.stats import spearmanr
    rhoSpearman, pvalSpearman = spearmanr(x, y)
    if more_stats:
        txt = '<y/x>:%.3f - (y/x) median:%.3f - $\sigma(y/x)$:%.3f - Rs: %.2f' % (np.mean(y / x), np.ma.median((y / x)), np.ma.std(y / x), rhoSpearman)
    else:
        txt = 'Rs: %.2f' % (rhoSpearman)
    plot_text_ax(ax, txt, pos_x, pos_y, fontsize, verticalalignment, horizontalalignment)


def plot_OLSbisector_ax(ax, x, y, **kwargs):
    from .functions import OLS_bisector
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


def plot_histo_stats_txt(x, first=False, dataset_name=None):
    if first:
        txt = [r'$N(x)$: %d' % len(x),
               r'$<x>$: %.3f' % np.mean(x), r'med($x$): %.3f' % np.median(x),
               r'$\sigma(x)$: %.3f' % np.std(x), r'max$(x)$: %.3f' % np.max(x),
               r'min$(x)$: %.3f' % np.min(x)]
        first = False
    else:
        if len(x) > 0:
            txt = ['%d' % len(x), '%.3f' % np.mean(x), '%.3f' % np.median(x),
                   '%.3f' % np.std(x), '%.3f' % np.max(x), '%.3f' % np.min(x)]
        else:
            txt = ['0', '0', '0', '0', '0', '0']
    if dataset_name is not None:
        txt.insert(0, dataset_name)
    return txt, first


def plot_histo_ax(ax, x_dataset, **kwargs):
    first = kwargs.get('first', False)
    va = kwargs.get('verticalalignment', kwargs.get('va', 'top'))
    ha = kwargs.get('verticalalignment', kwargs.get('ha', 'right'))
    fs = kwargs.get('fontsize', kwargs.get('fs', 14))
    stats_txt = kwargs.get('stats_txt', True)
    pos_x = kwargs.get('pos_x', 0.98)
    y_v_space = kwargs.get('y_v_space', 0.08)
    y_h_space = kwargs.get('y_h_space', 0.1)
    ini_pos_y = kwargs.get('ini_pos_y', 0.96)
    kwargs_histo = dict(bins=30, range=None, color='b', align='mid', alpha=0.6, histtype='bar', normed=False)
    kwargs_histo.update(kwargs.get('kwargs_histo', {}))
    c = kwargs.get('c', kwargs_histo.get('color', 'b'))
    histo = kwargs.get('histo', True)
    dataset_names = kwargs.get('dataset_names', None)
    return_text_list = kwargs.get('return_text_list', False)
    return_histo_vars = kwargs.get('return_histo_vars', False)
    text_list = []
    if histo:
        _n, _bins, _patches = ax.hist(x_dataset, **kwargs_histo)
        # print _n, _n.shape
        # print _bins, _bins.shape
        # print (_bins[:-1] + _bins[1:])/2.
    n_elem = 6
    if dataset_names is not None:
        n_elem += 1
    pos_y = [ini_pos_y - (i * y_v_space) for i in xrange(n_elem)]
    if isinstance(x_dataset, list):
        for i, x in enumerate(x_dataset):
            if dataset_names is None:
                txt, first = plot_histo_stats_txt(x, first)
            else:
                txt, first = plot_histo_stats_txt(x, first, dataset_names[i])
            text_list.append(txt)
            if stats_txt:
                for j, pos in enumerate(pos_y):
                    plot_text_ax(ax, txt[j], **dict(pos_x=pos_x, pos_y=pos, fs=fs, va=va, ha=ha, c=c[i]))
            pos_x -= y_h_space
    else:
        txt, first = plot_histo_stats_txt(x_dataset, first, dataset_names)
        text_list.append(txt)
        if stats_txt:
            for i, pos in enumerate(pos_y):
                plot_text_ax(ax, txt[i], **dict(pos_x=pos_x, pos_y=pos, fs=fs, va=va, ha=ha, c=c))
    if return_text_list:
        if return_histo_vars:
            return ax, text_list, _n, _bins, _patches
        else:
            return ax, text_list
    if return_histo_vars:
        return ax, _n, _bins, _patches
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
