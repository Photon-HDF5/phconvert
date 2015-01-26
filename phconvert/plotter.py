#
# phconvert - Convert files to Photon-HDF5 format
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#

import matplotlib.pyplot as plt
import numpy as np

_green = 'g'
_red  = 'r'


def alternation_hist(d, bins=None, ax=None, **kwargs):
    """Plot the alternation histogram for the the data in dictionary `d`.
    """
    assert d['alex']
    if d['lifetime']:
        plot_alternation = alternation_hist_nsalex
    else:
        plot_alternation = alternation_hist_usalex
    plot_alternation(d, bins=bins, ax=ax, **kwargs)


def alternation_hist_usalex(d, bins=None, ax=None,
                            hist_style={}, span_style={}):
    """Plot the us-ALEX alternation histogram for the data in dictionary `d`.
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()

    if bins is None:
        bins = 100

    ph_times_t, det_t, period = d['timestamps'], d['detectors'], d['alex_period']
    d_ch, a_ch = d['donor'], d['acceptor']
    d_em_t = (det_t == d_ch)
    a_em_t = (det_t == a_ch)
    D_ON = d['alex_period_donor']
    A_ON = d['alex_period_acceptor']
    D_label = 'Donor: %d-%d' % (D_ON[0], D_ON[1])
    A_label = 'Accept: %d-%d' % (A_ON[0], A_ON[1])

    hist_style_ = dict(bins=bins, alpha=0.2)
    hist_style_.update(hist_style)

    span_style_ = dict(alpha=0.1)
    span_style_.update(span_style)

    plt.hist(ph_times_t[d_em_t] % period, color=_green, label=D_label,
         **hist_style_)
    plt.hist(ph_times_t[a_em_t] % period, color=_red, label=A_label,
         **hist_style_)
    plt.xlabel('Timestamp MODULO Alternation period')

    if D_ON[0] < D_ON[1]:
        plt.axvspan(D_ON[0], D_ON[1], color=_green, **span_style_)
    else:
        plt.axvspan(0, D_ON[1], color=_green, **span_style_)
        plt.axvspan(D_ON[0], period, color=_green, **span_style_)

    if A_ON[0] < A_ON[1]:
        plt.axvspan(A_ON[0], A_ON[1], color=_red, **span_style_)
    else:
        plt.axvspan(0, A_ON[1], color=_red, **span_style_)
        plt.axvspan(A_ON[0], period, color=_red, **span_style_)

    plt.legend(loc='best')


def alternation_hist_nsalex(d, bins=None, ax=None):
    """Plot the ns-ALEX alternation histogram for the data in dictionary `d`.
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()

    if bins is None:
        bins = np.arange(d['tcspc_num_bins'])

    d_ch, a_ch = d['donor'], d['acceptor']
    D_ON = d['alex_period_donor']
    A_ON = d['alex_period_acceptor']
    D_label = 'Donor: %d-%d' % (D_ON[0], D_ON[1])
    A_label = 'Accept: %d-%d' % (A_ON[0], A_ON[1])

    nanotimes_d = d['nanotimes'][d['detectors'] == d_ch]
    nanotimes_a = d['nanotimes'][d['detectors'] == a_ch]
    plt.hist(nanotimes_d, bins=bins, histtype='step', label=D_label, lw=1.2,
             alpha=0.5, color=_green)
    plt.hist(nanotimes_a, bins=bins, histtype='step', label=A_label, lw=1.2,
             alpha=0.5, color=_red)
    plt.xlabel('TCSPC Nanotime')

    plt.yscale('log')
    plt.axvspan(D_ON[0], D_ON[1], color=_green, alpha=0.1)
    plt.axvspan(A_ON[0], A_ON[1], color=_red, alpha=0.1)
