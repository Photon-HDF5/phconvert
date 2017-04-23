#
# phconvert - Convert files to Photon-HDF5 format
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#

import matplotlib.pyplot as plt
import numpy as np

_green = 'g'
_red = 'r'


def alternation_hist(d, bins=None, ich=0, ax=None, **kwargs):
    """Plot the alternation histogram for the the data in dictionary `d`.
    """
    setup = d['setup']

    ph_data = d.get('photon_data', d.get('photon_data%d' % ich))
    measurement_type = ph_data['measurement_specs']['measurement_type']
    if ((measurement_type == 'generic' and not setup['lifetime']) or
            measurement_type == 'smFRET-usALEX'):
        TYPE = 'CW'
    elif ((measurement_type == 'generic' and setup['lifetime']) or
            measurement_type == 'smFRET-nsALEX'):
        TYPE = 'lifetime'
    else:
        msg = ('Alternation histogram for measurement %s not supported.' %
               measurement_type)
        raise ValueError(msg)

    plot_alex = {'CW': alternation_hist_usalex,
                 'lifetime': alternation_hist_nsalex}
    plot_alex[TYPE](d, bins=bins, ich=ich, ax=ax, **kwargs)


def alternation_hist_usalex(d, bins=None, ich=0, ax=None,
                            hist_style={}, span_style={}):
    """Plot the us-ALEX alternation histogram for the data in dictionary `d`.
    """
    msg = ("At least one source needs to be alternated "
           "(i.e. intensity-modulated)")
    assert any(d['setup']['excitation_alternated']), msg
    if ax is None:
        plt.figure()
        ax = plt.gca()

    if bins is None:
        bins = 101

    ph_data = d.get('photon_data', d.get('photon_data%d' % ich))
    ph_times_t = ph_data['timestamps'][:]
    det_t = ph_data['detectors'][:]
    meas_specs = ph_data['measurement_specs']
    period = meas_specs['alex_period']

    det_specs = meas_specs['detectors_specs']
    d_ch = det_specs['spectral_ch1']
    a_ch = det_specs['spectral_ch2']

    d_em_t = (det_t == d_ch)
    a_em_t = (det_t == a_ch)
    D_ON = meas_specs.get('alex_excitation_period1', None)
    A_ON = meas_specs.get('alex_excitation_period2', None)
    offset = meas_specs.get('alex_offset', 0)
    D_label = 'Donor: '
    A_label = 'Accept: '
    D_label += 'no selection' if D_ON is None else ('%d-%d' % tuple(D_ON))
    A_label += 'no selection' if A_ON is None else ('%d-%d' % tuple(A_ON))

    hist_style_ = dict(bins=bins, alpha=0.5, histtype='stepfilled', lw=1.3)
    hist_style_.update(hist_style)

    span_style_ = dict(alpha=0.1)
    span_style_.update(span_style)

    ax.hist((ph_times_t[d_em_t] - offset) % period, color=_green, label=D_label,
            **hist_style_)
    ax.hist((ph_times_t[a_em_t] - offset) % period, color=_red, label=A_label,
            **hist_style_)
    ax.set_xlabel('(timestamps - alex_offset) MOD alex_period')

    if D_ON is not None:
        if D_ON[0] < D_ON[1]:
            ax.axvspan(D_ON[0], D_ON[1], color=_green, **span_style_)
        else:
            ax.axvspan(0, D_ON[1], color=_green, **span_style_)
            ax.axvspan(D_ON[0], period, color=_green, **span_style_)

    if A_ON is not None:
        if A_ON[0] < A_ON[1]:
            ax.axvspan(A_ON[0], A_ON[1], color=_red, **span_style_)
        else:
            ax.axvspan(0, A_ON[1], color=_red, **span_style_)
            ax.axvspan(A_ON[0], period, color=_red, **span_style_)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)


def alternation_hist_nsalex(d, bins=None, ich=0, ax=None):
    """Plot the ns-ALEX alternation histogram for the data in dictionary `d`.
    """
    msg = 'At least one source needs to be pulsed.'
    assert not all(d['setup']['excitation_cw']), msg
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ph_data = d.get('photon_data', d.get('photon_data%d' % ich))
    if bins is None:
        bins = np.arange(ph_data['nanotimes_specs']['tcspc_num_bins'])
    meas_specs = ph_data['measurement_specs']
    det_specs = meas_specs['detectors_specs']
    d_ch = det_specs['spectral_ch1']
    a_ch = det_specs['spectral_ch2']

    D_ON = meas_specs['alex_excitation_period1']
    A_ON = meas_specs['alex_excitation_period2']

    D_label = 'Donor: %d-%d' % (D_ON[0], D_ON[1])
    A_label = 'Accept: %d-%d' % (A_ON[0], A_ON[1])

    nanotimes_d = ph_data['nanotimes'][ph_data['detectors'] == d_ch]
    nanotimes_a = ph_data['nanotimes'][ph_data['detectors'] == a_ch]

    ax.hist(nanotimes_d, bins=bins, histtype='step', label=D_label, lw=1.2,
            alpha=0.5, color=_green)
    ax.hist(nanotimes_a, bins=bins, histtype='step', label=A_label, lw=1.2,
            alpha=0.5, color=_red)
    ax.set_xlabel('TCSPC nanotimes bins')

    ax.set_yscale('log')
    ax.axvspan(D_ON[0], D_ON[1], color=_green, alpha=0.1)
    ax.axvspan(A_ON[0], A_ON[1], color=_red, alpha=0.1)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
