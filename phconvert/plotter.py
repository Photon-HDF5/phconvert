#
# phconvert - Convert files to Photon-HDF5 format
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#

import re
from itertools import chain, repeat
from typing import Union
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

has_numba = True
try:
    import numba
except ImportError:
    has_numba = False

_green = 'g'
_red = 'r'
_ch_rgx = re.compile(r'(spectral|polarization|split)_ch(1-9]\d*)')
_spec_rgx = re.compile(r'(spectral)_ch([1-9]\d*)')
_pol_rgx = re.compile(r'(polarization)_ch([1-9]\d*)')
_split_rgx = re.compile(r'(split)_ch([1-9]\d*)')
_nph_rgx = re.compile(r'non_photon_id([1-9]\d*)')
_al_rgx = re.compile(r'(alex_excitation_period)([1-9]\d*)')


def _mask_phot(times, dets, dgroup):
    """
    Return only times of photons with a detector in dgroup

    Parameters
    ----------
    times : np.ndarray
        Values to mask.
    dets : np.ndarray
        Array of detectors to match with the values in dgroup.
    dgroup : np.ndarray|set
        Values to include in time.

    Returns
    -------
    filt : np.ndarray
        Filtered array of times.

    """
    dgroup = np.atleast_1d(dgroup)
    mask = np.zeros(times.size, dtype=np.bool_)
    for d in dgroup:
        mask += dets == d
    filt = times[mask]
    return filt

if has_numba:
    _mask_phot = numba.jit(_mask_phot)


def _toint(nstr):
    """Convert a string with variable leading zeros into an int"""
    nstr = nstr.lstrip('0')
    return int(nstr) if nstr else '0'


def _unions(args):
    """Union of a list of sets"""
    comp = args[0]
    for arg in args[1:]:
        comp = comp | arg # do not use |= b/c cause side effect
    return comp


def _intersects(args):
    """Common elements across all sets in list of sets"""
    comp = args[0]
    for arg in args[1:]:
        comp &= arg
    return comp

def _pos_in(idx, subs):
    """Find index in subs where idx is present in list of sets"""
    for i, s in enumerate(subs):
        if idx in subs:
            return i
    return None

def _get_detector_arrays(idxs:dict, det_spec:dict, rgx:re.Pattern, sort:bool):
    """
    Fill out idxs (dictionary) of detectors based on detectors_specs dictionary
    for channel defined by re.Match rgx, raises error if sort is True

    Parameters
    ----------
    idxs : dict
        Dictionary to fill out detector channels.
    det_spec : dict
        Dictionary of /photon_dataX/measurement_specs/detectors_specs group.
    rgx : re.Pattern
        re.Pattern for given channel, group 1 should be name, group 2 should
        be the number of the channel.
    sort : bool
        Whether or not to raise an error if no channels defined.

    Raises
    ------
    ValueError
        det_spec is missing channels required by rgx.

    """
    groups = list()
    for key, val in det_spec.items():
        mch = rgx.fullmatch(key)
        if mch is None:
            continue
        i = _toint(mch.group(2))
        name = mch.group(1)
        groups.append((i,  set(val)))
    if groups:
        groups = sorted(groups)
        idxs[name] = groups
    if sort and not groups:
        raise ValueError(f"No {name} groups specified in detectors_specs")


def _get_det_combos(det_groups, det_id, markers):
    """
    Find all possible combinations of each channel type, and the cooresponding
    detectors.

    Parameters
    ----------
    det_groups : dict
        /photon_dataX/measurement_specs/detectors_specs group.
    det_id : set
        set of detector markers, from np.unique(detectros).
    markers : set
        set of all detector ids that are markers.

    Returns
    -------
    out_groups : dict
        dict of {channel_name:set[detector ids]}.

    """
    # generate single set with all ids in it
    det_all = _unions(tuple(d for _, d in chain.from_iterable(det_groups.values())))
    det_all |= det_id - markers # add any uncategorized markers
    sets = dict() # final output
    det_types = tuple(det_groups.keys())
    for d in det_all:
        idx = [None for _ in det_types]
        for i, det_t in enumerate(det_types):
            for j, s in det_groups[det_t]:
                if d in s:
                    idx[i] = j
                    break
        idx = tuple(idx)
        if idx not in sets:
            sets[idx] = list()
        sets[idx].append(d)
    # convert idx into names, and lists of idx into sets
    out_groups = dict()
    blank = tuple(None for _ in det_types)
    for idx, det_set in sets.items():
        if idx == blank:
            for j in det_set:
                out_groups[f'Unassigned id:{j}'] = np.array([j, ])
        else:
            name = ' '.join(f'{name} {i}' for name, i in zip(det_types, idx))
            out_groups[name] = np.array(det_set)
    out_groups = {(key if 'Unassigned' in key else f'{key} {list(val)}'):val 
                  for key, val in out_groups.items()}
    return out_groups


def _get_detectors_specs(ph_data:dict, group_dets:bool, sort_spectral:bool, 
                         sort_polarization:bool, sort_split:bool):
    detectors = ph_data['detectors'][:]
    det_id = set(np.unique(detectors))
    
    det_spec = ph_data['measurement_specs']['detectors_specs']
    
    non_photon = set(val for key, val in det_spec.items() if _nph_rgx.fullmatch(key))
    if non_photon:
        non_photon = set(chain(*non_photon))
    
    det_groups = dict()
    if group_dets:
        _get_detector_arrays(det_groups, det_spec, _spec_rgx, sort_spectral)
        _get_detector_arrays(det_groups, det_spec, _pol_rgx, sort_polarization)
        _get_detector_arrays(det_groups, det_spec, _split_rgx, sort_split)
        det_groups = _get_det_combos(det_groups, det_id, non_photon)
    else:
        det_groups = {f'Detector id:{i}':np.array([i,]) for i in 
                      det_id if i not in non_photon}
    return det_groups


def _plot_histograms(ax:plt.Axes, values:np.ndarray, detectors:np.ndarray[np.uint8], 
                     dgroups, hist_style:dict):
    if len(dgroups) == 2 and all('spectral' in key for key in dgroups.keys()):
        for label, dgroup in dgroups.items():
            c = _green if 'spectral 1' in label else _red
            label = "Donor" if 'spectral 1' in label else "Acceptor"
            ax.hist(_mask_phot(values, detectors, dgroup), color=c, 
                    label=label, **hist_style)
    else:
        for label, dgroup in dgroups.items():
            ax.hist(_mask_phot(values, detectors, dgroup), label=label, **hist_style)


def _plot_spans(ax:plt.Axes, meas_spec:dict, span_style:dict):
    """
    Plot excitation ranges based on meas_spec from photon_data/measurement_specs
    into ax, according to span_style keyword arguments
    """
    spans = sorted([(_toint(_al_rgx.fullmatch(name).group(2)), span) 
                    for name, span in meas_spec.items() 
                    if _al_rgx.fullmatch(name)])
    
    if len(spans) == 2:
        for (i, span), color, label in zip(spans, (_green, _red), ('Donor Ex ', 'Acceptor Ex ')):
            label = label + ' '.join(f'{b}-{e}' for b, e in zip(span[::2], span[1::2]))
            ax.axvspan(span[0], span[1], color=color, label=label, **span_style)
            for b, e in zip(span[2::2], span[3::2]):
                ax.axvspan(b, e, color=color, **span_style)
    else:
        for j, (i, span) in enumerate(spans):
            label = f'Excitation period {i} ' + ', '.join(f'{b}-{e}' for b, e in zip(span[::2], span[1::2]))
            r = ax.axvspan(span[0], span[1], label=label, color=mpl.colormaps['Spectral_r']((j+1)/(len(spans)+1)), **span_style)
            for b, e in zip(span[2::2], span[3::2]):
                ax.axvspan(b, e, color=r.color, **span_style)


def alternation_hist(d:dict, bins:Union[int,np.ndarray]=None, ich:int=0, ax:plt.Axes=None, **kwargs):
    """
    Plot the alternation histogram or TCSPC decay of the data dictionary, given
    the currently present settings containted within, loaded by
    :func:`loader.loadfile_bh`, :func:`loader.loadfile_pq` or
    :func:`loader.loadfile_sm`.

    Parameters
    ----------
    d : dict
        Dictionary loaded from loader function and with user inputed parameters
        for which to plot the alternation histogram.
    bins : int|np.ndarray, optional
        Input to matplotlib.axes.Axes.hist, either the number of bins to use
        or the bin edges to plot in the alternation histogram. The default is None.
    ich : int, optional
        Which photon_dataX group to plot, if only one, default of 0 will plot
        photon_data group. The default is 0.
    ax : matplolib.axes.Axes, optional
        Matplotlib Axes in which to plot the alternation histogram. The default is None.
    **kwargs : 
        kwargs passed to either :func:`plotter.alternation_hist_cw` or
        :func:`plotter.alternation_hist_pulsed` depending on if data contains
        nanotimes.

    """
    setup = d['setup']

    if setup['lifetime']:
        TYPE = 'lifetime'
    else:
        TYPE = 'CW'
    plot_alex = {'CW': alternation_hist_cw,
                 'lifetime': alternation_hist_pulsed}
    plot_alex[TYPE](d, bins=bins, ich=ich, ax=ax, **kwargs)



def alternation_hist_cw(d, bins=None, ich=0, group_dets=False,
                        sort_spectral=False, sort_polarization=False,
                        sort_split=False, ax=None, 
                        hist_style=None, span_style=None):
    """
    Plot the laser alternation histogram for the data dictionary d assuming
    d uses continuous wave alternating laser excitation

    Parameters
    ----------
    d : dict
        Raw data dictionary of loaded photon information.
    bins : numpy.ndarray, optional
        Time bins for alternation period. The default is None.
    ich : int, optional
        Which photon_data spot to use (multispot only, ignored for single spot).
        The default is 0.
    use_spectral : bool, optional
        If true, use definition from ``measurement_specs/detectors`` instead of 
        plotting detectors independently. The default is False.
    ax : mpl.axes.Axes, optional
        Matplotlib axes in which to plot alternation histogram. If None, calls
        plt.figure() and then plt.gca() to get new axes. The default is None.
    hist_style : dict, optional
        Keyword arguments passed to ax.hist for alternation histogram.
        If None, generates defautlt dictionary. The default is None.
    span_style : dict, optional
        Keyword arguments passed to ax.axvspan for alternation period.
        If None, generates defautlt dictionary. The default is None.

    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    bins = 101 if bins is None else bins
    hist_style_ = dict(bins=bins, alpha=0.5, histtype='stepfilled', lw=1.3)
    hist_style_.update(dict() if hist_style is None else hist_style)
    
    span_style_ = dict(alpha=0.1)
    span_style_.update(dict() if span_style is None else span_style)
    
    # extract fields from d dictionary, use [:] in case fields are tables arrays
    ph_data = d.get('photon_data', d.get('photon_data%d' % ich))
    detectors = ph_data['detectors'][:]
    det_groups = _get_detectors_specs(ph_data, group_dets, 
                                      sort_spectral, sort_polarization, sort_split)
    ph_times_t = ph_data['timestamps'][:]
    meas_specs = ph_data['measurement_specs']
    # Calculate the periods
    period = meas_specs['alex_period']
    offset = meas_specs.get('alex_offset', 0)
    ph_times_mod = (ph_times_t - offset) % period
    # Plot the spans of the alex periodss
    _plot_spans(ax, meas_specs, span_style_)
    # Plot the histograms of detectors
    _plot_histograms(ax, ph_times_mod, detectors, det_groups, hist_style_)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    
    
def alternation_hist_pulsed(d:dict, ich:int=0, bins:Union[int,np.ndarray]=None, 
                            group_dets:bool=False, sort_spectral:bool=False, 
                            sort_polarization:bool=False, sort_split:bool=False, 
                            ax:plt.Axes=None, hist_style:dict=None, 
                            span_style:dict=None):
    """
    Plot TCSPC decays for data in d. Must contain nanotimes

    Parameters
    ----------
    d : dict
        Data dictionary from `loader_`` type function, with additional fields specified.
    ich : int, optional
        which photon_dataX group to plot, if only one in d, then default to photon_data.
        The default is 0.
    bins : int|numpy.ndarray, optional
        Bins of TCSPC bins. The default is None.
    group_dets : bool, optional
        Whether or not to group dets of same type together in histogram. The default is False.
    sort_spectral : bool, optional
        DESCRIPTION. The default is False.
    sort_polarization : bool, optional
        Check that polarization exists in d. The default is False.
    sort_split : bool, optional
        Check that split exists in d. The default is False.
    ax : matplotlib.axes.Axex, optional
        Matplotlib Axes in which to plot histogram. The default is None.
    hist_style : dict, optional
        keyword arguments to pass to matplotlib.axes.Axes.hist. The default is None.
    span_style : dict, optional
        dict of keyword arguments to be passed to matplotlib.aesx.Axes.axvspan. The default is None.

    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    hist_style_ = dict(histtype='step', lw=1.2, alpha=0.5)
    hist_style_.update(dict() if hist_style is None else hist_style)
    span_style_ = dict(alpha=0.3)
    span_style_.update(dict() if span_style is None else span_style)
    
    multispot = 'photon_data' not in d 
    ich = ich if multispot else 0
    ph_data = d['photon_data%d'%ich] if multispot else d['photon_data']
    
    detectors = ph_data['detectors'][:]
    det_groups = _get_detectors_specs(ph_data, group_dets, 
                                      sort_spectral, sort_polarization, sort_split)
    meas_spec = ph_data['measurement_specs']
    nanotimes = ph_data['nanotimes']
    setup_det = d['setup'].get('detectors', dict())
    if 'tcspc_offsets' in setup_det:
        if np.any(setup_det['tcspc_offsets']!= 0):
            idxs = setup_det['id']
            spots = setup_det['spot'] if multispot else repeat(0)
            nanotimes = nanotimes.copy().astype(np.int16)
            for idx, offset, sp in zip(idxs, setup_det['tcspc_offsets'], spots):
                if sp != ich:
                    continue
                nanotimes[detectors==idx] -= offset
                
    # if 'tcspc_offset' in d['setup'].get('detectors', dict()):
    #     # code to extract offsets
    hist_style_.update(bins=np.arange(0,nanotimes.max()+1,1) if bins is None else bins)
    _plot_spans(ax, meas_spec, span_style_)
    _plot_histograms(ax, nanotimes, detectors, det_groups, hist_style_)
    
    # Final plotting niceties
    ax.set_xlabel('TCSPC nanotimes bins')
    ax.set_yscale('log')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)


def alternation_hist_usalex(d:dict, bins=None, ich:int=0, ax:plt.Axes=None,
                            hist_style:dict=None, span_style:dict=None):
    """Plot the us-ALEX alternation histogram for the data in dictionary `d`.
    
    .. note::
        
        This is an older function will be deprecated, replace with alternation_hist
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
    hist_style_.update(dict() if hist_style is None else hist_style)

    span_style_ = dict(alpha=0.1)
    span_style_.update(dict() if span_style is None else span_style)

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
    """
    Plot the ns-ALEX alternation histogram for the data in dictionary `d`.
    
    .. note::
        
        This is an older function, will be deprecated, replace with alternation_hist.
    
    
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
