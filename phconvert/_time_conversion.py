#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal module for converting date strings of more aribtrary format into Photon-HDF5 sanctioned format
"""

import warnings
import time
import re

_date_regex = re.compile(r'(?P<first>\d{1,4})([\-/])(?P<month>\d{1,2})\2(?P<last>\d{1,4})')
_time_regex = re.compile(r'(?P<hour>\d{1,2})(?P<tsep>[\.:])(?P<min>\d{1,2})(?P<hassec>\2(?P<sec>[0-6]?\d)((\.|\2)(?P<subsec>\d+))?)?(?P<ampm>\s*[AaPp][Mm])?')


_full_regex = re.compile(r'(?P<first>\S+)\s+(?P<second>\S+)')

def _fill_num(text:str, default:str, name:str, short:int=0) -> int:
    short = short if short != 0 else len(default)
    if len(text) > len(default):
        raise ValueError(f"Field {name} too long")
    elif len(text) != short:
        warnings.warn(f"Expected {name} field to be of width {short}")
    return int((default[:len(default)-len(text)]+ text).lstrip('0'))
    
def _normalize_time(text:str, year_first=None, year_width:int=4)->str:
    # see if string has both date and time
    match = _full_regex.match(text)
    if match is None or match.span()[1] != len(text):
        raise ValueError(f"The string '{text}' cannot be interpreted as a date in any acceptable format")
    # extract date and time groups
    fdate = _date_regex.match(match.group('first'))
    ftime = _time_regex.match(match.group('first'))
    sdate = _date_regex.match(match.group('second'))
    stime = _time_regex.match(match.group('second'))
    # identify which order date and time are in
    if fdate and not ftime:
        date_match, time_match = fdate, stime
    elif ftime and not fdate:
        date_match, time_match = sdate, ftime
    else:
        raise ValueError(f"The string '{text}' cannot be interpreted as a date in any acceptable format, appears as repeated day or time of day")
    # process time fields
    hour = _fill_num(time_match.group('hour'), '00', 'hour')
    if time_match.group('ampm') and time_match.group('ampm').lower() == 'pm':
        hour = hour + 12
    minute = _fill_num(time_match.group('min'), '00', 'minute')
    if not time_match.group('hassec'):
        second = 0
    else:
        second = _fill_num(time_match.group('sec'), '00', 'second')
    microsec = '0'*6
    if time_match.group('subsec'):
        microsec = time_match.group('subsec') + '0'*(6-len(time_match.group('subsec')))
    # process date fields
    first_num, last_num = date_match.group('first'), date_match.group('last')
    if len(first_num) > 2 and len(last_num) <= 2:
        year = _fill_num(first_num, '2000', 'year', short=year_width)
        day = _fill_num(last_num, '01', 'day')
        if year_first is False:
            warnings.warn("String places year first, expected year to be last")
    elif len(first_num) <= 2 and len(last_num) > 2:
        year = _fill_num(last_num, '2000', 'year', short=year_width)
        day = _fill_num(first_num, '01', 'day')
        if year_first is True:
            warnings.warn("String places year last, expected year to be first")
    elif len(first_num) <= 2 and len(last_num) <= 2:
        if year_first is True:
            year = _fill_num(first_num, '2000', 'year', short=year_width)
            day = _fill_num(last_num, '01', 'day')
        elif year_first is False:
            year = _fill_num(last_num, '2000', 'year', short=year_width)
            day = _fill_num(first_num, '01', 'day')
        else:
            raise ValueError("When year_first is not specified, must have one field with 4 digits")
    else:
        raise ValueError(f"Invalid date field {date_match.group()} contains multiple values with more than 2 digits")
            
    month = _fill_num(date_match.group('month'), '01', 'month')
    if month > 12:
        warnings.warn("Month value exceeds 12, assuming date read as YYYY/DD/MM, instead of expected YYYY/MM/DD")
        day, month = month, day
    tm = time.strptime(f'{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}.{microsec}', '%Y-%m-%d %H:%M:%S.%f')
    return time.strftime('%Y-%m-%d %H:%M:%S', tm)