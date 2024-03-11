
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
unified dft parser for coordinating dft protocols with
1. xc functionals
2. dispersion corrections / nonlocal correction
3. GTO basis (TODO)
4. geometrical counterpoise (gCP) correction (TODO)
'''

from functools import lru_cache

# supported dispersion corrections
DISP_VERSIONS = ['d3bj', 'd3zero', 'd3bjm', 'd3zerom', 'd3op', 'd4']

@lru_cache(128)
def parse_dft(dft_method):
    ''' conventional dft method ->
    (xc, enable nlc, dispersion, with 3-body dispersion)
    '''
    if not isinstance(dft_method, str):
        return dft_method, None, None, False
    method_lower = dft_method.lower()
    xc = dft_method
    disp = None

    # special cases
    if method_lower == 'wb97m-d3bj':
        return 'wb97m-v', False, ('wb97m', 'd3bj', False)
    if method_lower == 'b97m-d3bj':
        return 'b97m-v', False, ('b97m-v', 'd3bj', False)
    if method_lower == 'wb97x-d3bj':
        return 'wb97x-v', False, ('wb97x', 'd3bj', False)
    if method_lower in ['wb97x-d3', 'wb97x-d3zero']:
        return 'wb97x-d3', False, ('wb97x', 'd3zero', False)
    if method_lower.endswith('-3c'):
        raise NotImplementedError('*-3c methods are not supported yet.')

    for d in DISP_VERSIONS:
        if method_lower.endswith(d):
            disp = d
            xc = method_lower.replace(f'-{d}','')
            return xc, None, (xc, disp, False)
        if method_lower.endswith(d+'2b'):
            disp = d
            xc = method_lower.replace(f'-{d}2b', '')
            return xc, None, (xc, disp, False)
        if method_lower.endswith(d+'atm'):
            disp = d
            xc = method_lower.replace(f'-{d}atm', '')
            return xc, None, (xc, disp, True)
    return xc, None, (xc, None, False)