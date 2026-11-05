"""Render the saved one-at-a-time parasitic study without optimization."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


def compensation_matrix(study, field):
    names = list(study['cases'])
    components = [c['component'] for c in study['cases'][names[0]]['fit']['changes']
                  if c['nominal_ff'] > 0]
    matrix = np.array([[next(c[field] for c in study['cases'][name]['fit']['changes']
                            if c['component'] == component) for name in names]
                       for component in components])
    return components, names, matrix


def compensation_scale(studies, field):
    if field == 'percent_change':
        return dict(boundaries=np.array([-10., -8., -5., -3., -1., 0., 1., 3., 5., 8., 10.]),
                    ticks=np.array([-10., -8., -5., -3., -1., 0., 1., 3., 5., 8., 10.]),
                    colors=['#44337A', '#2B6CB0', '#168C96', '#7CC5C0', '#B8D8F0', '#E9EFF7',
                            '#FFF6D8', '#F5DE89', '#F4B45E', '#EA7A44', '#CA4958', '#8F285D'],
                    extend='both', spacing='proportional')
    magnitudes = np.concatenate([abs(compensation_matrix(study, field)[2]).ravel()
                                 for study in studies])
    percentiles = np.array([25, 50, 75, 90, 95, 100])
    thresholds = np.percentile(magnitudes, percentiles)
    if np.any(np.diff(thresholds) <= 0):
        raise ValueError('Percentile color bands require distinct magnitude thresholds')
    zero = max(float(thresholds[0]) * 1e-4, 1e-10)
    boundaries = np.r_[-thresholds[::-1], -zero, zero, thresholds]
    colors = ['#44337A', '#2B6CB0', '#168C96', '#7CC5C0', '#B8D8F0', '#E9EFF7',
              '#FFFFFF', '#FFF6D8', '#F5DE89', '#F4B45E', '#EA7A44', '#CA4958', '#8F285D']
    return dict(boundaries=boundaries, ticks=np.r_[-thresholds[::-1], 0, thresholds],
                colors=colors, extend='neither', spacing='uniform')


def draw_compensation_map(study, field, output, scale):
    components, names, matrix = compensation_matrix(study, field)
    percentage = field == 'percent_change'
    unit = '%' if percentage else 'fF'
    amount = study.get('amount_ff', .1)
    figure, axis = plt.subplots(figsize=(9.5, 10.4))
    figure.subplots_adjust(left=.20, right=.77, bottom=.18, top=.92)
    cmap = ListedColormap(scale['colors'])
    cmap.set_bad('white')
    norm = BoundaryNorm(scale['boundaries'], cmap.N, extend=scale['extend'])
    digits = 1 if percentage else 3
    rounded = np.round(matrix, digits)
    displayed = np.ma.masked_where(rounded == 0, matrix)
    heat = axis.imshow(displayed, aspect='auto', cmap=cmap, norm=norm)
    axis.set_xticks(range(len(names)), names, rotation=40, ha='right', fontsize=10)
    labels = []
    for component in components:
        category, name = component.split(':', 1)
        if category == 'ground':
            labels.append(rf'$C_{{g,\mathrm{{{name}}}}}$')
        elif category == 'self':
            labels.append(rf'$C_{{s,\mathrm{{{name}}}}}$')
        elif category == 'resonator':
            labels.append(rf'$C_{{r_{name[-1]}}}$')
        else:
            a, b = name.split('-')
            labels.append(rf'$C_{{\mathrm{{{a}}},\mathrm{{{b}}}}}$')
    axis.set_yticks(range(len(components)), labels, fontsize=10)
    axis.set_xlabel(f'Added parasitic, one at a time ({amount:.2f} fF)', fontsize=11)
    axis.set_ylabel('Compensated physical capacitor', fontsize=11)
    axis.set_xticks(np.arange(-.5, len(names), 1), minor=True)
    axis.set_yticks(np.arange(-.5, len(components), 1), minor=True)
    axis.grid(which='minor', color='white', linewidth=.6)
    axis.tick_params(which='minor', bottom=False, left=False)
    for i, j in np.ndindex(matrix.shape):
        value = matrix[i, j]
        red, green, blue, _ = cmap(norm(value))
        luminance = .2126 * red + .7152 * green + .0722 * blue
        color = 'white' if rounded[i, j] != 0 and luminance < .55 else '#151515'
        label = f'{rounded[i, j]:+.{digits}f}' if rounded[i, j] != 0 else f'{0:.{digits}f}'
        axis.text(j, i, label, ha='center', va='center', fontsize=7.6, color=color)
    ticks = scale['ticks']
    bar_axis = figure.add_axes([.80, .20, .025, .66])
    bar = figure.colorbar(heat, cax=bar_axis, ticks=ticks,
                         spacing=scale['spacing'], extend=scale['extend'])
    tick_labels = [f'{value:+.{digits}f}' if value else f'{0:.{digits}f}' for value in ticks]
    bar.ax.set_yticklabels(tick_labels, fontsize=8)
    bar_label = r'$100\,\Delta C/C_{\mathrm{nom}}$ (%)' if percentage else r'$\Delta C$ (fF)'
    bar.set_label(bar_label, fontsize=10)
    symbol = r'$100\,\Delta C/C_{\mathrm{nom}}$ (%)' if percentage else r'$\Delta C=C_{\mathrm{fit}}-C_{\mathrm{nom}}$ (fF)'
    figure.suptitle(f'Compensation for a {amount:.2f} fF parasitic\n' + symbol, fontsize=15, y=.98)
    note = ('Percentage boundaries are 0, ±1, ±3, ±5, ±8 and ±10%. End colors cover values beyond ±10%.'
            if percentage else 'The two amplitudes share the same fF color boundaries. Values show signed capacitor adjustments.')
    figure.text(.20, .025, note + '\n'
                'Warm colors show increases. Cool colors show decreases. White marks changes rounding to zero.',
                fontsize=9, ha='left', va='bottom')
    figure.savefig(output, dpi=190)
    plt.close(figure)


