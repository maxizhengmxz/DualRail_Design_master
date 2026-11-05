"""Opt-in green group-one and red group-two parasitic drawing layers."""
import numpy as np

DARK_GREEN = '#145A32'
RED = '#B22222'
GROUP_ONE_NUMBERS = dict(zip(('A11-r1', 'A22-r1', 'B22-r2', 'B11-r2',
                              'A21-B11', 'B12-A22', 'A11-A22', 'B22-B11'),
                             (1, 2, 3, 4, 5, 6, 11, 12)))


def draw_group_one_parasitics(ax, requested, values, show_notes=True):
    """Use approved transmon geometry; dots identify contacts, bridges crossings."""
    routes = {
        'A11-r1': [(-1.1, 3.2), (-1.1, 4.75), (1.65, 4.75), (1.65, 4.3), (2.2, 4.3)],
        'A22-r1': [(7.1, 3.2), (7.1, 4.75), (4.35, 4.75), (4.35, 4.3), (3.8, 4.3)],
        'B22-r2': [(19.1, 3.2), (19.1, 4.75), (16.35, 4.75), (16.35, 4.3), (15.8, 4.3)],
        'B11-r2': [(10.9, 3.2), (10.9, 4.75), (13.65, 4.75), (13.65, 4.3), (14.2, 4.3)],
        'A21-B11': [(4.9, 3.2), (4.9, 7.3), (10.9, 7.3), (10.9, 3.2)],
        'B12-A22': [(13.1, 3.2), (13.1, 7.95), (7.1, 7.95), (7.1, 3.2)],
        'A11-A22': [(-1.1, 3.2), (-1.1, 6.90), (7.1, 6.90), (7.1, 3.2)],
        'B22-B11': [(19.1, 3.2), (19.1, 6.90), (10.9, 6.90), (10.9, 3.2)],
    }
    unknown = set(requested) - set(routes)
    if unknown:
        raise ValueError(f'No approved parasitic drawing route for {sorted(unknown)}')

    def line(a, b, **kwargs):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=DARK_GREEN, lw=1.6,
                solid_capstyle='butt', zorder=8, **kwargs)

    for name in requested:
        number = GROUP_ONE_NUMBERS[name]
        points = routes[name]
        # The long horizontal second segment contains the capacitor.
        for segment, (a, b) in enumerate(zip(points[:-1], points[1:])):
            if segment != 1:
                line(a, b)
                continue
            left, right = sorted((a[0], b[0]))
            midpoint = (left + right) / 2
            y = a[1]
            line((left, y), (midpoint - .105, y))
            line((midpoint + .105, y), (right, y))
            for x in (midpoint - .105, midpoint + .105):
                line((x, y - .30), (x, y + .30))
            ax.text(midpoint, y + .43, rf'$C_{{p{number}}}$', color=DARK_GREEN,
                    fontsize=14, ha='center', va='center', zorder=14)
        for x, y in (points[0], points[-1]):
            ax.plot(x, y, 'o', color=DARK_GREEN, ms=4, zorder=15)

    def bridge(x, y):
        # Erase the straight vertical crossing, restore the horizontal wire,
        # then draw a white-backed overpass. The two nets stay visually separate.
        ax.plot([x, x], [y - .19, y + .19], color='white', lw=4.5, zorder=9)
        ax.plot([x - .22, x + .35], [y, y], color=DARK_GREEN, lw=1.6, zorder=10)
        angle = np.linspace(0, np.pi, 80)
        xx = x + .17 * np.sin(angle)
        yy = y - .17 * np.cos(angle)
        ax.plot(xx, yy, color='white', lw=5, zorder=11)
        ax.plot(xx, yy, color=DARK_GREEN, lw=1.6, zorder=12)

    if 'A21-B11' in requested and 'A22-r1' in requested:
        bridge(4.9, 4.75)
    if 'B12-A22' in requested and 'B11-r2' in requested:
        bridge(13.1, 4.75)
    if 'B12-A22' in requested and 'A21-B11' in requested:
        bridge(7.1, 7.3)
    if 'A11-A22' in requested and 'A21-B11' in requested:
        bridge(4.9, 6.90)
    if 'B22-B11' in requested and 'B12-A22' in requested:
        bridge(13.1, 6.90)
    for name, endpoints in (('A11-A22', (-1.1, 7.1)), ('B22-B11', (10.9, 19.1))):
        if name in requested:
            for x in endpoints:
                ax.plot(x, 6.90, 'o', color=DARK_GREEN, ms=4, zorder=15)
    for present, other, x in [('A22-r1', 'B12-A22', 7.1),
                               ('B11-r2', 'A21-B11', 10.9)]:
        if present in requested and other in requested:
            ax.plot(x, 4.75, 'o', color=DARK_GREEN, ms=4, zorder=15)

    # Match the existing convention of numerical values in a lower legend.
    for index, name in enumerate(requested):
        columns = (1.5, 6.5, 11.5, 16.5) if len(requested)>6 else (3., 9., 15.)
        x = columns[index % len(columns)]
        y = -2.65 - .48 * (index // len(columns))
        a, b = name.split('-')
        a = a.replace('r1', 'r_1').replace('r2', 'r_2')
        b = b.replace('r1', 'r_1').replace('r2', 'r_2')
        ax.text(x, y, rf'$C_{{p{GROUP_ONE_NUMBERS[name]}}}\;({a},{b})={values.get(name, 0.):.2f}$',
                color=DARK_GREEN, fontsize=12, ha='center', va='center')
    if show_notes:
        ax.text(9, -3.68, 'Dark green: parasitic capacitors. Dots mark contacts; bridges mark separate conductors.',
                color=DARK_GREEN, fontsize=11, ha='center', va='center')
        ax.text(9, -4.08, 'All capacitances are in fF. Defaults are 0.00; each study case adds one link at 0.10.',
                fontsize=11, ha='center', va='center')


def draw_group_two_parasitics(ax, requested, values, group_one=()):
    """Route the four outer-transmon-to-C1 capacitors through a shared C1 bus."""
    routes = {
        'A11-C1': [(-1.1, 4.75), (-1.1, 10.30), (10.55, 10.30)],
        'A12-C1': [(1.1, 3.20), (1.1, 9.20), (10.55, 9.20)],
        'B21-C1': [(16.9, 3.20), (16.9, 9.20), (10.55, 9.20)],
        'B22-C1': [(19.1, 4.75), (19.1, 10.30), (10.55, 10.30)],
    }
    unknown = set(requested) - set(routes)
    if unknown:
        raise ValueError(f'No group-two drawing route for {sorted(unknown)}')

    def line(a, b):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=RED, lw=1.6,
                solid_capstyle='butt', zorder=20)

    # Branch from the upper terminal of the coupler self-capacitor, which is C1.
    line((9.8, 2.30), (10.55, 2.30))
    line((10.55, 2.30), (10.55, 10.30))
    ax.plot(9.8, 2.30, 'o', color=RED, ms=4, zorder=26)
    ax.text(10.78, 10.60, r'$C_1$', color=RED, fontsize=12, ha='left', va='center')

    for number, name in enumerate(requested, start=7):
        start, corner, end = routes[name]
        line(start, corner)
        left, right = sorted((corner[0], end[0]))
        midpoint = (left + right) / 2
        y = corner[1]
        line((left, y), (midpoint - .105, y))
        line((midpoint + .105, y), (right, y))
        for x in (midpoint - .105, midpoint + .105):
            line((x, y - .30), (x, y + .30))
        ax.text(midpoint, y + .43, rf'$C_{{p{number}}}$', color=RED,
                fontsize=14, ha='center', va='center', zorder=27)
        for x, yy in (start, end):
            ax.plot(x, yy, 'o', color=RED, ms=4, zorder=26)

    def bridge(x, y, underneath, right_edge=None):
        ax.plot([x, x], [y - .19, y + .19], color='white', lw=4.5, zorder=21)
        # Only restore the actual underlying conductor, avoiding nearby plates.
        ax.plot([x - .23, x + .23 if right_edge is None else right_edge], [y, y],
                color=underneath, lw=1.6, zorder=22)
        angle = np.linspace(0, np.pi, 80)
        xx = x - .17 * np.sin(angle)
        yy = y - .17 * np.cos(angle)
        ax.plot(xx, yy, color='white', lw=5, zorder=23)
        ax.plot(xx, yy, color=RED, lw=1.6, zorder=24)

    if 'A12-C1' in requested:
        bridge(1.1, 4.75, DARK_GREEN)
        if 'A11-A22' in group_one:
            bridge(1.1, 6.90, DARK_GREEN)
    if 'B21-C1' in requested:
        bridge(16.9, 4.75, DARK_GREEN)
        if 'B22-B11' in group_one:
            bridge(16.9, 6.90, DARK_GREEN)
    bridge(10.55, 2.80, '#202020', right_edge=10.6)
    bridge(10.55, 3.85, '#202020', right_edge=10.6)
    bridge(10.55, 7.30, DARK_GREEN)
    bridge(10.55, 7.95, DARK_GREEN)

    for index, name in enumerate(requested, start=7):
        a, b = name.split('-')
        ax.text((1.5, 6.5, 11.5, 16.5)[index - 7], -3.73,
                rf'$C_{{p{index}}}\;({a},C_1)={values.get(name, 0.):.2f}$',
                color=RED, fontsize=12, ha='center', va='center')
    ax.text(5.7, -4.22, 'Dark green: group 1 parasitics', color=DARK_GREEN,
            fontsize=11, ha='center', va='center')
    ax.text(12.3, -4.22, 'Red: group 2 parasitics', color=RED,
            fontsize=11, ha='center', va='center')
    ax.text(9, -4.62, 'All capacitances are in fF; parasitic defaults are 0.00. Dots mark contacts; bridges mark separate conductors.',
            fontsize=11, ha='center', va='center')
