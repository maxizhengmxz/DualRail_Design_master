"""Shared reproducible-method text for the completed parasitic study report."""

from reportlab.platypus import PageBreak


def append_protocol(story, title, text, quantum_status):
    story.append(PageBreak())
    title('Minimum-change compensation protocol')
    text('<b>Inputs and circuit reduction.</b> Each of the 24 independent cases '
         'starts from the unrounded nominal physical record embedded as '
         'nominal-compensation-input.json. Add one registered parasitic at '
         '0.1 or 0.2 fF. A capacitor c between nodes a,b contributes '
         '+c to Caa,Cbb and -c to Cab,Cba. Ground C2 and build the 11-node '
         'matrix. The charge embedding P places +1/-1 on floating transmon '
         'pad pairs, +1 on C1 and +1 on each resonator node. Common-mode '
         'charge of each floating transmon is fixed to zero. Compute '
         'K = P<super>T</super>C<sub>node</sub><super>-1</super>P in order '
         'A1,A2,C,B1,B2,r1,r2. In SI units, ECi/h = e<super>2</super>Kii/(2h).')
    text('<b>Adjustable geometry.</b> Write Ck = Ck,0(1+xk). The 20 fractional '
         'variables comprise four rail ground pairs, nine intentional links, '
         'two resonator self capacitors and five transmon across-pad self '
         'capacitors. Both ground capacitors of each floating transmon move '
         'together. The added parasitic and resonator inductances stay fixed. '
         'All seven K diagonals equal their nominal values, preserving all '
         'five loaded ECs and the two bare resonator frequencies. Positive '
         'physical capacitors and a numerical search guard |xk| &lt;= 0.95 apply.')
    text('<b>Matching tolerance.</b> Set epsilon = 0.001, or 1 ppt = 0.1% = '
         '1000 ppm. For each intentional pair require '
         '|Kij-Kij,0|/|Kij,0| &lt;= epsilon. The nine pairs are A1-A2, A2-C, '
         'C-B1, A2-B1, B1-B2, A1-r1, A2-r1, B1-r2 and B2-r2. Also require '
         '||K-K0||<sub>F,M</sub>/||K0||<sub>F,M</sub> &lt;= epsilon. '
         'Group 1 includes every entry in the symmetric mask M. Group 2 '
         'includes the seven diagonals and nine intentional pairs. Its twelve '
         'other off-diagonal entries remain residual diagnostics. Diagonal '
         'constraints are exact numerical equalities. Near-zero and excluded '
         'channels require absolute-coupling and quantum checks.')
    text('<b>Stage 1. Minimize the worst capacitor change.</b> Introduce t '
         'with -t &lt;= xk &lt;= t and 0 &lt;= t &lt;= 0.95. Minimize '
         't + 10<super>-8</super> sum(xk<super>2</super>) subject to the '
         'diagonal, individual-link and matched-matrix constraints above. '
         'The engineering matrix tolerance is a constraint. This explicitly '
         'favors smaller geometry changes once adequate matching is achieved.')
    text('<b>Stage 2. Reduce combined movement.</b> With the best t = t*, set '
         'the ceiling to t* + max(10<super>-10</super>,10<super>-6</super>t*), '
         'clipped to 0.95. Minimize sum(xk<super>2</super>) within that ceiling '
         'and all matching constraints. This resolves equivalent minimax '
         'allocations toward smaller total movement. Local convergence is '
         'checked for both stages. The results establish feasible local fits.')
    text('<b>Numerical reproduction.</b> Use '
         'studies.bounded_parasitic_compensation.compensate_bounded(physical, '
         'ports, amount, max_fraction=0.95, match_rtol=0.001, '
         'excluded_mode_pairs=excluded). Use excluded=() for group 1. For '
         'group 2 exclude the twelve pairs outside its intentional set. Start '
         'from an EC-preserving self-only adjustment. SLSQP uses analytic '
         'Jacobians, ftol=1e-11/maxiter=500 in stage 1 and '
         'ftol=1e-12/maxiter=200 in stage 2. For incidence vector b, '
         'dK/dc = -P<super>T</super>C<super>-1</super>bb<super>T</super>'
         'C<super>-1</super>P. Verify EC error below 1 Hz, fixed diagonals, '
         'positive capacitors and both matching inequalities after rebuilding '
         'the physical network. The exact solver and audit JSON are embedded.')
    text('<b>Meaning of the tolerance and result status.</b> The 1-ppt value '
         'is a configurable engineering fit target. EM and fabrication '
         'uncertainty require separate estimates. All 24 new electrostatic '
         'fits and secondary stages converged. ' +
         quantum_status)


