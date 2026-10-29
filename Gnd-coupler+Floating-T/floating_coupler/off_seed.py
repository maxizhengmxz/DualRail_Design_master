"""Analytic upper-OFF seeds for the minimal same-pad floating-coupler circuit.

Assumptions: weak physical links, harmonic charge matrix elements, symmetric
bare pad grounds, negligible fabricated direct rail capacitor and parasitic
cross-pad links. This describes reduced physical-port exchange, not a proof
about the five-transmon selected-CZ-channel root. eta is the ground-series
fraction of the coupler's OWN capacitance, before external loading.

Combining Sete et al. (PRApplied 15,064063), Eqs.5,7-9, for same-pad dominant
links gives 2*x^2-(2+eta)*(fa^2+fb^2)*x+2*(1+eta)*fa^2*fb^2=0,
where x=fc^2. Equal-qubit limit: fc= fq*sqrt(1+eta).
"""
import math


def _frequencies(fa,fb):
    if not all(math.isfinite(v) and v>0 for v in (fa,fb)):
        raise ValueError('Rail frequencies must be finite positive cyclic GHz')
    return fa*fa,fb*fb


def same_pad_off_frequency(fa,fb,eta):
    a,b=_frequencies(fa,fb)
    if not math.isfinite(eta) or not 0<eta<=1:raise ValueError('Own-ground fraction must be in (0,1]')
    s,p=a+b,a*b
    d=(2+eta)**2*s*s-16*(1+eta)*p
    return math.sqrt(((2+eta)*s+math.sqrt(max(0.,d)))/4)


def own_ground_fraction_for_off(fa,fb,fc):
    a,b=_frequencies(fa,fb)
    if not math.isfinite(fc) or fc<=max(fa,fb):raise ValueError('Target OFF must exceed both rail frequencies')
    x=fc*fc
    return 2*(x-a)*(x-b)/((a+b)*x-2*a*b)


def seed_record(fa,fb,on,off,upper=10.):
    eta=own_ground_fraction_for_off(fa,fb,off)
    return dict(rail_frequencies_ghz=[fa,fb],on_ghz=on,off_target_ghz=off,
                frequency_order_valid=bool(on<off<upper),required_own_ground_fraction=eta,
                ground_to_mutual_capacitance_ratio=2*eta/(1-eta) if eta<1 else None,
                within_minimal_weak_link_family=bool(0<eta<=1),
                estimated_family_upper_off_ghz=same_pad_off_frequency(fa,fb,1.))
