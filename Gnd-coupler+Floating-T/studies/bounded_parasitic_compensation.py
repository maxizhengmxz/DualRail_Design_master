"""Full charging-matrix fit with a bound on every physical capacitor change."""
from copy import deepcopy
import time

import numpy as np
from scipy.optimize import minimize

from floating_coupler.mixed_electrostatics import fit_mixed_shunts
from floating_coupler.parasitics import DEFAULT_PARASITIC_CAPACITANCES_FF


def compensate_bounded(nominal, ports, amount, max_fraction=.05,
                       match_rtol=None, excluded_mode_pairs=()):
    """Preserve loaded ECs and equal rail grounds with positive capacitors.

    With match_rtol, constrain intended links and the matched matrix norm,
    minimize the largest fractional capacitor movement, then its squared sum.
    SLSQP establishes a feasible local fit, not a global-minimum certificate.
    match_rtol=None retains the older error-minimizing diagnostic objective.
    """
    from studies.parasitic_absorption import rebuild
    if not np.isfinite(amount) or amount<=0 or not 0<max_fraction<1:
        raise ValueError('Supply a positive parasitic and a fractional bound between zero and one')
    if ports not in DEFAULT_PARASITIC_CAPACITANCES_FF:
        raise ValueError('Unknown parasitic capacitor '+ports)
    start = time.perf_counter()
    _, circuit = rebuild(nominal)
    nodes = circuit['node_order']
    embedding = circuit['embedding']
    target = circuit['K']
    scale = np.sqrt(np.outer(np.diag(target), np.diag(target)))
    upper = np.triu_indices(len(target))
    matched = np.ones(target.shape,dtype=bool)
    for left,right in excluded_mode_pairs:
        i,j = circuit['modal_order'].index(left),circuit['modal_order'].index(right)
        if i==j:
            raise ValueError('Diagonal entries must remain matched')
        matched[i,j] = matched[j,i] = False
    selected_upper = matched[upper]
    if match_rtol is not None and (not np.isfinite(match_rtol) or match_rtol<=0):
        raise ValueError('Matching tolerance must be finite and positive')

    def incidence(a, b=None):
        direction = np.zeros(len(nodes))
        if a in nodes:
            direction[nodes.index(a)] = 1
        if b in nodes:
            direction[nodes.index(b)] -= 1
        return direction

    parameters = []
    for pair in nominal['pad_pairs'].values():
        value = nominal['ground_capacitances_ff'][pair[0]]
        if value:
            direction = sum(np.outer(incidence(pad),incidence(pad)) for pad in pair)
            parameters.append(('ground', tuple(pair), value, direction))
    for index, edge in enumerate(nominal['physical_links']):
        direction = incidence(edge['from'],edge['to'])
        parameters.append(('link', index, edge['capacitance_ff'], np.outer(direction,direction)))
    for name, value in nominal['resonator_self_capacitances_ff'].items():
        direction = incidence(name)
        parameters.append(('resonator',name,value,np.outer(direction,direction)))
    for name, pair in nominal['pad_pairs'].items():
        direction = incidence(*pair)
        parameters.append(('self',name,nominal['pad_mutual_capacitances_ff'][name],np.outer(direction,direction)))
    matrices = np.asarray([value*direction for _,_,value,direction in parameters])
    a,b = ports.split('-')
    direction = incidence(a,b)
    fixed = circuit['nodal_capacitance_ff'] + amount*np.outer(direction,direction)
    parasitic = dict(name='Cpar_'+ports, **{'from':a,'to':b},capacitance_ff=amount,parasitic=True)
    targets = dict(zip(nominal['pad_pairs'],circuit['loaded_ec_ghz']))
    seed,_ = fit_mixed_shunts(nominal['pad_pairs'],nominal['ground_capacitances_ff'],
                              [*nominal['physical_links'],parasitic],targets,
                              resonator_caps_ff=nominal['resonator_self_capacitances_ff'])
    initial = np.zeros(len(parameters))
    for index,(category,key,value,_) in enumerate(parameters):
        if category=='self':
            initial[index] = seed[key]/value-1
    if max(abs(initial))>max_fraction:
        raise ValueError('The EC-preserving starting point exceeds the capacitor-change budget')

    def evaluate(x):
        capacitance = fixed + np.einsum('i,ijk->jk',x,matrices)
        response = np.linalg.solve(capacitance,embedding)
        charging = embedding.T@response
        derivative = -np.einsum('ai,nab,bj->nij',response,matrices,response)
        return charging,derivative

    def objective(x):
        charging,derivative = evaluate(x)
        residual = ((charging-target)/scale)[upper][selected_upper]*1000
        jacobian = (derivative/scale)[:,upper[0],upper[1]].T[selected_upper]*1000
        return residual@residual,2*residual@jacobian

    def equality(x):
        charging,_ = evaluate(x)
        return np.diag(charging)[:5]/np.diag(target)[:5]-1

    def equality_jacobian(x):
        _,derivative = evaluate(x)
        return np.diagonal(derivative,axis1=1,axis2=2)[:,:5].T/np.diag(target)[:5,None]

    if match_rtol is None:
        combined_movement_converged = None
        fit = minimize(objective,initial,jac=True,method='SLSQP',
                       bounds=[(-max_fraction,max_fraction)]*len(parameters),
                       constraints=[dict(type='eq',fun=equality,jac=equality_jacobian)],
                       options=dict(ftol=1e-12,maxiter=500))
        fractions = fit.x
    else:
        combined_movement_converged = False
        # Minimize the largest physical change once an engineering match
        # tolerance has been explicitly supplied. All seven diagonals stay fixed.
        intended = (('A1','A2'),('A2','C'),('C','B1'),('A2','B1'),('B1','B2'),
                    ('A1','r1'),('A2','r1'),('B1','r2'),('B2','r2'))
        indices = np.asarray([(circuit['modal_order'].index(a),circuit['modal_order'].index(b))
                              for a,b in intended])
        denominators = abs(target[indices[:,0],indices[:,1]])
        if np.any(denominators==0):
            raise ValueError('Zero nominal intended coupling needs an absolute matching tolerance')
        matrix_budget = match_rtol**2*np.sum(target[matched]**2)
        require_matrix_budget = np.sum(matched)>len(target)+2*len(indices)
        count = len(parameters)

        def movement(x):
            return x[-1]+1e-8*(x[:-1]@x[:-1]),np.r_[2e-8*x[:-1],1.]

        def diagonal(x):
            charging,_ = evaluate(x[:-1])
            return np.diag(charging)/np.diag(target)-1

        def diagonal_jacobian(x):
            _,derivative = evaluate(x[:-1])
            jacobian = np.diagonal(derivative,axis1=1,axis2=2).T/np.diag(target)[:,None]
            return np.column_stack((jacobian,np.zeros(len(target))))

        def tolerance_constraints(x):
            charging,_ = evaluate(x[:-1])
            error = charging-target
            relative = error[indices[:,0],indices[:,1]]/denominators
            residual = np.r_[x[-1]-x[:-1],x[-1]+x[:-1],match_rtol-relative,match_rtol+relative]
            if require_matrix_budget:
                residual = np.r_[residual,1-np.sum(error[matched]**2)/matrix_budget]
            return residual

        def tolerance_jacobian(x):
            charging,derivative = evaluate(x[:-1])
            relative = derivative[:,indices[:,0],indices[:,1]].T/denominators[:,None]
            relative = np.column_stack((relative,np.zeros(len(indices))))
            matrix_row = -2*((charging-target)[matched]@derivative[:,matched].T)/matrix_budget
            jacobian = np.vstack((np.column_stack((-np.eye(count),np.ones(count))),
                                  np.column_stack((np.eye(count),np.ones(count))),
                                  -relative,relative))
            if require_matrix_budget:
                jacobian = np.vstack((jacobian,np.r_[matrix_row,0.]))
            return jacobian

        initial = np.r_[initial,min(max_fraction,max(abs(initial))+.01)]
        fit = minimize(movement,initial,jac=True,method='SLSQP',
                       bounds=[(-max_fraction,max_fraction)]*count+[(0,max_fraction)],
                       constraints=[dict(type='eq',fun=diagonal,jac=diagonal_jacobian),
                                    dict(type='ineq',fun=tolerance_constraints,jac=tolerance_jacobian)],
                       options=dict(ftol=1e-11,maxiter=500))
        if fit.success:
            # Break minimax ties by reducing combined physical-capacitor movement.
            ceiling = min(max_fraction,fit.x[-1]+max(1e-10,fit.x[-1]*1e-6))
            start_secondary = fit.x.copy()
            start_secondary[-1] = ceiling

            def combined_movement(x):
                return x[:-1]@x[:-1],np.r_[2*x[:-1],0.]

            secondary = minimize(combined_movement,start_secondary,jac=True,method='SLSQP',
                                 bounds=[(-ceiling,ceiling)]*count+[(ceiling,ceiling)],
                                 constraints=[dict(type='eq',fun=diagonal,jac=diagonal_jacobian),
                                              dict(type='ineq',fun=tolerance_constraints,jac=tolerance_jacobian)],
                                 options=dict(ftol=1e-12,maxiter=200))
            if (secondary.success and max(abs(diagonal(secondary.x)))<=1e-9
                    and min(tolerance_constraints(secondary.x))>=-1e-10):
                fit = secondary
                combined_movement_converged = True
        fractions = fit.x[:-1]
        if fit.success and (max(abs(diagonal(fit.x)))>1e-9 or min(tolerance_constraints(fit.x)) < -1e-10):
            raise ValueError('Minimum-change solution failed its explicit matching constraints')
    if not fit.success:
        raise ValueError('Bounded capacitor compensation did not converge: '+fit.message)
    physical = deepcopy(nominal)
    changes = []
    for (category,key,value,_),fraction in zip(parameters,fractions):
        delta = float(value*fraction)
        if category=='ground':
            labels = list(key)
            for pad in key:
                physical['ground_capacitances_ff'][pad] = value+delta
        elif category=='link':
            edge = physical['physical_links'][key]
            edge['capacitance_ff'] = value+delta
            labels = [edge['from']+'-'+edge['to']]
        else:
            field = 'pad_mutual_capacitances_ff' if category=='self' else 'resonator_self_capacitances_ff'
            physical[field][key] = value+delta
            labels = [key]
        for label in labels:
            changes.append(dict(component=category+':'+label,nominal_ff=value,
                                compensated_ff=value+delta,delta_ff=delta,percent_change=float(100*fraction)))
    physical['physical_links'].append(parasitic)
    physical,rebuilt = rebuild(physical)
    ec_error = float(max(abs(rebuilt['loaded_ec_ghz']-circuit['loaded_ec_ghz']))*1e9)
    if ec_error>=1:
        raise ValueError('Bounded capacitor fit did not preserve loaded ECs to 1 Hz')
    error = rebuilt['K']-target
    return dict(parasitic=parasitic,physical=physical,changes=changes,
                optimizer_success=True,optimizer_message=fit.message,evaluations=fit.nfev,
                all_cap_fractional_bound=max_fraction,loaded_ec_error_hz=ec_error,
                match_rtol=match_rtol,excluded_mode_pairs=list(excluded_mode_pairs),
                combined_movement_converged=combined_movement_converged,
                matched_relative_frobenius_K_error=float(np.linalg.norm(error[matched])/np.linalg.norm(target[matched])),
                relative_frobenius_K_error=float(np.linalg.norm(error)/np.linalg.norm(target)),
                normalized_rms_K_error=float(np.sqrt(np.mean((error/scale)[upper]**2))),
                seconds=time.perf_counter()-start)
