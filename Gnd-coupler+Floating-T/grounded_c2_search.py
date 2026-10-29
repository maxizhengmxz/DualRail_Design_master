"""Exact mixed circuit: floating A2/B1, C2 directly grounded, C1 dynamical."""
import json,argparse,sys
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.optimize import least_squares,minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.transmon import Basis
from floating_coupler.model import Model,SCALE
from floating_coupler.pad_electrostatics import build_pad_circuit
from staged_three_body import two_excitation,single_excitation,frequency_ej
from floating_coupler.channel_policy import selection_from_request,off_target_definition


def mixed_matrix(grounds,links,shunts):
    # C2 is removed from the dynamical nodes, not assigned a large capacitor.
    names=['A21','A22','C1','B11','B12'];index={n:i for i,n in enumerate(names)}
    c=np.diag([grounds['A21'],grounds['A22'],0.,grounds['B11'],grounds['B12']])
    def edge(a,b,value):
        i,j=index[a],index[b];c[i,i]+=value;c[j,j]+=value;c[i,j]-=value;c[j,i]-=value
    for e in links:edge(e['from'],e['to'],e['capacitance_ff'])
    edge('A21','A22',shunts[0]);c[2,2]+=shunts[1];edge('B11','B12',shunts[2])
    p=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,0,1],[0,0,-1.]])
    k=p.T@np.linalg.solve(c,p)
    return c,k


def fit_circuit(ca,cb,direct,ec,request):
    grounds={n:request['rail_ground_fraction']*2*SCALE/e for ns,e in [(['A21','A22'],.25),(['B11','B12'],.16)] for n in ns}
    links=[{'from':a,'to':b,'capacitance_ff':v} for a,b,v in [('A22','C1',ca),('C1','B11',cb),('A22','B11',direct)]]
    target=np.array([.25,ec,.16]);starts=SCALE/target*np.array([.5,.7,.5])
    def residual(x):return (SCALE*np.diag(mixed_matrix(grounds,links,np.exp(x))[1])-target)/target
    fit=least_squares(residual,np.log(starts),bounds=(-20,20),xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=50)
    if max(abs(residual(fit.x)))>1e-8:raise ValueError('No positive grounded-C shunt fit')
    shunts=np.exp(fit.x);matrix,k=mixed_matrix(grounds,links,shunts)
    _,without=mixed_matrix(grounds,[links[2]],shunts)
    loading=1-np.diag(k)/np.diag(without)
    j=8*SCALE*k;np.fill_diagonal(j,0.)
    ej=np.array([frequency_ej(f,e,request['ncut']) for f,e in zip([4.61,8.,5.39],target)])
    model=Model(('A2','C','B1'),target,ej,j,Basis(ncut=request['ncut'],transmon_levels=request['levels']),{})
    physical={'grounded_pad':'C2','rail_ground_capacitances_ff':grounds,'physical_links':links,'A2_across_pad_ff':shunts[0],'C1_total_own_capacitance_to_ground_ff':shunts[1],'B1_across_pad_ff':shunts[2],'own_coupler_fraction':float(shunts[1]/(SCALE/ec)),'node_order':['A21','A22','C1','B11','B12'],'nodal_capacitance_ff':matrix.tolist(),'J_ghz':j.tolist()}
    return model,loading,physical


def fixed_comparison(row,request):
    e=row['config']['electrostatics'];c=build_pad_circuit(e['pad_pairs'],e['ground_capacitances_ff'],{},e['physical_links'],e['pad_mutual_capacitances_ff'])
    keep=[i for i,n in enumerate(c.nodes) if n!='C2'];matrix=c.nodal_capacitance_ff[np.ix_(keep,keep)];p=c.charge_embedding[keep,:]
    k=p.T@np.linalg.solve(matrix,p);ecs=SCALE*np.diag(k);j=8*SCALE*k;np.fill_diagonal(j,0.)
    ej=np.array([row['config']['transmons'][n]['ej_ghz'] for n in ['A2','C','B1']]);model=Model(('A2','C','B1'),ecs,ej,j,Basis(ncut=request['ncut'],transmon_levels=request['levels']),{})
    from floating_coupler.model import local_spectrum
    freqs=[float(local_spectrum(a,b,model.basis)[0][1]) for a,b in zip(ej,ecs)]
    from scipy.optimize import brentq
    from floating_coupler.off_search import exchange_roots
    roots=exchange_roots(lambda f:single_excitation(model,f)['g_mhz'],np.linspace(6.5,10,25))
    return {'fixed_hardware_loaded_ec_mhz':(ecs*1000).tolist(),'fixed_EJ_bare_frequencies_ghz':freqs,'off_roots_with_fixed_rail_EJs_ghz':roots}


def run(source,output):
    previous=json.loads(Path(source).read_text());req=previous['request']
    selection=selection_from_request(req)
    if selection.auxiliary!='B20':raise ValueError('This reduced grounded-C2 search implements the B20 channel only')
    from staged_three_body import speed_seed
    current_seed=speed_seed(req)
    target=current_seed['three_body_11_to_B1_two_seed_mhz'];data={'off_definition':off_target_definition('three'),'scope':'grounded C2 reduced-model comparison; no convergence or five-body gate','source':source,'request':req,'channel_selection':selection.to_dict(),'speed_seed':current_seed,'target_two_excitation_mhz':target,'fixed_hardware_comparison':{},'attempts':[],'selected':{}}
    path=Path(output)
    if path.exists():
        from floating_coupler.channel_policy import require_off_definition
        require_off_definition(json.loads(path.read_text()).get('off_definition'),'three')
    for lower in [6.5,7.]:
        old=previous['selected'][str(lower)];data['fixed_hardware_comparison'][str(lower)]=fixed_comparison(old,req)
        for q in [.25,.5,1.,2.,4.]:
            capscale=np.sqrt(old['parameters'][0]*old['parameters'][1]);x0=np.array([np.log(capscale*np.sqrt(q)),np.log(capscale/np.sqrt(q)),np.log(old['parameters'][2]),old['parameters'][3],9.95,lower+.001])
            bounds=[(np.log(.05),np.log(120)),(np.log(.05),np.log(120)),(np.log(1e-9),np.log(5)),(.15,.6),(lower+.002,9.999),(lower+.001,9.998)]
            cache={}
            def evaluate(x):
                key=tuple(x)
                if key in cache:return cache[key]
                p=np.array(x);p[:3]=np.exp(p[:3]);model,loads,physical=fit_circuit(*p[:4],req)
                off=single_excitation(model,p[4]);model.ej[2]=frequency_ej(4.76,.16,req['ncut']);on=two_excitation(model,p[5])
                value=(p,loads,physical,off,on);cache[key]=value;return value
            def objective(x):
                try:return float(evaluate(x)[1][[0,2]].sum())
                except ValueError:return 10.
            def eq(x):
                try:
                    p,l,ph,off,on=evaluate(x);return np.array([off['g_mhz'],abs(on['g_mhz'])-target])/target
                except ValueError:return np.ones(2)*100
            def ine(x):
                try:
                    p,l,ph,off,on=evaluate(x);return np.r_[np.array([.1,.44999,.1])-l,p[4]-p[5]-.0001,off['minimum_overlap']-.8,on['minimum_overlap']-.8]
                except ValueError:return np.ones(6)*-100
            fit=minimize(objective,x0,method='SLSQP',bounds=bounds,constraints=[{'type':'eq','fun':eq},{'type':'ineq','fun':ine}],options={'maxiter':100,'ftol':1e-10,'eps':2e-6})
            p,l,ph,off,on=evaluate(fit.x);ok=bool(max(abs(eq(fit.x)))<1e-4 and min(ine(fit.x))>=-1e-7)
            row={'version_lower_ghz':lower,'parameters':p.tolist(),'loading_A2_C_B1':l.tolist(),'physical':ph,'off':off,'on':on,'feasible':ok,'optimizer_success':bool(fit.success),'objective':objective(fit.x)};data['attempts'].append(row)
            if ok and (str(lower) not in data['selected'] or row['objective']<data['selected'][str(lower)]['objective']):data['selected'][str(lower)]=row
            path.write_text(json.dumps(data,indent=2));print(lower,q,ok,p,l,flush=True)
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.source,a.output)
