"""Fit only complete, converged GREEN curves and retain every physical check."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
from scipy.constants import physical_constants
from scipy.optimize import least_squares

CONVERSION=physical_constants['Hartree energy'][0]/1e-21
EXPECTED_A=[5.35,5.4375,5.525,5.6125,5.7,5.7875,5.875,5.9625,6.05]
CASES=[(method,mode,backend) for method,modes in [('GW',['cpu','gpu']),('GF2',['cpu','hybrid'])]
       for mode in modes for backend in ('ccgdf','rsgdf')]

def bm(volume,e0,v0,b0,bp):
    x=(v0/np.asarray(volume))**(2/3)
    return e0+9*v0*b0/16*((x-1)**3*bp+(x-1)**2*(6-4*x))

def pressure(volume,fit):
    ratio=fit['v0_angstrom3']/np.asarray(volume)
    return 1.5*fit['b_e_gpa']*(ratio**(7/3)-ratio**(5/3))*(1+.75*(fit['bprime']-4)*(ratio**(2/3)-1))

def fit_curve(v,e):
    v,e=np.asarray(v),np.asarray(e)
    assert np.all(np.diff(v)>0) and np.argmin(e) not in (0,len(e)-1)
    offset=e.min();relative=e-offset
    initial=[0,v[np.argmin(e)],75/CONVERSION,4]
    solve=least_squares(lambda p:bm(v,*p)-relative,initial,
        bounds=([-1,v.min(),1e-8,-20],[1,v.max(),1,40]),
        x_scale=[.01,45,.02,4],ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=5000)
    assert solve.success,solve.message
    e0,v0,b0,bp=solve.x
    assert v.min()<v0<v.max() and b0>0
    assert not np.any(solve.active_mask),'Fit is constrained by a bound'
    residual=bm(v,*solve.x)-relative
    return {'e0_hartree':float(e0+offset),'v0_angstrom3':float(v0),'a0_angstrom':float((4*v0)**(1/3)),
            'b_e_gpa':float(b0*CONVERSION),'b0_hartree_angstrom3':float(b0),'bprime':float(bp),
            'rms_residual_hartree':float(np.sqrt(np.mean(residual**2))),
            'max_residual_hartree':float(abs(residual).max()),'npoints':len(v),'minimum_bracketed':True}

def analyze(stage='matched-refined'):
    global EXPECTED_A
    root=Path(os.environ['STUDY_RUNS']);out=root/('analysis-stock' if stage=='stock-validation' else 'analysis');out.mkdir(exist_ok=True)
    window=root/'eos-window.json'
    if window.exists():
        configuration=json.loads(window.read_text())
        EXPECTED_A=configuration['a_angstrom']
        assert len(EXPECTED_A)==len(set(EXPECTED_A)) and EXPECTED_A==sorted(EXPECTED_A)
    expected_count=len(EXPECTED_A)*len(CASES)
    records=[];missing=[];fits=[];pair_checks=[];bridges=[]
    for a in EXPECTED_A:
        folder=root/stage/f'a{a:.6f}'
        path=folder/'comparison.json'
        if path.exists(): pair_checks += json.loads(path.read_text())['comparisons']
        path=folder/'canonical-bridges.json'
        if path.exists(): bridges += json.loads(path.read_text())
        for method,mode,backend in CASES:
            path=folder/f'{method.lower()}-{mode}-{backend}'/'run.json'
            if not path.exists(): missing.append(str(path));continue
            record=json.loads(path.read_text())
            if record['status']!='PASS' or not record['converged']:
                missing.append(str(path));continue
            assert (record['a_angstrom'],record['method'],record['mode'],record['backend'])==(a,method,mode,backend)
            if stage=='stock-validation':
                assert record['consumer_kind']=='unmodified upstream main executable'
                assert record['fixed_chemical_potential'] is None
            else:
                assert record['consumer_kind']=='public-API fixed-mu diagnostic with unchanged kernels'
                assert record['fixed_chemical_potential'] is not None
            records.append({'a_angstrom':a,'volume_angstrom3':a**3/4,'method':method,'mode':mode,'backend':backend,
                'total_energy_hartree':record['numerical']['last']['total_energy_hartree'],
                'correlation_energy_hartree':record['numerical']['last']['Energy_2b'],
                'electron_number_error':record['numerical']['electron_number_error'],
                'mu_hartree':record['numerical']['last']['mu'],'iteration':record['numerical']['iteration'],
                'energy_change_hartree':record['numerical']['independent_energy_change_hartree'],
                'binary_sha256':record['binary_sha256'],'grid_sha256':record['grid_sha256'],
                'model':record['model'],'path':str(path.parent)})
    for method,mode,backend in CASES:
        group=sorted([r for r in records if (r['method'],r['mode'],r['backend'])==(method,mode,backend)],key=lambda r:r['a_angstrom'])
        if len(group)<7: continue
        assert len({r['binary_sha256'] for r in group})==1 and len({r['grid_sha256'] for r in group})==1
        v=np.array([r['volume_angstrom3'] for r in group]);e=np.array([r['total_energy_hartree'] for r in group])
        entry={'method':method,'mode':mode,'backend':backend,'points':group}
        if np.argmin(e) in (0,len(e)-1): entry.update(status='MINIMUM_NOT_BRACKETED');fits.append(entry);continue
        # Bulk curvature is local to each method's minimum. All raw points
        # remain in the numerical comparison, including distant SCF behavior.
        center_a=group[int(np.argmin(e))]['a_angstrom']
        center_index=EXPECTED_A.index(center_a);count=min(7,len(EXPECTED_A))
        start=max(0,min(center_index-count//2,len(EXPECTED_A)-count))
        selected_a=EXPECTED_A[start:start+count]
        lookup={r['a_angstrom']:r for r in group}
        if not all(a in lookup for a in selected_a):continue
        selected_group=[lookup[a] for a in selected_a]
        fit_v=np.array([r['volume_angstrom3'] for r in selected_group])
        fit_e=np.array([r['total_energy_hartree'] for r in selected_group])
        entry.update(status='FITTED',fit=fit_curve(fit_v,fit_e),
                     fit_points=selected_group,fit_window_rule='Seven nearest grid points around the discrete minimum, clamped at the common-grid boundary',
                     complete_common_grid=len(group)==len(EXPECTED_A))
        entry['fit']['volume_min_angstrom3']=float(fit_v.min())
        entry['fit']['volume_max_angstrom3']=float(fit_v.max())
        try:entry['broad_window_diagnostic']=fit_curve(v,e)
        except (AssertionError,ValueError) as error:entry['broad_window_diagnostic']={'status':'UNQUALIFIED','reason':str(error)}
        windows=[]
        for chosen in (slice(1,None),slice(None,-1),slice(1,-1)):
            if len(fit_v[chosen])>=5 and np.argmin(fit_e[chosen]) not in (0,len(fit_e[chosen])-1):
                windows.append(fit_curve(fit_v[chosen],fit_e[chosen]))
        entry['fit']['window_spread_b_e_gpa']=max([abs(w['b_e_gpa']-entry['fit']['b_e_gpa']) for w in windows] or [0])
        entry['fit']['window_spread_a0_angstrom']=max([abs(w['a0_angstrom']-entry['fit']['a0_angstrom']) for w in windows] or [0])
        entry['fit']['window_sensitivity_count']=len(windows)
        entry['fit']['window_sensitivity_fits']=windows
        center=fit_v.mean();coeff=np.polyfit(fit_v-center,fit_e-fit_e.min(),4)
        roots=np.roots(np.polyder(coeff));candidates=[float(r.real) for r in roots if abs(r.imag)<1e-7 and fit_v.min()<r.real+center<fit_v.max() and np.polyval(np.polyder(coeff,2),r.real)>0]
        if candidates:
            x=min(candidates,key=lambda x:np.polyval(coeff,x));pv=x+center
            entry['quartic_cross_check']={'a0_angstrom':float((4*pv)**(1/3)),
                'b_e_gpa':float(pv*np.polyval(np.polyder(coeff,2),x)*CONVERSION)}
        fits.append(entry)
    differences=[]
    mapping={(r['method'],r['mode'],r['backend']):r for r in fits if r['status']=='FITTED'}
    for method in ('GW','GF2'):
        grids={tuple(p['a_angstrom'] for p in f['fit_points']) for f in mapping.values() if f['method']==method}
        assert len(grids)<=1,'All backend/device fits for a method must use the same volumes'
    for method,modes in [('GW',['cpu','gpu']),('GF2',['cpu','hybrid'])]:
        pairs=[((method,m,'ccgdf'),(method,m,'rsgdf'),'backend') for m in modes]
        pairs +=[((method,modes[0],b),(method,modes[1],b),'device') for b in ('ccgdf','rsgdf')]
        for lk,rk,kind in pairs:
            if lk not in mapping or rk not in mapping: continue
            left,right=mapping[lk],mapping[rk]
            de=np.array([r['total_energy_hartree']-l['total_energy_hartree'] for l,r in zip(left['points'],right['points'])])
            da=right['fit']['a0_angstrom']-left['fit']['a0_angstrom'];db=right['fit']['b_e_gpa']-left['fit']['b_e_gpa']
            differences.append({'method':method,'kind':kind,'left':list(lk),'right':list(rk),
                'delta_a0_angstrom':da,'delta_b_e_gpa':db,'max_abs_energy_difference_hartree':float(abs(de).max()),
                'energy_difference_peak_to_peak_hartree':float(np.ptp(de)),
                'status':'PASS' if abs(da)<1e-4 and abs(db)<.1 and abs(de).max()<1e-7 else 'FAIL'})
    ready=(not missing and len(fits)==8 and all(f['status']=='FITTED' for f in fits)
           and len(pair_checks)==expected_count and all(c['status']=='PASS' for c in pair_checks)
           and len(bridges)==expected_count and all(c['status']=='PASS' for c in bridges)
           and len(differences)==8 and all(c['status']=='PASS' for c in differences))
    result={'status':'PASS' if ready else 'INCOMPLETE_OR_FAILED','model':'Ge GTH-SZV / GTH-PBE valence, 2x2x2',
        'stage':stage,
        'scope':'Finite-model GREEN backend/device validation, not an experimental accuracy claim',
        'energy_kind':'GREEN correlated internal energy including nuclear repulsion',
        'curvature_label':'B_E at beta=1000 Hartree^-1; not a Helmholtz isothermal modulus',
        'point_count':len(records),'expected_point_count':expected_count,'a_angstrom':EXPECTED_A,'missing_or_failed':missing,
        'conversion_hartree_angstrom3_to_gpa':CONVERSION,'fits':fits,'fit_comparisons':differences,
        'physical_comparison_count':len(pair_checks),'physical_comparison_failures':[c for c in pair_checks if c['status']!='PASS'],
        'canonical_bridge_count':len(bridges),'canonical_bridge_failures':[c for c in bridges if c['status']!='PASS'],
        'points':records,'max_abs_electron_number_error':max([abs(r['electron_number_error']) for r in records] or [0]),
        'max_abs_canonical_bridge_energy_difference_hartree':max([abs(b['energy_difference_to_canonical_hartree']) for b in bridges] or [0])}
    (out/'correlated-eos-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if records:
        with (out/'correlated-energies.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print('CORRELATED_EOS_ANALYSIS',result['status'],len(records),'points',len(fits),'fits',flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['matched-refined','stock-validation'],default='matched-refined')
    options=parser.parse_args()
    v=np.linspace(38,54,9)
    check=fit_curve(v,bm(v,-7,45.2,75/CONVERSION,4.2))
    assert abs(check['v0_angstrom3']-45.2)<1e-7 and abs(check['b_e_gpa']-75)<1e-5
    h=.001;x=47.3
    numerical_pressure=-(bm(x+h,-7,45.2,75/CONVERSION,4.2)-bm(x-h,-7,45.2,75/CONVERSION,4.2))/(2*h)*CONVERSION
    assert abs(float(pressure(x,check))-numerical_pressure)<1e-6
    h=.01;x=45.2
    numerical_curvature=(bm(x+h,-7,x,75/CONVERSION,4.2)-2*bm(x,-7,x,75/CONVERSION,4.2)+bm(x-h,-7,x,75/CONVERSION,4.2))/h**2*x*CONVERSION
    assert abs(numerical_curvature-75)<1e-4
    analyze(options.stage)
