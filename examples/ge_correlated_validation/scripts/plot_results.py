"""Scientific figures from verified small JSON evidence; no electronic solver."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

EH_MEV=27211.386245988
STYLE={('cpu','ccgdf'):('#202e3b','o','-'),('cpu','rsgdf'):('#008571','x','--'),
       ('gpu','ccgdf'):('#c67b28','s',':'),('gpu','rsgdf'):('#4266aa','+','-.'),
       ('hybrid','ccgdf'):('#c67b28','s',':'),('hybrid','rsgdf'):('#4266aa','+','-.')}

def bm(v,f):
    x=(f['v0_angstrom3']/np.asarray(v))**(2/3)
    return f['e0_hartree']+9*f['v0_angstrom3']*f['b0_hartree_angstrom3']/16*((x-1)**3*f['bprime']+(x-1)**2*(6-4*x))

def pressure(v,f):
    x=f['v0_angstrom3']/np.asarray(v)
    return 1.5*f['b_e_gpa']*(x**(7/3)-x**(5/3))*(1+.75*(f['bprime']-4)*(x**(2/3)-1))

def main(options):
    data=json.loads(options.input.read_text());out=options.output;out.mkdir(parents=True,exist_ok=True)
    assert data['status']=='PASS','No final EOS figure is made from an incomplete study'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.titlesize':18,
                         'axes.labelsize':16,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42})
    fits=[f for f in data['fits'] if f['status']=='FITTED']
    pdf=PdfPages(out/'green-correlated-ge-figures.pdf')
    foot='Ge GTH-SZV / GTH-PBE valence model, 2×2×2 k mesh, β = 1000 Ha⁻¹. Small-basis validation model.'
    def save(fig,name,note=None):
        caption=(note+'\n' if note else '')+foot
        fig.text(.5,.014,caption,ha='center',fontsize=8,color='#52616c')
        fig.tight_layout(rect=[0,.08 if note else .045,1,.94]);fig.savefig(out/(name+'.png'),dpi=180);pdf.savefig(fig);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.7))
    fig.suptitle('GREEN correlated energy–volume curves',fontsize=17,color='#36566f')
    for ax,method in zip(axes,['GF2','GW']):
        group=[f for f in fits if f['method']==method]
        baseline=next(f for f in group if f['mode']=='cpu' and f['backend']=='ccgdf')['fit']['e0_hartree']
        for item in group:
            color,marker,line=STYLE[item['mode'],item['backend']]
            v=np.array([p['volume_angstrom3'] for p in item['points']]);e=np.array([p['total_energy_hartree'] for p in item['points']])
            grid=np.linspace(item['fit']['volume_min_angstrom3'],item['fit']['volume_max_angstrom3'],250)
            label=item['backend'].upper()+' '+item['mode'].upper()
            ax.plot(grid/2,(bm(grid,item['fit'])-baseline)*EH_MEV/2,color=color,ls=line,lw=1.4,label=label)
            ax.plot(v/2,(e-baseline)*EH_MEV/2,marker=marker,ls='none',color=color,ms=5,mfc='none')
        ax.set(title=method,xlabel='Volume per atom (Å³)',ylabel=r'$E-E_0$ (meV/atom)')
        ax.grid(alpha=.16);ax.legend(fontsize=11)
    save(fig,'green-correlated-ev-curves','Markers: every accepted volume. Lines: local seven-point Birch–Murnaghan fits.')
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.7))
    fig.suptitle('Pressure from the correlated internal-energy curves',fontsize=17,color='#36566f')
    for ax,method in zip(axes,['GF2','GW']):
        for item in [f for f in fits if f['method']==method]:
            color,marker,line=STYLE[item['mode'],item['backend']]
            v=np.linspace(item['fit']['volume_min_angstrom3'],item['fit']['volume_max_angstrom3'],250)
            ax.plot(v/2,pressure(v,item['fit']),color=color,ls=line,lw=1.5,label=item['backend'].upper()+' '+item['mode'].upper())
        ax.axhline(0,color='#777777',lw=.7)
        ax.set(title=method,xlabel='Volume per atom (Å³)',ylabel=r'$P_E$ (GPa)')
        ax.grid(alpha=.16);ax.legend(fontsize=11)
    save(fig,'green-correlated-pressure','Pressure is shown only inside each method’s local fitting window.')
    points=data['points']
    for method in ['GF2','GW']:
        fig,axes=plt.subplots(1,2,figsize=(11.8,4.7),sharex=True)
        fig.suptitle(method+': backend and device energy differences',fontsize=17,color='#36566f')
        device='hybrid' if method=='GF2' else 'gpu'
        lookup={(p['mode'],p['backend'],p['a_angstrom']):p for p in points if p['method']==method}
        aa=sorted({p['a_angstrom'] for p in points if p['method']==method})
        for mode,color in [('cpu','#008571'),(device,'#4266aa')]:
            delta=[lookup[mode,'rsgdf',a]['total_energy_hartree']-lookup[mode,'ccgdf',a]['total_energy_hartree'] for a in aa]
            axes[0].plot(aa,np.array(delta)*EH_MEV*1000/2,'o-',color=color,label=mode.upper())
        for backend,color in [('ccgdf','#202e3b'),('rsgdf','#c67b28')]:
            delta=[lookup[device,backend,a]['total_energy_hartree']-lookup['cpu',backend,a]['total_energy_hartree'] for a in aa]
            axes[1].plot(aa,np.array(delta)*EH_MEV*1000/2,'o-',color=color,label=backend.upper())
        for col in range(2):
            axes[col].axhline(0,color='#777777',lw=.6);axes[col].grid(alpha=.16)
            axes[col].legend(fontsize=11);axes[col].set_ylabel('ΔE (µeV/atom)')
            axes[col].set_xlabel('Lattice constant (Å)')
        axes[0].set_title('RSGDF − CCGDF');axes[1].set_title(device.upper()+' − CPU')
        save(fig,'green-correlated-energy-differences-'+method.lower())
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.7))
    fig.suptitle('Equilibrium lattice constant and energy curvature',fontsize=17,color='#36566f')
    labels=[]
    for i,item in enumerate(fits):
        color,marker,_=STYLE[item['mode'],item['backend']]
        labels.append(item['method']+'\n'+item['backend'].upper()+'\n'+item['mode'].upper())
        axes[0].errorbar(i,item['fit']['a0_angstrom'],yerr=item['fit']['window_spread_a0_angstrom'],fmt=marker,color=color,capsize=3)
        axes[1].errorbar(i,item['fit']['b_e_gpa'],yerr=item['fit']['window_spread_b_e_gpa'],fmt=marker,color=color,capsize=3)
    for ax in axes:
        ax.set_xticks(range(len(fits)),labels,fontsize=7);ax.grid(axis='y',alpha=.16)
    axes[0].set_ylabel(r'$a_0$ (Å)')
    axes[1].set_ylabel(r'$B_E$ (GPa)')
    axes[0].set_title('Lattice minimum')
    axes[1].set_title('Energy curvature')
    save(fig,'green-correlated-eos-properties','Error bars indicate fit-window sensitivity, not statistical uncertainty.')
    runs=options.runs or options.input.parent.parent/'runs/compact'
    gw=json.loads((runs/'canonical-refined/a5.700000/gw-cpu-ccgdf/run.json').read_text())
    gf2=json.loads((runs/'canonical-gf2-damped/a5.700000/gf2-cpu-ccgdf/run.json').read_text())
    expanded=json.loads((runs/'canonical-refined/a6.225000/gw-cpu-ccgdf/run.json').read_text())
    prior=runs/'canonical-refined/a6.225000/history/gw-cpu-ccgdf-linear-300/run.json'
    if prior.exists():
        history=json.loads(prior.read_text())['numerical']['history']
        offset=history[-1]['iteration']
        expanded['numerical']['history']=history+[dict(r,iteration=r['iteration']+offset) for r in expanded['numerical']['history']]
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.7))
    fig.suptitle('Convergence diagnostics',fontsize=17,color='#36566f')
    for record,label,color in [(gw,'GW a=5.7 Å','#008571'),(expanded,'GW a=6.225 Å','#c67b28')]:
        history=record['numerical']['history'];final=history[-1]['total_energy_hartree']
        axes[0].plot([r['iteration'] for r in history],[(r['total_energy_hartree']-final)*EH_MEV/2 for r in history],color=color,label=label)
    axes[0].set(title='GW energy trajectories',xlabel='Continuation iteration',ylabel=r'$E-E_{\rm final}$ (meV/atom)')
    axes[0].legend(fontsize=11)
    for record,label,color in [(gw,'GW a=5.7 Å','#008571'),(expanded,'GW a=6.225 Å','#c67b28'),(gf2,'GF2 a=5.7 Å','#4266aa')]:
        h=record['numerical']['history']
        residual=[sum(abs(b[k]-a[k]) for k in ('Energy_1b','Energy_HF','Energy_2b')) for a,b in zip(h,h[1:])]
        axes[1].semilogy([r['iteration'] for r in h[1:]],np.maximum(residual,1e-16),color=color,label=label)
    axes[1].set(title='Energy-component residual',xlabel='Continuation iteration',ylabel=r'$\sum_i |\Delta E_i|$ (Ha)')
    axes[1].legend(fontsize=11)
    for ax in axes:ax.grid(alpha=.16)
    save(fig,'green-correlated-convergence')
    pdf.close()
    (out/'render-provenance.json').write_text(json.dumps({'input_sha256':hashlib.sha256(options.input.read_bytes()).hexdigest(),
        'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__,'pdf_pages':6,
        'input_status':data['status'],'energy_source_stage':data['stage']},indent=2)+'\n')
    print('SCIENTIFIC_FIGURES_WRITTEN',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runs',type=Path,help='Actual compact run root for rendering on the workstation; omit for an extracted evidence tree')
    main(p.parse_args())
