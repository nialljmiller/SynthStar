#!/usr/bin/env python3
"""Deterministic space-filling coverage of the existing non-MESA generators.

These are template coverage samples, NOT occurrence-rate or population priors.
Generate on Newton, which has the five baseline evolutionary SED tracks.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def samples(n, d, seed):
    return qmc.Sobol(d, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n]


def logspan(u, low, high):
    return float(np.exp(np.log(low) + u * np.log(high/low)))


def item(family, definition, **info):
    definition = {k: float(v) if isinstance(v, np.floating) else v for k, v in definition.items()}
    key = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()[:14]
    definition = dict(name=f"libnm_{family}_{key}", **definition)
    return dict(family=family, definition=definition, **info)


def dwarf_relation():
    rows = []
    for line in (ROOT/'data/EEM_dwarf_UBVIJHK_colors_Teff.txt').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        try:
            mass, teff, radius = float(f[-2]), float(f[1]), float(f[6])
            if mass > 0 and radius > 0:
                rows.append((mass, teff, radius))
        except (ValueError, IndexError):
            continue
    rows = np.array(sorted(rows))
    assert len(rows) > 20
    def at(mass):
        return [float(np.exp(np.interp(np.log(mass), np.log(rows[:,0]), np.log(rows[:,i])))) for i in (1,2)]
    return at


def roche_fraction(q):
    return 0.49*q**(2/3)/(0.6*q**(2/3)+np.log1p(q**(1/3)))


def orbit(m1,m2,r1,r2,u):
    ecc = 0.0 if u[0] < 0.5 else float((u[0]-0.5)*1.6)
    per0 = float(360*u[1])
    a_min = max(r1/roche_fraction(m1/m2), r2/roche_fraction(m2/m1))/(1-ecc)
    sma = a_min * logspan(u[2],1.15,8.0)
    period = 2*np.pi*np.sqrt((sma*6.957e10)**3/(6.674e-8*(m1+m2)*1.989e33))/86400
    if period < 3:
        ecc = 0.0  # short-period circular subset
    conjunction = sma*(1-ecc**2)/(1+ecc*np.sin(np.radians(per0)))
    # Mostly eclipse/grazing coverage; retain 20% less inclined controls.
    if u[3] < 0.8:
        b = 1.1*u[3]/0.8
        cosi = min(0.99,b*(r1+r2)/conjunction)
    else:
        cosi = 0.2+0.7*(u[3]-0.8)/0.2
    return dict(period=float(period),incl=float(np.degrees(np.arccos(cosi))),ecc=ecc,per0=per0)


def binaries():
    at = dwarf_relation()
    from synthstar.binary import ms_mass, sma_rsun
    result = []
    for u in samples(512,7,203):
        m1 = logspan(u[0],0.65,12)
        m2 = 0.65+(m1-0.65)*u[1]
        t1,r1 = at(m1); t2,r2 = at(m2)
        p = orbit(m1,m2,r1,r2,u[2:6])
        result.append(item('detached_ms',dict(generator='detached EB',m1=m1,m2=m2,
                           teff1=t1,teff2=t2,r1=r1,r2=r2,distance=10.0,mh=0.0,**p),
                           physics='paired empirical solar-metallicity unevolved main-sequence stars'))
    contact = []
    for u in samples(8192,7,204):
        t1 = float(4400+3100*u[0]); t2 = float(t1*(0.88+0.14*u[1]))
        q = float(0.12+0.88*u[2]); period=logspan(u[3],0.22,1.0)
        m1 = ms_mass(t1); sma=sma_rsun(period,m1*(1+q))
        rl = sma*roche_fraction(1/q)
        _,ms_r = at(m1)
        if not 0.65 <= rl/ms_r <= 2.0:
            continue
        incl = float(np.degrees(np.arccos(0.35*u[4] if u[4]<0.8 else 0.3+0.6*(u[4]-0.8)/0.2)))
        contact.append(item('contact',dict(generator='W UMa',period=period,q=q,incl=incl,
                            teff1=t1,teff2=t2,fillout=float(0.02+0.78*u[5]),distance=10.0,mh=0.0),
                            physics='Roche contact; common-envelope temperatures; approximate MS-density prefilter'))
        if len(contact)==512:
            break
    assert len(contact)==512
    result += contact
    # Evolved primaries and lower-mass companions at the SAME age and metallicity.
    import h5py
    tracks = {}
    for path in (ROOT/'runs').glob('*_sed/seds.h5'):
        with h5py.File(path) as f:
            if abs(float(f.attrs['feh'])) > 1e-6:
                continue
            mass=float(f.attrs['initial_mass'])
            tracks[mass]={n:np.asarray(f['model/'+n]) for n in
                          ['star_age','star_mass','Teff','log_R','mh','phase','inside_atm_grid','log_g']}
    pairs=[(a,b) for a in sorted(tracks) for b in sorted(tracks) if a>b and a>=3]
    evolved=[]
    for u in samples(8192,7,205):
        m1,m2=pairs[min(int(u[0]*len(pairs)),len(pairs)-1)]
        a,b=tracks[m1],tracks[m2]
        good=np.flatnonzero((a['phase']>0)&(a['inside_atm_grid']>0))
        if not len(good): continue
        i=good[min(int(u[1]*len(good)),len(good)-1)]
        age=float(a['star_age'][i])
        if not b['star_age'][0]<=age<=b['star_age'][-1]: continue
        j=int(np.searchsorted(b['star_age'],age)); j=min(max(1,j),len(b['star_age'])-1)
        if not np.all(b['inside_atm_grid'][j-1:j+1]): continue
        interp=lambda n:float(np.interp(age,b['star_age'],b[n]))
        r1=float(10**a['log_R'][i]);r2=10**interp('log_R')
        mm1=float(a['star_mass'][i]);mm2=interp('star_mass')
        p=orbit(mm1,mm2,r1,r2,u[2:6])
        if p['period']>3000: continue
        evolved.append(item('detached_evolved',dict(generator='detached EB',m1=mm1,m2=mm2,
                            teff1=float(a['Teff'][i]),teff2=interp('Teff'),r1=r1,r2=r2,
                            distance=10.0,mh=float(a['mh'][i]),**p),
                            physics='coeval components from existing solar-metallicity MESA tracks',age_years=age))
        if len(evolved)==256: break
    assert len(evolved)==256
    return result+evolved


def parametrics():
    result=[]
    for u in samples(1024,7,206):
        result.append(item('microlensing',dict(generator='microlensing',tE=logspan(u[0],0.01,500),
                           u0=logspan(u[1],0.001,3),base=float(8+17*u[2]),bp_rp=float(-0.3+2.6*u[3]),
                           rho=0.0 if u[4]<0.25 else logspan((u[4]-0.25)/0.75,1e-5,0.1),
                           fs=logspan(u[5],0.01,1))))
    for u in samples(512,10,207):
        mdq=logspan(u[0],1e-13,3e-11)
        mdp=max(10*mdq,logspan(u[1],1e-10,3e-8))
        incl=float(np.degrees(np.arccos(1-u[9]*(1-np.cos(np.radians(80))))))
        result.append(item('dwarf_nova',dict(generator='dwarf nova',d=logspan(u[2],100,10000),
                           mdot_q=mdq,mdot_peak=mdp,t_rise=logspan(u[3],0.05,3),plateau=float(20*u[4]),
                           tau_decay=logspan(u[5],0.1,20),r_out=logspan(u[6],0.1,0.8),
                           t_wd=float(7000+43000*u[7]),m_wd=float(0.5+0.8*u[8]),
                           incl=incl),
                           physics='phenomenological steady-disc outburst; not disc-instability hydrodynamics'))
    return result


def supernovae():
    from synthstar.transients import catalogue
    latest={}
    for name,version,kind,reference in catalogue():
        latest[name]=(version,kind,reference)
    result=[]; excluded=[]
    for n,(version,kind,ref) in sorted(latest.items()):
        if kind=='PopIII':
            excluded.append(dict(model=n,version=version,reason='high-z PopIII requires IGM treatment not supported by this optical generator'))
            continue
        peak=-19.3 if kind=='SN Ia' else -18.5 if kind in ('SN IIn','SN Ic-BL') else -17.0
        if '91bg' in n: peak=-17.5
        if '91t' in n: peak=-19.5
        if 'superc' in n: peak=-20.0
        seed=int(hashlib.sha256(n.encode()).hexdigest()[:7],16)
        for u in samples(16,3,seed)[:12]:
            result.append(item('sncosmo',dict(generator='supernova',model=n,version=str(version),
                               z=logspan(u[0],0.001,0.3),M=float(peak+(u[1]-0.5)*(1.2 if kind=='SN Ia' else 3.0)),
                               host_ebv=float(0.7*u[2]),band='bessellb',magsys='ab',params={}),
                               source_type=kind,reference=ref))
        if n.startswith('salt'):
            for u in samples(64,5,seed+1):
                x1=float(-3+6*u[0]);c=float(-0.25+0.65*u[1])
                result.append(item('salt_shape',dict(generator='supernova',model=n,version=str(version),
                                   z=logspan(u[2],0.001,0.3),M=float(-19.3-0.14*x1+3*c+0.5*(u[3]-0.5)),
                                   host_ebv=float(0.5*u[4]),params=dict(x1=x1,c=c),band='bessellb',magsys='ab'),
                                   source_type=kind,reference=ref,
                                   physics='SALT shape/colour coverage; colour-host dust degeneracy; not a population prior'))
    return result,excluded


def generate():
    all_items=binaries()+parametrics()
    sn,excluded=supernovae();all_items+=sn
    from synthstar.sources import check
    seen=set()
    for job in all_items:
        gen,keys=check(job['definition'])
        assert gen.engine not in ('MESA RSP','MESA evolution')
        assert job['definition']['name'] not in seen
        seen.add(job['definition']['name'])
    return dict(version=1,created='2026-10-20',jobs=all_items,excluded_sources=excluded,
                population_warning='Space-filling template coverage, not astrophysical occurrence probabilities; distance/dust/cadence/selection must be applied separately.')


if __name__=='__main__':
    from collections import Counter
    m=generate()
    out=ROOT/'data/library/nonmesa_20261020.json';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and json.loads(out.read_text())!=m:
        raise SystemExit(f'refusing to change existing manifest {out}')
    out.write_text(json.dumps(m,indent=1)+'\n')
    from synthstar.tomlfile import table
    folder=ROOT/'examples/stars/nonmesa_20261020';folder.mkdir(parents=True,exist_ok=True)
    grouped={f:[j['definition'] for j in m['jobs'] if j['family']==f] for f in {j['family'] for j in m['jobs']}}
    for family,es in grouped.items():
        for start in range(0,len(es),32):
            text='# Non-MESA coverage templates; not population weights.\n\n'+'\n'.join(s for e in es[start:start+32] for s in table('[[star]]',e))
            (folder/f'{family}_{start//32:03d}.toml').write_text(text)
    print(out)
    print(json.dumps(Counter(j['family'] for j in m['jobs']),indent=2))
    print('TOTAL',len(m['jobs']),'EXCLUDED_SOURCES',len(m['excluded_sources']))
