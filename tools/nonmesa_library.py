#!/usr/bin/env python3
"""Run the non-MESA coverage library in isolated, single-threaded lanes.

One PHOEBE lane and one transient lane keep cheap families out of the MESA
backlog. Run inside the documented CPU/memory-limited systemd user service.
No MESA model is launched by this runner. Resume skips accepted/failed jobs.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'

import argparse
from collections import Counter,defaultdict,deque
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime
import fcntl
import json
import multiprocessing
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/nonmesa_library_20261020'
MANIFEST=ROOT/'data/library/nonmesa_20261020.json'
SOURCE_FAILURES={}
TEMPLATE_CACHE=None


def cached_templates():
    """Reuse quality-filtered ordinary-star spectra in the transient worker."""
    global TEMPLATE_CACHE
    if TEMPLATE_CACHE is None:
        import h5py
        import numpy as np
        from synthstar.seds import Templates
        template=Templates()
        masks=[]
        for p in sorted((ROOT/'runs').glob('*_sed/seds.h5')):
            with h5py.File(p) as f:
                good=(np.asarray(f['model/inside_atm_grid'])>0)&(np.asarray(f['model/phase'])>=0)
                ratio=np.asarray(f['model/fbol_sed_over_L'])
                good &= np.isfinite(ratio)&(ratio>0.9)&(ratio<1.1)
                masks.append(good)
        keep=np.concatenate(masks)
        assert len(keep)==len(template.colour)
        for key in ('flux','colour','g','feh'):
            setattr(template,key,getattr(template,key)[keep])
        if not len(template.colour): raise ValueError('no usable ordinary-star template spectra')
        TEMPLATE_CACHE=template
    return TEMPLATE_CACHE


def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,indent=1,allow_nan=False)+'\n')
    os.replace(temp,path)


def state_path(job):
    return OUT/'status'/(job['definition']['name']+'.json')


def read_state(job):
    p=state_path(job)
    return json.loads(p.read_text()) if p.exists() else dict(state='pending')


def initialise_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    signal.signal(signal.SIGALRM,lambda *args: (_ for _ in ()).throw(TimeoutError('15-minute forward-model budget exceeded')))


def validate(path,definition):
    import h5py
    import numpy as np
    with h5py.File(path) as f:
        w=np.asarray(f['wavelength']);t=np.asarray(f['time_days']);flux=np.asarray(f['flux'])
        if flux.shape!=(len(t),len(w)) or len(t)<2 or not np.all(np.diff(t)>0) or not np.all(np.diff(w)>0):
            raise ValueError('invalid spectral time-series dimensions or axes')
        if not np.isfinite(flux).all() or not np.isfinite(t).all() or not np.isfinite(w).all():
            raise ValueError('non-finite SED/axis values')
        if w[0]>3600 or w[-1]<8300 or not np.any(flux>0):
            raise ValueError('SED lacks positive optical flux or full Argus optical wavelength coverage')
        # Empirical SN spectra can have noisy negative individual samples. Do not
        # silently clip them; require nonnegative broad-band integrals instead.
        for low,high in ((3600,5250),(5600,8300)):
            use=(w>=low)&(w<=high)
            integrated=np.trapezoid(flux[:,use],w[use],axis=1)
            if integrated.min() < -1e-5*max(float(abs(integrated).max()),1e-300):
                raise ValueError(f'negative integrated optical flux in {low}-{high} A')
        if definition['generator'] in ('W UMa','detached EB'):
            from synthstar.history import inside_atmosphere_grid
            a=f.attrs
            grav=[np.log10(6.674e-8*float(a[m])*1.989e33/(float(a[r])*6.957e10)**2)
                  for m,r in (('mass','R1'),('mass2','R2'))]
            if not inside_atmosphere_grid(np.array([a['teff1'],a['teff2']]),np.array(grav),
                                           np.full(2,definition.get('mh',0.0))).all():
                raise ValueError('binary component outside supported atmosphere grid')
        return dict(time_samples=len(t),wavelength_samples=len(w),bytes=path.stat().st_size,
                    negative_spectral_sample_fraction=float(np.mean(flux<0)),
                    min_time_days=float(t[0]),max_time_days=float(t[-1]))


def prepare_sn_window(path):
    """Retain a contiguous physically usable phase window containing optical peak.

    Empirical template interpolation can become negative near phase edges.
    Never clip or extrapolate flux. Reject if less than half the supplied phase
    interval is usable, or if the optical peak itself has invalid broad flux.
    """
    import h5py
    import numpy as np
    with h5py.File(path,'r+') as f:
        w=np.asarray(f['wavelength']);t=np.asarray(f['time_days']);flux=np.asarray(f['flux'])
        good=np.isfinite(flux).all(axis=1);bands=[]
        for low,high in ((3600,5250),(5600,8300)):
            use=(w>=low)&(w<=high)
            bands.append(np.trapezoid(flux[:,use],w[use],axis=1))
        peak=int(np.nanargmax(bands[0]+bands[1]))
        for band in bands:
            good &= band>=-1e-5*max(float(np.nanmax(abs(band))),1e-300)
        if not good[peak]: raise ValueError('supernova optical peak has invalid flux')
        left=peak;right=peak
        while left>0 and good[left-1]: left-=1
        while right+1<len(t) and good[right+1]: right+=1
        if left==0 and right==len(t)-1: return
        if not left<peak<right or t[right]-t[left]<0.5*(t[-1]-t[0]):
            raise ValueError('supernova lacks a sufficiently broad valid phase window around peak')
        f.attrs['quality_original_mintime']=float(t[0]);f.attrs['quality_original_maxtime']=float(t[-1])
        f.attrs['quality_phase_window_trimmed']=True
        del f['time_days'];del f['flux']
        f.create_dataset('time_days',data=t[left:right+1])
        f.create_dataset('flux',data=flux[left:right+1],compression='gzip',compression_opts=4)


def one(job):
    start=time.time();name=job['definition']['name'];status=state_path(job)
    running=dict(state='running',name=name,family=job['family'],pid=os.getpid(),
                 started=datetime.now().isoformat(timespec='seconds'))
    atomic(status,running)
    log=OUT/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    stage=OUT/'staging'/name
    try:
        with log.open('w',buffering=1) as output,redirect_stdout(output),redirect_stderr(output):
            signal.alarm(900)
            from synthstar.sources import check
            from synthstar.tomlfile import table
            import make_stars
            gen,keys=check(job['definition'])
            if gen.engine in ('MESA RSP','MESA evolution') or gen.prepare is not None:
                raise ValueError('non-MESA runner refuses MESA/network-preparation generators')
            if gen.engine=='PHOEBE':
                import phoebe
                phoebe.multiprocessing_off();phoebe.mpi_off()
            if gen.name=='microlensing':
                from synthstar import transients
                transients.Templates=cached_templates
                template=cached_templates()
                if not float(template.colour.min())<=keys['bp_rp']<=float(template.colour.max()):
                    raise ValueError('source colour outside quality-filtered stellar template coverage')
            if gen.engine=='sncosmo':
                import sncosmo
                key=(keys['model'],keys['version'])
                if key in SOURCE_FAILURES:
                    raise RuntimeError('source unavailable earlier in this worker: '+SOURCE_FAILURES[key])
                try:
                    sncosmo.get_source(keys['model'],version=keys['version'] or None)
                except Exception as err:
                    SOURCE_FAILURES[key]=str(err)
                    raise
            definition=dict(name=name,generator=gen.name,**keys)
            target=ROOT/'stars'/name
            written={k:v for k,v in definition.items() if v is not None}
            if (target/'star.toml').exists():
                import tomllib
                if tomllib.loads((target/'star.toml').read_text())['star']!=written:
                    raise ValueError('existing model has a different definition')
                qc=validate(target/'sed.h5',definition)
            else:
                if target.exists():
                    raise ValueError('existing incomplete output requires manual review')
                make_stars.STARS=OUT/'staging'
                if not make_stars.build_star('star',gen,definition,keys):
                    raise ValueError('generator did not produce a model')
                if gen.engine=='sncosmo': prepare_sn_window(stage/'sed.h5')
                qc=validate(stage/'sed.h5',definition)
                atomic(stage/'coverage.json',dict(family=job['family'],selection=job,
                                                quality=qc,population_weight=None))
                os.replace(stage,target)
            final=dict(running,state='done',elapsed_seconds=round(time.time()-start,2),quality=qc)
    except (Exception,SystemExit) as err:
        with log.open('a') as output:
            traceback.print_exc(file=output)
        final=dict(running,state='failed',elapsed_seconds=round(time.time()-start,2),
                   error=f'{type(err).__name__}: {err}')
    finally:
        signal.alarm(0)
    atomic(status,final)
    return final


def interleave(jobs):
    groups=defaultdict(deque)
    for j in jobs: groups[j['family']].append(j)
    result=[]
    while any(groups.values()):
        for f in sorted(groups):
            if groups[f]: result.append(groups[f].popleft())
    return result


def snapshot(jobs):
    grouped=defaultdict(Counter);counts=Counter();active=[]
    for j in jobs:
        d=read_state(j);s=d['state'];counts[s]+=1;grouped[j['family']][s]+=1
        if s=='running': active.append({k:d[k] for k in ('name','family','pid','started')})
    data=dict(updated=datetime.now().isoformat(timespec='seconds'),total=len(jobs),
              counts=dict(counts),families={f:dict(c) for f,c in grouped.items()},active=active)
    atomic(OUT/'summary.json',data)
    return data


def memory_available():
    info=Path('/proc/meminfo').read_text().splitlines()
    return int(next(x.split()[1] for x in info if x.startswith('MemAvailable:')))*1024


def run(jobs,pilot=False):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'runner.lock').open('a') as lock:
        print('Waiting for exclusive non-MESA runner lock (pilots finish first).',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        todo=[j for j in jobs if read_state(j)['state'] not in ('done','failed')]
        if pilot:
            selected=[];seen=set()
            for j in interleave(todo):
                if j['family'] not in seen:
                    selected.append(j);seen.add(j['family'])
            todo=selected
        lanes={
            'binary':deque(interleave([j for j in todo if j['definition']['generator'] in ('W UMa','detached EB')])),
            'transient':deque(interleave([j for j in todo if j['definition']['generator'] not in ('W UMa','detached EB')])),
        }
        ctx=multiprocessing.get_context('spawn')
        capacity={'binary':1 if pilot else 2,'transient':1}
        pools={lane:ProcessPoolExecutor(max_workers=capacity[lane],mp_context=ctx,initializer=initialise_worker) for lane in lanes}
        active={}
        try:
            while any(lanes.values()) or active:
                busy=Counter(active.values())
                for lane,queue in lanes.items():
                    if queue and busy[lane]<capacity[lane] and memory_available()>8*1024**3 and shutil.disk_usage(ROOT).free>100*1024**3:
                        j=queue.popleft()
                        active[pools[lane].submit(one,j)]=lane
                        print(datetime.now().isoformat(timespec='seconds'),'start',j['definition']['name'],flush=True)
                if active:
                    ready,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
                    for future in ready:
                        result=future.result();del active[future]
                        print(result['state'],result['name'],result['elapsed_seconds'],result.get('error',''),flush=True)
                else:
                    time.sleep(15)
                snapshot(jobs)
        finally:
            for pool in pools.values(): pool.shutdown(wait=True,cancel_futures=True)
        print(json.dumps(snapshot(jobs),indent=1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['status','run','pilot','validate-existing'])
    args=p.parse_args()
    if args.command=='validate-existing':
        import tomllib
        for n in ['test_wuma','test_detached_eb','test_microlensing','test_dwarf_nova','test_sn_ia','test_sn_iip']:
            d=tomllib.loads((ROOT/'stars'/n/'star.toml').read_text())['star']
            print(n,validate(ROOT/'stars'/n/'sed.h5',d))
    else:
        jobs=json.loads(MANIFEST.read_text())['jobs']
        if args.command=='status': print(json.dumps(snapshot(jobs),indent=1))
        else: run(jobs,pilot=args.command=='pilot')
