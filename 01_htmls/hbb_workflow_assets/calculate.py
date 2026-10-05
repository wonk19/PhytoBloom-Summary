"""SBAC attenuation-only sensitivity using Zhang et al. 2021 Eq. 25.

Geometry and depth bins: author's public readme2.m. W_T ratio: bb_weighting_Z.m.
Integrate each cone-overlap cross section with Gauss quadrature instead of pixels.
This is B(c), NOT the phase-function-dependent B(c_m), and NOT Hyper-BB calibration.
ACS saved c is used as a numerical model input, without adding water or subtracting
background. Recover the c-channel wavelengths from the saved dataframe (the shared
wl array in these pickles is the a-channel wavelength grid).
"""
from pathlib import Path
import json
import pickle
import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parent
DATA = Path('C:/Codes/PhytoBloom_1.4/results/PB12_0730_SBAC')
WL = np.array([430, 440, 490, 550, 620, 675, 700])

def ellipse(z, pos, zenith, half):
    # All author geometries lie in x-z plane; signed zenith accounts for azimuth.
    t, h = np.deg2rad([zenith, half])
    zz = z - pos[2]
    aa = np.cos(h)**2 - np.sin(t)**2
    center = pos[0] + zz*np.sin(t)*np.cos(t)/aa
    rx = zz*np.sin(h)*np.cos(h)/aa
    ry = zz*np.sin(h)/np.sqrt(aa)
    return center, rx, ry

def factor(c, sensor, nx=160, ny=24):
    if sensor == 'HydroScat-6':
        source, detector = (0,0,0), (.1926,0,-.1126)
        zs, hs, zd, hd, face = 19,2.5,-19,2.5,.15354
        edges = np.linspace(face,.30,600)
    else:
        source, detector = (0,0,0), (.01655,0,.00386)
        zs, hs, zd, hd, face = 30,17.5,-31.56,22.5,.00655
        # Exact MATLAB colon range from author's readme (ends near 0.1 m).
        logs = np.arange(np.log10(face), -1+1e-12, .001)
        edges = 10**logs
    gx, wx = leggauss(nx)
    gy, wy = leggauss(ny)
    c = np.r_[0.,np.asarray(c).ravel()]
    sums = np.zeros((2,len(c)))
    for z in (edges[:-1]+edges[1:])/2:
        sx,sa,sb = ellipse(z,source,zs,hs)
        dx,da,db = ellipse(z,detector,zd,hd)
        lo,hi = max(sx-sa,dx-da),min(sx+sa,dx+da)
        if lo >= hi:
            continue
        x = (lo+hi)/2 + gx*(hi-lo)/2
        ymax = np.sqrt(np.minimum(sb**2*np.maximum(0,1-((x-sx)/sa)**2),
                                 db**2*np.maximum(0,1-((x-dx)/da)**2)))
        y = ymax[:,None]*gy
        weights = wx[:,None]*(hi-lo)/2*ymax[:,None]*wy
        for j,pos in enumerate([source,detector]):
            r = np.sqrt((x[:,None]-pos[0])**2+y**2+(z-pos[2])**2)
            cosz = (z-pos[2])/r
            path = (z-face)/cosz
            domega = weights*cosz/r**2
            sums[j] += np.sum(domega[None,:,:]*np.exp(-c[:,None,None]*path),axis=(1,2))
    # All c-independent prefactors Vt / At / Nt cancel in W_T(0)/W_T(c).
    wt = sums[0]*sums[1]
    out = wt[0]/wt
    assert out[0] == 1 and np.all(out >= 1)
    return out[1:]

def main():
    rows=[]
    for sample in ['DIL1A','DIL1B']:
        with (DATA/f'0730_SBAC_{sample}.pkl').open('rb') as f:
            acs=pickle.load(f)['acs_fts']
        cols=[k for k in acs['df'].columns if k.startswith('c') and k[1:2].isdigit()]
        cwl=np.array([float(k[1:]) for k in cols])
        assert len(cwl)==len(acs['c'])
        values=np.interp(WL,cwl,acs['c'])
        for wl,c in zip(WL,values):
            rows.append(dict(sample=sample,wavelength_nm=int(wl),c_m_inv=float(c)))
    cs=np.array([r['c_m_inv'] for r in rows])
    validation={}
    for sensor in ['HydroScat-6','ECO-BB']:
        low=factor(cs,sensor,80)
        high=factor(cs,sensor,240)
        validation[sensor]={'max_relative_quadrature_difference':float(np.max(abs(low/high-1)))}
        for row,v in zip(rows,high):
            row[sensor]=float(v)
    result={'definition':'B(c) = W_T(0)/W_T(c); single-scattering attenuation-only scenario',
            'source_doi':'10.1364/AO.437735','validation':validation,'rows':rows}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
