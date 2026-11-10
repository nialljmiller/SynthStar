// SExtractor photometry of the page's exposures: a port of SEP 1.4.1 (Barbary 2016; SExtractor's algorithms, Bertin &
// Arnouts 1996) as synthstar/photometry.py calls it, and of photometry.py itself. Runs in the browser and in node
// (demo/validate/sep.mjs checks it against SEP on the same images).
//
// The C code works in single precision (PIXTYPE float) in most places; Math.fround reproduces that where it decides
// which pixels belong to which object (thresholds, convolution, background mesh).

const SEP = (() => {
  const f = Math.fround, BIG = 1e30;
  const OBJ_MERGED = 0x0001, OBJ_TRUNC = 0x0002, OBJ_SINGU = 0x0008, APER_TRUNC = 0x0010;

  // ================================================================ background (background.c)
  const QUANTIF_NSIGMA = 5, QUANTIF_NMAXLEVELS = 4096, QUANTIF_AMIN = 4;

  // sep.Background(data, bw, bh, fw, fh) -> {back (w*h, float values), global, globalrms}
  function background(data, w, h, bw = 64, bh = 64, fw = 3, fh = 3) {
    const nx = Math.max(1, Math.floor((w - 1) / bw) + 1), ny = Math.max(1, Math.floor((h - 1) / bh) + 1), nb = nx * ny;
    const back = new Float32Array(nb), sigma = new Float32Array(nb);
    const step = Math.sqrt(2 / Math.PI) * QUANTIF_NSIGMA / QUANTIF_AMIN;
    for (let j = 0; j < ny; j++) {
      const y0 = j * bh, mh = Math.min(bh, h - y0);
      for (let m = 0; m < nx; m++) {
        const x0 = m * bw, mw = (m === nx - 1 && w % bw) ? w % bw : bw;
        // backstat: mean and sigma, then again within mean +- 2 sigma
        let mean = 0, sig2 = 0, npix = 0;
        for (let y = y0; y < y0 + mh; y++) for (let x = x0; x < x0 + mw; x++) {
          const pix = f(data[y * w + x]);
          if (pix > -BIG) { mean += pix; sig2 += pix * pix; npix++; }
        }
        const k = m + nx * j;
        if (f(npix) < f(mw * mh * 0.5)) { back[k] = sigma[k] = -BIG; continue; }
        mean /= npix;
        let s = sig2 / npix - mean * mean, sd = s > 0 ? Math.sqrt(s) : 0;
        const lcut = f(mean - 2 * sd), hcut = f(mean + 2 * sd);
        mean = 0; sig2 = 0; npix = 0;
        for (let y = y0; y < y0 + mh; y++) for (let x = x0; x < x0 + mw; x++) {
          const pix = f(data[y * w + x]);
          if (pix <= hcut && pix >= lcut) { mean += pix; sig2 += pix * pix; npix++; }
        }
        mean /= npix;
        s = sig2 / npix - mean * mean;
        sd = s > 0 ? Math.sqrt(s) : 0;
        const bm = { mean: f(mean), sigma: f(sd), nlevels: Math.min(Math.trunc(step * npix + 1), QUANTIF_NMAXLEVELS) };
        bm.qscale = f(sd > 0 ? 2 * QUANTIF_NSIGMA * sd / bm.nlevels : 1);
        bm.qzero = f(mean - QUANTIF_NSIGMA * sd);
        // backhisto
        const histo = new Float64Array(bm.nlevels), cste = f(0.499999 - f(bm.qzero / bm.qscale));
        for (let y = y0; y < y0 + mh; y++) for (let x = x0; x < x0 + mw; x++) {
          const bin = Math.trunc(f(f(f(data[y * w + x]) / bm.qscale) + cste));
          if (bin >= 0 && bin < bm.nlevels) histo[bin]++;
        }
        const g = backguess(bm, histo);
        back[k] = g.mean; sigma[k] = g.sigma;
      }
    }
    const { global, globalrms } = filterback(back, sigma, nx, ny, fw, fh);
    const dback = makebackspline(back, nx, ny);
    return { back: backArray(back, dback, nx, ny, w, h, bw, bh), global, globalrms };
  }

  function backguess(bm, histo) {
    const EPS = 1e-4, nlevelsm1 = bm.nlevels - 1;
    let lcut = 0, hcut = nlevelsm1, sig = 10 * nlevelsm1, sig1 = 1, mea = bm.mean, med = bm.mean;
    const at = (i) => histo[i] ?? 0;
    for (let n = 100; n-- && sig >= 0.1 && Math.abs(sig / sig1 - 1) > EPS;) {
      sig1 = sig;
      let sum = 0, lowsum = 0, highsum = 0;
      mea = sig = 0;
      let hilow = lcut, hihigh = hcut;
      for (let i = lcut; i <= hcut; i++) {
        if (lowsum < highsum) lowsum += at(hilow++);
        else highsum += at(hihigh--);
        const pix = at(i);
        sum += pix;
        const dpix = pix * i;
        mea += dpix;
        sig += dpix * i;
      }
      med = hihigh >= 0 ? hihigh + 0.5 + (highsum - lowsum) / (2 * (at(hilow) > at(hihigh) ? at(hilow) : at(hihigh))) : 0;
      if (sum) { mea /= sum; sig = sig / sum - mea * mea; }
      sig = sig > 0 ? Math.sqrt(sig) : 0;
      let t = med - 3 * sig;
      lcut = t > 0 ? Math.trunc(t > 0 ? t + 0.5 : t - 0.5) : 0;
      t = med + 3 * sig;
      hcut = t < nlevelsm1 ? Math.trunc(t > 0 ? t + 0.5 : t - 0.5) : nlevelsm1;
    }
    const mean = Math.abs(sig) > 0
      ? (Math.abs((mea - med) / sig) < 0.3 ? bm.qzero + (2.5 * med - 1.5 * mea) * bm.qscale : bm.qzero + med * bm.qscale)
      : bm.qzero + mea * bm.qscale;
    return { mean: f(mean), sigma: f(sig * bm.qscale) };
  }

  // fqmedian: the median of float values, (a + b) / 2 for an even count
  function fqmedian(values) {
    const a = Float32Array.from(values).sort(), n = a.length;
    return n < 2 ? a[0] : (n & 1 ? a[n >> 1] : f((a[n / 2 - 1] + a[n / 2]) / 2));
  }

  // median filter of the mesh map (fthresh 0: every mesh takes its filtered value); the window shrinks at the edges
  function filterback(back, sigma, nx, ny, fw, fh) {
    const np = nx * ny, npx = fw >> 1, npy = fh >> 1;
    for (let i = 0; i < np; i++) { // bad meshes take the mean of the nearest good ones
      if (back[i] > -BIG) continue;
      const px = i % nx, py = Math.floor(i / nx);
      let d2min = BIG, val = 0, sval = 0, nmin = 0;
      for (let j = 0; j < np; j++) {
        if (!(back[j] > -BIG)) continue;
        const d2 = f((j % nx - px) ** 2 + (Math.floor(j / nx) - py) ** 2);
        if (d2 < d2min) { val = back[j]; sval = sigma[j]; nmin = 1; d2min = d2; }
        else if (d2 === d2min) { val = f(val + back[j]); sval = f(sval + sigma[j]); nmin++; }
      }
      back[i] = nmin ? f(val / nmin) : 0;
      sigma[i] = nmin ? f(sval / nmin) : 1;
    }
    const back2 = new Float32Array(np), sigma2 = new Float32Array(np);
    for (let py = 0; py < ny; py++) {
      const npy2 = Math.min(npy, ny - py - 1, py);
      for (let px = 0; px < nx; px++) {
        const npx2 = Math.min(npx, nx - px - 1, px), bm = [], sm = [];
        for (let dy = -npy2; dy <= npy2; dy++) for (let dx = -npx2; dx <= npx2; dx++) {
          bm.push(back[(py + dy) * nx + px + dx]); sm.push(sigma[(py + dy) * nx + px + dx]);
        }
        back2[py * nx + px] = fqmedian(bm);
        sigma2[py * nx + px] = fqmedian(sm);
      }
    }
    back.set(back2); sigma.set(sigma2);
    let globalrms = fqmedian(sigma2);
    if (globalrms <= 0) {
      const pos = Array.from(sigma2).filter((v) => v > 0);
      globalrms = pos.length ? fqmedian(pos) : 1; // (SEP takes the trailing positive run; meshes are never 0 here)
    }
    return { global: fqmedian(back2), globalrms };
  }

  // natural cubic spline second derivatives along y at the mesh nodes
  function makebackspline(map, nbx, nby) {
    const dmap = new Float32Array(nbx * nby);
    if (nby < 2) return dmap;
    const u = new Float32Array(nby);
    for (let x = 0; x < nbx; x++) {
      dmap[x] = 0; u[0] = 0;
      for (let y = 1; y < nby - 1; y++) {
        let temp = f(-1 / f(dmap[(y - 1) * nbx + x] + 4));
        dmap[y * nbx + x] = temp;
        temp = f(temp * f(u[y - 1] - f(6 * f(f(map[(y + 1) * nbx + x] + map[(y - 1) * nbx + x]) - f(2 * map[y * nbx + x])))));
        u[y] = temp;
      }
      dmap[(nby - 1) * nbx + x] = 0;
      for (let y = nby - 2; y >= 1; y--) {
        dmap[y * nbx + x] = f(f(f(dmap[y * nbx + x] * dmap[(y + 1) * nbx + x]) + u[y]) / 6);
      }
    }
    return dmap;
  }

  // the full-resolution background: bicubic spline between the mesh nodes, line by line (bkg_line_flt_internal)
  function backArray(values, dvalues, nbx, nby, width, height, bw, bh) {
    const out = new Float32Array(width * height), nbxm1 = nbx - 1;
    const node = new Float32Array(nbx), dnode = new Float32Array(nbx), u = new Float32Array(nbx);
    for (let y = 0; y < height; y++) {
      let nodeArr = node, dnodeArr = dnode;
      if (nby > 1) {
        let dy = f(f(f(y) / bh) - 0.5);
        let yl = Math.trunc(dy);
        dy = f(dy - yl);
        if (yl < 0) { yl = 0; dy = f(dy - 1); } else if (yl >= nby - 1) { yl = nby < 2 ? 0 : nby - 2; dy = f(dy + 1); }
        const cdy = f(1 - dy), dy3 = f(f(f(dy * dy) * dy) - dy), cdy3 = f(f(f(cdy * cdy) * cdy) - cdy);
        for (let x = 0; x < nbx; x++) {
          const lo = yl * nbx + x, hi = lo + nbx;
          node[x] = f(f(f(f(cdy * values[lo]) + f(dy * values[hi])) + f(cdy3 * dvalues[lo])) + f(dy3 * dvalues[hi]));
        }
        if (nbx > 1) {
          dnode[0] = 0; u[0] = 0;
          for (let k = 0; k < nbx - 2; k++) {
            let temp = f(-1 / f(dnode[k] + 4));
            dnode[k + 1] = temp;
            temp = f(temp * f(u[k] - f(6 * f(f(node[k + 2] + node[k]) - f(2 * node[k + 1])))));
            u[k + 1] = temp;
          }
          dnode[nbx - 1] = 0;
          for (let k = nbx - 2; k >= 1; k--) dnode[k] = f(f(f(dnode[k] * dnode[k + 1]) + u[k]) / 6);
        }
      } else { nodeArr = values; dnodeArr = dvalues; }
      const row = y * width;
      if (nbx > 1) {
        const xstep = f(1 / bw), changepoint = bw >> 1, dx0 = f(f(((bw + 1) % 2) * xstep) / 2);
        let dx = f(f(xstep - 1) / 2), lo = 0;
        for (let x = 0, i = 0, j = 0; j < width; j++) {
          if (i === changepoint && x > 0 && x < nbxm1) { lo++; dx = dx0; }
          const cdx = f(1 - dx);
          out[row + j] = f(f(cdx * f(nodeArr[lo] + f(f(f(cdx * cdx) - 1) * dnodeArr[lo])))
                         + f(dx * f(nodeArr[lo + 1] + f(f(f(dx * dx) - 1) * dnodeArr[lo + 1]))));
          if (i === bw) { x++; i = 0; }
          i++; dx = f(dx + xstep);
        }
      } else out.fill(nodeArr[0], row, row + width);
    }
    return out;
  }

  // ================================================================ pixel lists (extract.h plist)
  // one pixel per index: next (-1 ends a chain), x, y, value (image), cdvalue (filtered image)
  class PList {
    constructor(cap = 16) { this.n = 0; this.alloc(cap); }
    alloc(cap) {
      const old = this;
      const next = new Int32Array(cap), x = new Int32Array(cap), y = new Int32Array(cap);
      const value = new Float32Array(cap), cd = new Float32Array(cap);
      if (old.next) { next.set(old.next); x.set(old.x); y.set(old.y); value.set(old.value); cd.set(old.cd); }
      Object.assign(this, { next, x, y, value, cd, cap });
    }
    grow(n) { if (n > this.cap) this.alloc(Math.max(n, 2 * this.cap)); }
  }
  const newObj = (o = {}) => ({ thresh: 0, mthresh: 0, fdnpix: 0, dnpix: 0, xpeak: 0, ypeak: 0, xcpeak: 0, ycpeak: 0, mx: 0, my: 0,
    xmin: 0, xmax: 0, ymin: 0, ymax: 0, mx2: 0, my2: 0, mxy: 0, a: 0, b: 0, theta: 0, abcor: 0, cxx: 0, cyy: 0, cxy: 0,
    fdflux: 0, dflux: 0, fdpeak: 0, dpeak: 0, flag: 0, firstpix: -1, lastpix: -1, ...o });
  const newList = (thresh = 0) => ({ obj: [], plist: new PList(), npix: 0, thresh });

  // addobjdeep: copy object `no` of l1 (with its pixels, in chain order) to the end of l2
  function addobjdeep(no, l1, l2) {
    const o = l1.obj[no], p1 = l1.plist, p2 = l2.plist, fp = l2.npix;
    p2.grow(fp + o.fdnpix);
    let j = fp;
    for (let i = o.firstpix; i !== -1; i = p1.next[i]) {
      if (j >= p2.cap) p2.grow(j + 1);
      p2.next[j] = j + 1; p2.x[j] = p1.x[i]; p2.y[j] = p1.y[i]; p2.value[j] = p1.value[i]; p2.cd[j] = p1.cd[i];
      j++;
    }
    p2.next[j - 1] = -1;
    l2.npix = j; p2.n = j;
    l2.obj.push({ ...o, firstpix: fp, lastpix: j - 1 });
  }

  // update (lutz.c): merge info b into info a
  function update(a, b, next) {
    a.pixnb += b.pixnb;
    a.flag |= b.flag;
    if (a.firstpix === -1) { a.firstpix = b.firstpix; a.lastpix = b.lastpix; }
    else if (b.lastpix !== -1) { next[a.lastpix] = b.firstpix; a.lastpix = b.lastpix; }
  }
  const info0 = () => ({ pixnb: 0, firstpix: -1, lastpix: -1, flag: 0 });
  const copyInfo = (i) => ({ pixnb: i.pixnb, firstpix: i.firstpix, lastpix: i.lastpix, flag: i.flag });

  // ================================================================ analysis (analyse.c)
  function preanalyse(o, pl) {
    let peak = -BIG, cpeak = -BIG, rv = 0, xmin = 2 * 1048576, ymin = xmin, xmax = 0, ymax = 0, n = 0;
    let xpeak = 0, ypeak = 0, xcpeak = 0, ycpeak = 0;
    for (let i = o.firstpix; i !== -1; i = pl.next[i]) {
      const x = pl.x[i], y = pl.y[i], val = pl.value[i], cval = pl.cd[i];
      if (peak < val) { peak = val; xpeak = x; ypeak = y; }
      if (cpeak < cval) { cpeak = cval; xcpeak = x; ycpeak = y; }
      rv += cval;
      if (xmin > x) xmin = x;
      if (xmax < x) xmax = x;
      if (ymin > y) ymin = y;
      if (ymax < y) ymax = y;
      n++;
    }
    Object.assign(o, { fdnpix: n, fdflux: f(rv), fdpeak: cpeak, dpeak: peak, xpeak, ypeak, xcpeak, ycpeak, xmin, xmax, ymin, ymax });
  }

  function analyse(o, pl, robust) {
    preanalyse(o, pl);
    let dnpix = 0, mx = 0, my = 0, tv = 0, mx2 = 0, my2 = 0, mxy = 0, area2 = 0;
    const thresh = o.thresh, peak = o.dpeak, rv = o.fdflux, thresh2 = (thresh + peak) / 2;
    const xmin = o.xmin, ymin = o.ymin;
    for (let i = o.firstpix; i !== -1; i = pl.next[i]) {
      const x = pl.x[i] - xmin, y = pl.y[i] - ymin, cval = pl.cd[i], val = pl.value[i];
      tv += val;
      if (val > thresh) dnpix++;
      if (val > thresh2) area2++;
      const cx = f(cval * x), cy = f(cval * y);
      mx += cx; my += cy; mx2 += f(cx * x); my2 += f(cy * y); mxy += f(cx * y);
    }
    let xm = mx / rv, ym = my / rv, xm2, ym2, xym;
    if (robust && (o.flag & OBJ_MERGED)) {
      const xn = o.mx - xmin, yn = o.my - ymin;
      xm2 = mx2 / rv + xn * xn - 2 * xm * xn;
      ym2 = my2 / rv + yn * yn - 2 * ym * yn;
      xym = mxy / rv + xn * yn - xm * yn - xn * ym;
      xm = xn; ym = yn;
    } else {
      xm2 = mx2 / rv - xm * xm; ym2 = my2 / rv - ym * ym; xym = mxy / rv - xm * ym;
    }
    let temp2 = xm2 * ym2 - xym * xym;
    if (temp2 < 0.00694) { xm2 += 0.0833333; ym2 += 0.0833333; temp2 = xm2 * ym2 - xym * xym; o.flag |= OBJ_SINGU; }
    let temp = xm2 - ym2;
    const theta = Math.abs(temp) > 0 ? Math.atan2(2 * xym, temp) / 2 : Math.PI / 4;
    temp = Math.sqrt(0.25 * temp * temp + xym * xym);
    const pmx2 = 0.5 * (xm2 + ym2) + temp, pmy2 = 0.5 * (xm2 + ym2) - temp;
    Object.assign(o, { dnpix, dflux: f(tv), mx: xm + xmin, my: ym + ymin, mx2: xm2, my2: ym2, mxy: xym,
      a: f(Math.sqrt(pmx2)), b: f(Math.sqrt(pmy2)), theta: f(theta),
      cxx: f(ym2 / temp2), cyy: f(xm2 / temp2), cxy: f(-2 * xym / temp2) });
    const darea = area2 - dnpix, t1t2 = thresh / thresh2;
    if (t1t2 > 0) {
      o.abcor = f((darea < 0 ? darea : -1) / (2 * Math.PI * Math.log(t1t2 < 1 ? t1t2 : 0.99) * o.a * o.b));
      if (o.abcor > 1) o.abcor = 1;
    } else o.abcor = 1;
  }

  // the minarea-th brightest pixel above threshold, for cleaning (analysemthresh, with SExtractor's heap as written)
  function analysemthresh(o, pl, minarea, thresh) {
    if (o.fdnpix < minarea) { o.mthresh = 0; return; }
    const heap = new Float32Array(minarea);
    let h = minarea, n = 0;
    for (let i = o.firstpix; i !== -1; i = pl.next[i]) {
      const tpix = f(pl.cd[i] - thresh);
      if (h > 0) heap[n++] = tpix;
      else if (h) {
        if (tpix > heap[0]) {
          heap[0] = tpix;
          for (let j = 0, k; (k = (j + 1) << 1) <= minarea; j = k) {
            let hk = k;
            if (k !== minarea && heap[k - 1] > heap[k]) { hk++; k++; }
            hk--;
            if (heap[j] <= heap[hk]) break;
            const s = heap[hk]; heap[hk] = heap[j]; heap[j] = s;
          }
        }
      } else heap.sort();
      h--;
    }
    o.mthresh = heap[0];
  }

  // clean (extract.c): drop detections that are only the wings of a brighter neighbour
  function clean(objs, beta) {
    const CLEAN_ZONE = 10, survives = objs.map(() => true);
    for (let i = 0; i < objs.length; i++) {
      if (!survives[i]) continue;
      const o1 = objs[i];
      const unitareain = Math.PI * o1.a * o1.b, ampin = o1.fdflux / (2 * unitareain * o1.abcor);
      const alphain = (Math.pow(ampin / o1.thresh, 1 / beta) - 1) * unitareain / o1.fdnpix;
      for (let j = i + 1; j < objs.length; j++) {
        if (!survives[j]) continue;
        const o2 = objs[j], dx = f(o1.mx - o2.mx), dy = f(o1.my - o2.my);
        let rlim = f(o1.a + o2.a);
        rlim = f(rlim * rlim);
        if (f(f(dx * dx) + f(dy * dy)) > rlim * CLEAN_ZONE * CLEAN_ZONE) continue;
        if (o2.fdflux < o1.fdflux) {
          const val = 1 + alphain * f(f(f(f(o1.cxx * dx) * dx) + f(f(o1.cyy * dy) * dy)) + f(f(o1.cxy * dx) * dy));
          if (val > 1 && f(val < 1e10 ? ampin * Math.pow(val, -beta) : 0) > o2.mthresh) survives[j] = false;
        } else {
          const unitarea = Math.PI * o2.a * o2.b, amp = o2.fdflux / (2 * unitarea * o2.abcor);
          const alpha = (Math.pow(amp / o2.thresh, 1 / beta) - 1) * unitarea / o2.fdnpix;
          const val = 1 + alpha * f(f(f(f(o2.cxx * dx) * dx) + f(f(o2.cyy * dy) * dy)) + f(f(o2.cxy * dx) * dy));
          if (val > 1 && f(val < 1e10 ? amp * Math.pow(val, -beta) : 0) > o1.mthresh) survives[i] = false;
        }
      }
    }
    return survives;
  }

  // ================================================================ Lutz's 8-connected segment scan (extract.c / lutz.c)
  // scans rows y0..y1 (y1: the empty closing row) and columns x0..x1 (x1: the closing column); pixel(x, y) returns a
  // new pixel index for a pixel above threshold (or -1); done(info) receives every completed object
  // (two sets of buffers, as SEP: the main scan's and deblending's (lutzalloc), which runs inside the main scan;
  // allocated once per image width, markers cleared over the scanned columns)
  const LB = {};
  function lutzBuffers(w, set) {
    if (!LB[set] || LB[set].size !== w + 1) {
      const size = w + 1;
      LB[set] = { size, marker: new Uint8Array(size), start: new Int32Array(size + 1), end: new Int32Array(size + 1),
             info: new Array(size + 1), store: new Array(size + 1), psstack: new Int8Array(size + 2) };
    }
    return LB[set];
  }
  function lutzScan(set, x0, x1, y0, y1, w, h, pixel, next, done, curFlag) {
    const { marker, start, end, info, store, psstack } = lutzBuffers(w, set);
    const COMPLETE = 0, INCOMPLETE = 1, NONOBJECT = 2, OBJECT = 3, UNKNOWN = -1;
    const S = 1, s_ = 2, f_ = 3, F = 4;
    let co = 0, pstop = 0;
    marker.fill(0, x0, x1 + 1);
    for (let yl = y0; yl <= y1; yl++) {
      let ps = COMPLETE, cs = NONOBJECT;
      const trunflag = yl === 0 || yl === h - 1 ? OBJ_TRUNC : 0;
      for (let xl = x0; xl <= x1; xl++) {
        const newmarker = marker[xl];
        marker[xl] = 0;
        const cn = (yl === y1 || xl === x1) ? -1 : pixel(xl, yl);
        const luflag = cn >= 0;
        let cur = null;
        if (luflag) {
          cur = { pixnb: 1, firstpix: cn, lastpix: cn, flag: curFlag(xl, yl, trunflag) };
          if (cs !== OBJECT) {
            cs = OBJECT;
            if (ps === OBJECT) {
              if (start[co] === UNKNOWN) { marker[xl] = S; start[co] = xl; } else marker[xl] = s_;
            } else {
              psstack[pstop++] = ps;
              marker[xl] = S;
              start[++co] = xl;
              ps = COMPLETE;
              info[co] = info0();
            }
          }
        }
        if (newmarker) {
          if (newmarker === S) {
            psstack[pstop++] = ps;
            if (cs === NONOBJECT) {
              psstack[pstop++] = COMPLETE;
              info[++co] = copyInfo(store[xl]);
              start[co] = UNKNOWN;
            } else update(info[co], store[xl], next);
            ps = OBJECT;
          } else if (newmarker === s_) {
            if (cs === OBJECT && ps === COMPLETE) {
              pstop--;
              const xl2 = start[co];
              update(info[co - 1], info[co], next);
              if (start[--co] === UNKNOWN) start[co] = xl2; else marker[xl2] = s_;
            }
            ps = OBJECT;
          } else if (newmarker === f_) {
            ps = INCOMPLETE;
          } else if (newmarker === F) {
            ps = psstack[--pstop];
            if (cs === NONOBJECT && ps === COMPLETE) {
              if (start[co] === UNKNOWN) done(info[co]);
              else { marker[end[co]] = F; store[start[co]] = copyInfo(info[co]); }
              co--;
              ps = psstack[--pstop];
            }
          }
        }
        if (luflag) update(info[co], cur, next);
        else if (cs === OBJECT) {
          cs = NONOBJECT;
          if (ps !== COMPLETE) { marker[xl] = f_; end[co] = xl; }
          else { ps = psstack[--pstop]; marker[xl] = F; store[start[co]] = copyInfo(info[co]); co--; }
        }
      }
    }
  }

  // lutz (lutz.c): the parts of `parent` above list.thresh, as new objects (with >= 3 pixels) of `list`
  function lutz(plistin, submap, subx, suby, subw, parent, list, minarea, w, h) {
    const debMaxarea = Math.min(minarea, 3), thresh = list.thresh;
    const stx = parent.xmin, sty = parent.ymin, enx = parent.xmax + 1, eny = parent.ymax + 1;
    const pl = list.plist; // scratch: its pixels are copied out (addobjdeep) before the next call
    pl.grow((eny - sty) * (enx - stx));
    list.obj = []; list.npix = 0;
    let cn = 0;
    lutzScan("deblend", stx, enx, sty, eny, w, h, (x, y) => {
      const idx = submap[(y - suby) * subw + (x - subx)];
      if (idx < 0 || !(plistin.cd[idx] > thresh)) return -1;
      pl.next[cn] = -1; pl.x[cn] = plistin.x[idx]; pl.y[cn] = plistin.y[idx];
      pl.value[cn] = plistin.value[idx]; pl.cd[cn] = plistin.cd[idx];
      return cn++;
    }, pl.next, (info) => {
      if (info.pixnb >= debMaxarea) {
        const o = newObj({ firstpix: info.firstpix, lastpix: info.lastpix, flag: info.flag });
        list.npix += info.pixnb;
        preanalyse(o, pl);
        list.obj.push(o);
      }
    }, (x, y, trunflag) => trunflag | (x === 0 || x === w - 1 ? OBJ_TRUNC : 0));
    pl.n = cn;
  }

  // macOS libc rand_r (Park-Miller minimal standard), which the SEP build used for the validation calls
  function randR(state) {
    let ctx = state.seed >>> 0;
    if (ctx === 0) ctx = 123459876;
    const hi = Math.floor(ctx / 127773), lo = ctx % 127773;
    let x = 16807 * lo - 2836 * hi;
    if (x < 0) x += 0x7fffffff;
    state.seed = x >>> 0;
    return x % 2147483648;
  }

  // gatherup (deblend.c): give the faint pixels of a deblended object to its children at random, by their profiles
  function gatherup(lin, lout, rand) {
    const objin = lin.obj, nobj = objin.length, pin = lin.plist;
    lout.thresh = lin.thresh;
    const amp = new Float32Array(nobj), p = new Float32Array(nobj), n = new Int32Array(nobj);
    for (let i = 1; i < nobj; i++) analyse(objin[i], pin, false);
    const root = objin[0], xs = root.xmin, ys = root.ymin, bmw = root.xmax - xs + 1;
    const bmp = new Uint8Array(bmw * (root.ymax - ys + 1));
    for (let i = 1; i < nobj; i++) {
      const o = objin[i];
      o.thresh = lin.thresh;
      for (let k = o.firstpix; k !== -1; k = pin.next[k]) bmp[(pin.x[k] - xs) + (pin.y[k] - ys) * bmw] = 1;
      addobjdeep(i, lin, lout);
      n[i] = lout.obj.length - 1;
      const dist = f(o.fdnpix / (2 * Math.PI * o.abcor * o.a * o.b));
      amp[i] = dist < 70 ? f(o.thresh * f(Math.exp(dist))) : f(4 * o.fdpeak);
      if (amp[i] > 4 * o.fdpeak) amp[i] = f(4 * o.fdpeak);
    }
    const pout = lout.plist;
    let k = lout.npix;
    for (let q = root.firstpix; q !== -1; q = pin.next[q]) {
      const x = pin.x[q], y = pin.y[q];
      if (bmp[(x - xs) + (y - ys) * bmw]) continue;
      pout.grow(k + 1);
      const l = k++;
      pout.next[l] = -1; pout.x[l] = x; pout.y[l] = y; pout.value[l] = pin.value[q]; pout.cd[l] = pin.cd[q];
      let distmin = f(1e31), iclst = 0;
      for (let i = 1; i < nobj; i++) {
        const o = objin[i], dx = f(x - o.mx), dy = f(y - o.my);
        const dist = f(0.5 * f(f(f(f(o.cxx * dx) * dx) + f(f(o.cyy * dy) * dy)) + f(f(o.cxy * dx) * dy)) / o.abcor);
        p[i] = f(p[i - 1] + (dist < 70 ? f(amp[i] * f(Math.exp(-dist))) : 0));
        if (dist < distmin) { distmin = dist; iclst = i; }
      }
      let i;
      if (p[nobj - 1] > 1e-31) {
        const drand = f(f(p[nobj - 1] * f(randR(rand))) / f(2147483647));
        for (i = 1; i < nobj && p[i] < drand; i++);
        if (i === nobj) i = iclst;
      } else i = iclst;
      const out = lout.obj[n[i]];
      pout.next[out.lastpix] = l;
      out.lastpix = l;
    }
    lout.npix = k; pout.n = k;
  }

  // deblend (deblend.c): multi-threshold tree of one detection; branches above the contrast become objects
  function deblend(lin, lout, nthresh, mincont, minarea, w, h, rand, nsonmax) {
    const xn = nthresh, root = lin.obj[0];
    const subx = root.xmin, suby = root.ymin, subw = root.xmax - subx + 1, subh = root.ymax - suby + 1;
    const submap = new Int32Array(subw * subh).fill(-1);
    for (let i = root.firstpix; i !== -1; i = lin.plist.next[i]) submap[(lin.plist.x[i] - subx) + (lin.plist.y[i] - suby) * subw] = i;
    const thresh0 = root.thresh;
    lout.thresh = thresh0;
    const levels = Array.from({ length: xn }, () => newList()), deb2 = newList(thresh0), deb = newList();
    deb.plist.grow(subw * subh);
    const son = Array.from({ length: xn }, () => []), ok = Array.from({ length: xn }, () => []);
    addobjdeep(0, lin, levels[0]);
    addobjdeep(0, lin, deb2);
    const value0 = levels[0].obj[0].fdflux * mincont;
    ok[0][0] = 1;
    for (let k = 1; k < xn; k++) {
      const t = root.fdpeak;
      deb.thresh = f(t > 0 ? thresh0 * Math.pow(t / thresh0, k / xn) : thresh0);
      if (levels[k - 1].obj.length >= nsonmax) throw new Error("SEP: deblending sub-object limit");
      for (let i = 0; i < levels[k - 1].obj.length; i++) {
        lutz(lin.plist, submap, subx, suby, subw, levels[k - 1].obj[i], deb, minarea, w, h);
        const sons = (son[k - 1][i] = []);
        const shell = levels[k - 1].obj[i], spl = levels[k - 1].plist;
        for (let j = 0; j < deb.obj.length; j++) {
          // belong: the core's first pixel is one of the shell's pixels
          const c = deb.obj[j], xc = deb.plist.x[c.firstpix], yc = deb.plist.y[c.firstpix];
          let inside = false;
          for (let q = shell.firstpix; q !== -1; q = spl.next[q]) if (spl.x[q] === xc && spl.y[q] === yc) { inside = true; break; }
          if (!inside) continue;
          c.thresh = deb.thresh;
          addobjdeep(j, deb, levels[k]);
          const m = levels[k].obj.length - 1;
          if (m >= nsonmax) throw new Error("SEP: deblending sub-object limit");
          sons.push(m);
          ok[k][m] = 1;
        }
      }
    }
    for (let k = xn - 2; k >= 0; k--) {
      const obj = levels[k + 1].obj;
      for (let i = 0; i < levels[k].obj.length; i++) {
        let m = 0;
        const sons = son[k][i] || [];
        const big = (j) => f(obj[j].fdflux - f(obj[j].thresh * obj[j].fdnpix)) > value0;
        for (const j of sons) { if (big(j)) m++; ok[k][i] &= ok[k + 1][j]; }
        if (m > 1) {
          for (const j of sons) {
            if (ok[k + 1][j] && big(j)) { obj[j].flag |= OBJ_MERGED; addobjdeep(j, levels[k + 1], deb2); }
          }
          ok[k][i] = 0;
        }
      }
    }
    if (ok[0][0]) addobjdeep(0, deb2, lout);
    else gatherup(deb2, lout, rand);
  }

  // ================================================================ sep.extract (sep_extract)
  // data: background-subtracted image (any numeric array, w*h); thresh: absolute threshold (SEP: thresh * err)
  // -> {objects: [{x, y, a, b, theta, peak, npix, flag, ...}], segmap: Int32Array (1-based object ids)}
  function extract(data, w, h, thresh, opt = {}) {
    const minarea = opt.minarea ?? 5, nthresh = opt.deblend_nthresh ?? 32, mincont = opt.deblend_cont ?? 0.005;
    const doClean = opt.clean ?? true, cleanParam = opt.clean_param ?? 1.0, nsonmax = opt.sub_object_limit ?? 8192;
    const kernel = opt.kernel ?? [1, 2, 1, 2, 4, 2, 1, 2, 1], kw = 3, kh = 3;
    thresh = f(thresh);
    // the data in single precision and the filtered image (convolve.c: zero outside the image, kernel / sum |kernel|)
    const img = Float32Array.from(data), conv = new Float32Array(w * h);
    let ksum = 0;
    for (const v of kernel) ksum = f(ksum + Math.abs(v));
    const kn = kernel.map((v) => f(v / ksum));
    for (let y = 0; y < h; y++) {
      let y0 = y - (kh >> 1), ch = kh, c0 = 0;
      if (y0 + ch > h) ch = h - y0;
      if (y0 < 0) { ch += y0; c0 = kw * -y0; y0 = 0; }
      const out = conv.subarray(y * w, (y + 1) * w);
      for (let i = 0; i < kw * ch; i++) {
        const cx = i % kw, cy = Math.floor(i / kw), line = (y0 + cy) * w, dcx = cx - (kw >> 1), c = kn[c0 + i];
        if (dcx >= 0) for (let x = 0; x < w - dcx; x++) out[x] = f(out[x] + f(c * img[line + x + dcx]));
        else for (let x = -dcx; x < w; x++) out[x] = f(out[x] + f(c * img[line + x + dcx]));
      }
    }
    // the main scan: detections of >= minarea pixels go through sortit (deblend, analyse) into `final`
    const pool = new PList(w * h + 1), final = newList(thresh), rand = { seed: 1 };
    let used = 0;
    const freeList = []; // freed pixel indices (SEP returns chains to its free list; the order does not matter)
    lutzScan("main", 0, w, 0, h, w, h, (x, y) => {
      if (!(conv[y * w + x] > thresh)) return -1;
      const cn = freeList.length ? freeList.pop() : used++;
      pool.next[cn] = -1; pool.x[cn] = x; pool.y[cn] = y; pool.value[cn] = img[y * w + x]; pool.cd[cn] = conv[y * w + x];
      return cn;
    }, pool.next, (info) => {
      if (info.pixnb >= minarea) {
        const list = { obj: [newObj({ firstpix: info.firstpix, lastpix: info.lastpix, flag: info.flag, thresh })], plist: pool,
          npix: info.pixnb, thresh };
        preanalyse(list.obj[0], pool);
        const out = newList();
        deblend(list, out, nthresh, mincont, minarea, w, h, rand, nsonmax);
        for (let i = 0; i < out.obj.length; i++) {
          analyse(out.obj[i], out.plist, true);
          addobjdeep(i, out, final);
        }
      }
      for (let i = info.firstpix; i !== -1; i = pool.next[i]) freeList.push(i);
    }, (x, y, trunflag) => trunflag | (x === 0 || x === w - 1 ? OBJ_TRUNC : 0));

    let survives = final.obj.map(() => true);
    if (doClean) {
      for (const o of final.obj) analysemthresh(o, final.plist, minarea, thresh);
      survives = clean(final.obj, cleanParam);
    }
    const objects = [], segmap = new Int32Array(w * h);
    final.obj.forEach((o, i) => {
      if (!survives[i]) return;
      objects.push({ thresh: o.thresh, npix: o.fdnpix, tnpix: o.dnpix, xmin: o.xmin, xmax: o.xmax, ymin: o.ymin, ymax: o.ymax,
        x: o.mx, y: o.my, x2: o.mx2, y2: o.my2, xy: o.mxy, a: o.a, b: o.b, theta: o.theta, cxx: o.cxx, cyy: o.cyy, cxy: o.cxy,
        cflux: o.fdflux, flux: o.dflux, cpeak: o.fdpeak, peak: o.dpeak, xpeak: o.xpeak, ypeak: o.ypeak,
        xcpeak: o.xcpeak, ycpeak: o.ycpeak, flag: o.flag });
      const id = objects.length;
      for (let q = o.firstpix; q !== -1; q = final.plist.next[q]) segmap[final.plist.x[q] + w * final.plist.y[q]] = id;
    });
    return { objects, segmap };
  }

  // ================================================================ apertures (aperture.c)
  function ellipseCoeffs(a, b, theta) {
    const c = Math.cos(theta), s = Math.sin(theta);
    return { cxx: c * c / (a * a) + s * s / (b * b), cyy: s * s / (a * a) + c * c / (b * b), cxy: 2 * c * s * (1 / (a * a) - 1 / (b * b)) };
  }
  function boxextent(x, y, rx, ry, w, h) {
    let flag = 0, xmin = Math.trunc(x - rx + 0.5), xmax = Math.trunc(x + rx + 1.4999999);
    let ymin = Math.trunc(y - ry + 0.5), ymax = Math.trunc(y + ry + 1.4999999);
    if (xmin < 0) { xmin = 0; flag |= APER_TRUNC; }
    if (xmax > w) { xmax = w; flag |= APER_TRUNC; }
    if (ymin < 0) { ymin = 0; flag |= APER_TRUNC; }
    if (ymax > h) { ymax = h; flag |= APER_TRUNC; }
    return { xmin, xmax, ymin, ymax, flag };
  }
  function boxextentEllipse(x, y, e, r, w, h) {
    let dx = e.cxx - e.cxy * e.cxy / (4 * e.cyy), dy = e.cyy - e.cxy * e.cxy / (4 * e.cxx);
    dx = dx > 0 ? r / Math.sqrt(dx) : 0;
    dy = dy > 0 ? r / Math.sqrt(dy) : 0;
    return boxextent(x, y, dx, dy, w, h);
  }
  const masked = (seg, i, id) => seg && (id > 0 ? seg[i] > 0 && seg[i] !== id : seg[i] !== -id);

  // sep.winpos: windowed centroid (XWIN_IMAGE / YWIN_IMAGE), subpix 11
  function winpos(data, w, h, x, y, sig, subpix = 11) {
    sig = Math.max(sig, 2.0 / 2.35 * 0.5);
    const scale = 1 / subpix, scale2 = scale * scale, offset = 0.5 * (scale - 1), inv2s2 = 1 / (2 * sig * sig);
    const r = 4 * sig, r2 = r * r, rin = r - 0.7072, rin2 = rin > 0 ? rin * rin : 0, rout2 = (r + 0.7072) ** 2;
    let flag = 0;
    for (let it = 0; it < 16; it++) {
      const box = boxextent(x, y, r, r, w, h);
      flag |= box.flag;
      let twv = 0, dxpos = 0, dypos = 0;
      for (let iy = box.ymin; iy < box.ymax; iy++) {
        for (let ix = box.xmin; ix < box.xmax; ix++) {
          const dx = ix - x, dy = iy - y, rpix2 = dx * dx + dy * dy;
          if (!(rpix2 < rout2)) continue;
          let overlap = 1;
          if (rpix2 > rin2) {
            overlap = 0;
            let sdy = dy + offset;
            for (let sy = subpix; sy--; sdy += scale) {
              let sdx = dx + offset;
              const dy2 = sdy * sdy;
              for (let sx = subpix; sx--; sdx += scale) if (sdx * sdx + dy2 < r2) overlap += scale2;
            }
          }
          const wpix = f(data[iy * w + ix]) * overlap * Math.exp(-rpix2 * inv2s2);
          twv += wpix; dxpos += wpix * dx; dypos += wpix * dy;
        }
      }
      if (!(twv > 0)) break;
      dxpos /= twv; dypos /= twv;
      x += dxpos * 2; y += dypos * 2;
      if (dxpos * dxpos + dypos * dypos < 1e-8) break;
    }
    return { x, y, flag };
  }

  // sep.kron_radius with a segmentation map (other objects' pixels masked)
  function kronRadius(data, w, h, x, y, a, b, theta, r, id, seg) {
    const e = ellipseCoeffs(a, b, theta), box = boxextentEllipse(x, y, e, r, w, h);
    let r1 = 0, v1 = 0, area = 0;
    for (let iy = box.ymin; iy < box.ymax; iy++) {
      for (let ix = box.xmin; ix < box.xmax; ix++) {
        const dx = ix - x, dy = iy - y, rpix2 = e.cxx * dx * dx + e.cyy * dy * dy + e.cxy * dx * dy;
        if (rpix2 > r * r) continue;
        const pix = f(data[iy * w + ix]);
        if (pix < -BIG || masked(seg, iy * w + ix, id)) continue;
        r1 += Math.sqrt(rpix2) * pix; v1 += pix; area++;
      }
    }
    return area === 0 || r1 <= 0 || v1 <= 0 ? 0 : r1 / v1;
  }

  // sep.sum_ellipse with subpix 1, a constant error and a segmentation map (masked pixels are left out)
  function sumEllipse(data, w, h, x, y, a, b, theta, r, err, id, seg) {
    if (!(r >= 0 && b >= 0 && a >= b && theta >= -Math.PI / 2 && theta <= Math.PI / 2)) throw new Error("SEP: invalid aperture parameters");
    const e = ellipseCoeffs(a, b, theta), box = boxextentEllipse(x, y, e, r, w, h), varpix = f(err * err);
    let tv = 0, sigtv = 0;
    for (let iy = box.ymin; iy < box.ymax; iy++) {
      for (let ix = box.xmin; ix < box.xmax; ix++) {
        const dx = ix - x, dy = iy - y;
        if (!(e.cxx * dx * dx + e.cyy * dy * dy + e.cxy * dx * dy < r * r)) continue;
        if (masked(seg, iy * w + ix, id)) continue;
        tv += f(data[iy * w + ix]); sigtv += varpix;
      }
    }
    return { sum: tv, err: Math.sqrt(sigtv) };
  }

  // ================================================================ synthstar/photometry.py
  // extract(image, fwhm): x, y (windowed), MAG_AUTO flux and error, and peak of every detection
  function photometryExtract(image, w, h, fwhm) {
    const bkg = background(image, w, h);
    const sub = new Float64Array(w * h);
    for (let i = 0; i < sub.length; i++) sub[i] = image[i] - bkg.back[i];
    const { objects, segmap } = extract(sub, w, h, f(1.5 * bkg.globalrms));
    const out = objects.map((o, i) => {
      const id = i + 1, wp = winpos(sub, w, h, o.x, o.y, fwhm / 2.3548);
      const th = Math.min(Math.max(o.theta, -Math.PI / 2), Math.PI / 2);
      const kr = kronRadius(sub, w, h, o.x, o.y, o.a, o.b, th, 6.0, id, segmap);
      const r = Math.max(2.5 * kr, 3.5);
      const s = sumEllipse(sub, w, h, o.x, o.y, o.a, o.b, th, r, bkg.globalrms, id, segmap);
      const good = wp.flag === 0;
      return { x: good ? wp.x : o.x, y: good ? wp.y : o.y, flux: s.sum, ferr: s.err, peak: o.peak, kron: kr };
    });
    return { detections: out, globalrms: bkg.globalrms, back: bkg.back, objects, segmap };
  }

  // calibration_stars: the k bright stars least contaminated by neighbours, away from the edge
  function calibrationStars(x, y, catFlux, n, k = 300, edge = 16, radius = 8) {
    let cand = [];
    for (let i = 0; i < x.length; i++) if (x[i] > edge && x[i] < n - edge && y[i] > edge && y[i] < n - edge && Number.isFinite(catFlux[i])) cand.push(i);
    cand.sort((a, b) => catFlux[b] - catFlux[a]);
    cand = cand.slice(0, 4 * k);
    const cont = cand.map((i) => {
      let s = 0;
      for (let j = 0; j < x.length; j++) {
        const d = Math.hypot(x[j] - x[i], y[j] - y[i]);
        if (d > 0 && d < 3 * radius) s += catFlux[j] * Math.exp(-0.5 * (d / (radius / 2)) ** 2);
      }
      return s / catFlux[i];
    });
    return cand.map((i, q) => [cont[q], q, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, k).map((t) => t[2]);
  }

  const median = (v) => { const a = Float64Array.from(v).sort(), n = a.length; return n % 2 ? a[n >> 1] : 0.5 * (a[n / 2 - 1] + a[n / 2]); };

  // SExtractorPhotometry: measure(image) -> the targets' flux, sigma, mag, mag_err (NaN where not detected), saturated
  class Photometry {
    constructor(x, y, catMag, catFlux, n, fullwell, targets, colour, fwhm, cal = null) {
      Object.assign(this, { x, y, catMag, colour, n, fullwell, targets, fwhm });
      const picks = cal ?? calibrationStars(x, y, catFlux, n);
      const t = new Set(targets);
      this.cal = [...new Set(picks.filter((i) => Number.isFinite(colour[i]) && !t.has(i)))].sort((a, b) => a - b);
    }
    measure(image) {
      const ex = photometryExtract(image, this.n, this.n, this.fwhm), det = ex.detections, sat = 0.98 * this.fullwell;
      const nearest = (i) => { // the nearest detection within 0.5 FWHM
        let best = -1, bd = Infinity;
        for (let j = 0; j < det.length; j++) { const d = Math.hypot(det[j].x - this.x[i], det[j].y - this.y[i]); if (d < bd) { bd = d; best = j; } }
        return bd <= 0.5 * this.fwhm ? best : -1;
      };
      // zero point with a colour term: mag + 2.5 log10(flux) = a + b (BP-RP), 3-sigma clipped
      const cy = [], cc = [];
      for (const i of this.cal) {
        const j = nearest(i);
        if (j >= 0 && det[j].flux > 0 && det[j].peak < sat) { cy.push(this.catMag[i] + 2.5 * Math.log10(det[j].flux)); cc.push(this.colour[i]); }
      }
      let use = cy.map(() => true), a = median(cy), b = 0;
      for (let it = 0; it < 10; it++) {
        const nu = use.filter(Boolean).length;
        if (nu < 5) break;
        [b, a] = linfit(cc.filter((_, q) => use[q]), cy.filter((_, q) => use[q]));
        const r = cy.map((v, q) => v - (a + b * cc[q]));
        const ru = r.filter((_, q) => use[q]), mu = median(ru), mad = median(ru.map((v) => Math.abs(v - mu)));
        const nw = r.map((v) => Math.abs(v) <= 3.0 * 1.4826 * mad);
        if (nw.every((v, q) => v === use[q])) break;
        use = nw;
      }
      const cmed = median(cc.filter((_, q) => use[q]));
      return this.targets.map((t) => {
        const zp = a + b * (Number.isFinite(this.colour[t]) ? this.colour[t] : cmed), j = nearest(t);
        const flux = j >= 0 && det[j].flux > 0 ? det[j].flux : NaN, sigma = j >= 0 ? det[j].ferr : NaN;
        return { flux, sigma, mag: zp - 2.5 * Math.log10(flux), mag_err: 2.5 / Math.LN10 * sigma / flux,
                 saturated: j >= 0 && det[j].peak >= sat, zp, ncal: use.filter(Boolean).length };
      });
    }
  }
  // least-squares line (np.polyfit deg 1): [slope, intercept]
  function linfit(x, y) {
    const n = x.length, mx = x.reduce((s, v) => s + v, 0) / n, my = y.reduce((s, v) => s + v, 0) / n;
    let sxy = 0, sxx = 0;
    for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; }
    const b = sxy / sxx;
    return [b, my - b * mx];
  }

  return { background, extract, winpos, kronRadius, sumEllipse, photometryExtract, calibrationStars, Photometry };
})();

if (typeof module !== "undefined") module.exports = SEP;
