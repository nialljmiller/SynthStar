// Minimal canvas axes in the movie's style (synthstar.argus.apply_style).

const Plot = (() => {
  const STYLE = { edge: "#86939e", label: "#253746", tick: "#435766", grid: "rgba(0,0,0,0.13)",
                  font: '"DejaVu Sans", "Verdana", sans-serif' };

  // "nice" tick values covering [lo, hi]
  function ticks(lo, hi, n = 6) {
    const a = Math.min(lo, hi), b = Math.max(lo, hi), span = b - a || 1;
    const raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9 * span; v += step) out.push(+v.toFixed(10));
    return out;
  }

  // axes in a canvas: box {x, y, w, h} in CSS pixels; xlim/ylim may be reversed; log: {y: true}
  function axes(ctx, box, xlim, ylim, opt = {}) {
    const logy = !!opt.logy;
    const ty = (v) => (logy ? Math.log10(v) : v);
    const [y0, y1] = [ty(ylim[0]), ty(ylim[1])];
    const X = (v) => box.x + ((v - xlim[0]) / (xlim[1] - xlim[0])) * box.w;
    const Y = (v) => box.y + box.h - ((ty(v) - y0) / (y1 - y0)) * box.h;
    const a = { ctx, box, xlim, ylim, X, Y };
    a.frame = () => {
      ctx.save();
      ctx.font = `11px ${STYLE.font}`;
      ctx.fillStyle = STYLE.tick;
      ctx.strokeStyle = STYLE.grid;
      ctx.lineWidth = 1;
      const xt = opt.xticks || ticks(xlim[0], xlim[1]);
      for (const v of xt) {
        const px = X(v);
        if (opt.grid !== false) { ctx.beginPath(); ctx.moveTo(px, box.y); ctx.lineTo(px, box.y + box.h); ctx.stroke(); }
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        if (opt.xlabels !== false) ctx.fillText(fmt(v), px, box.y + box.h + 5);
      }
      const yt = opt.yticks || (logy ? decades(ylim) : ticks(ylim[0], ylim[1]));
      for (const v of yt) {
        const py = Y(v);
        if (opt.grid !== false) { ctx.beginPath(); ctx.moveTo(box.x, py); ctx.lineTo(box.x + box.w, py); ctx.stroke(); }
        ctx.textAlign = opt.yright ? "left" : "right"; ctx.textBaseline = "middle";
        if (opt.ylabels !== false) ctx.fillText(logy ? `10${superscript(Math.round(Math.log10(v)))}` : fmt(v), opt.yright ? box.x + box.w + 5 : box.x - 5, py);
      }
      ctx.strokeStyle = STYLE.edge;
      ctx.beginPath();
      ctx.moveTo(box.x, box.y); ctx.lineTo(box.x, box.y + box.h); ctx.lineTo(box.x + box.w, box.y + box.h);
      ctx.stroke();
      ctx.fillStyle = STYLE.label;
      ctx.font = `12px ${STYLE.font}`;
      if (opt.xlabel) { ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(opt.xlabel, box.x + box.w / 2, box.y + box.h + 22); }
      if (opt.ylabel) {
        ctx.save(); ctx.translate(box.x - (opt.ylabelOffset || 42), box.y + box.h / 2); ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center"; ctx.textBaseline = "bottom";
        const lines = opt.ylabel.split("\n");
        lines.forEach((t, i) => ctx.fillText(t, 0, (i - lines.length + 1) * 14));
        ctx.restore();
      }
      if (opt.title) {
        ctx.font = `bold 14px ${STYLE.font}`; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
        ctx.fillStyle = "#000"; ctx.fillText(opt.title, box.x, box.y - 6);
      }
      if (opt.titleRight) {
        ctx.font = `12px ${STYLE.font}`; ctx.textAlign = "right"; ctx.textBaseline = "bottom";
        ctx.fillStyle = "#3c5262"; ctx.fillText(opt.titleRight, box.x + box.w, box.y - 6);
      }
      ctx.restore();
    };
    a.clip = (f) => { ctx.save(); ctx.beginPath(); ctx.rect(box.x, box.y, box.w, box.h); ctx.clip(); f(); ctx.restore(); };
    a.line = (xs, ys, style) => a.clip(() => {
      ctx.strokeStyle = style.color; ctx.lineWidth = style.lw || 1; ctx.globalAlpha = style.alpha ?? 1;
      ctx.setLineDash(style.dash || []);
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < xs.length; i++) {
        if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i]) || (logy && ys[i] <= 0)) { pen = false; continue; }
        const px = X(xs[i]), py = Y(ys[i]);
        if (pen) ctx.lineTo(px, py); else ctx.moveTo(px, py);
        pen = true;
      }
      ctx.stroke();
    });
    a.points = (xs, ys, style) => a.clip(() => {
      ctx.fillStyle = style.color; ctx.globalAlpha = style.alpha ?? 1;
      const r = style.r || 2;
      for (let i = 0; i < xs.length; i++) {
        if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i])) continue;
        ctx.beginPath(); ctx.arc(X(xs[i]), Y(ys[i]), r, 0, 2 * Math.PI); ctx.fill();
        if (style.edge) { ctx.strokeStyle = style.edge; ctx.lineWidth = 0.8; ctx.stroke(); }
      }
    });
    return a;
  }

  function decades(ylim) {
    const out = [];
    for (let e = Math.ceil(Math.log10(Math.min(...ylim))); e <= Math.floor(Math.log10(Math.max(...ylim))); e++) out.push(10 ** e);
    return out;
  }

  function fmt(v) {
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 1e5 || a < 0.01) return v.toPrecision(2).replace("-", "−");
    return String(+v.toFixed(3)).replace("-", "−");
  }

  const superscript = (n) => String(n).replace("-", "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);

  return { axes, ticks, STYLE };
})();
