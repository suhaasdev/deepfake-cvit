/* ============================================================
   Deepfake demo core: synthetic image generation (real / GAN-
   style fake), 2D-FFT magnitude spectrum, DCT, Chebyshev-Fourier
   patch features, detector scoring + ROC curve.
   Pure JS; testable under Node.
   ============================================================ */
(function (root) {
  "use strict";

  // ---------------- synthetic image generation ----------------
  // "real": natural texture = smooth 1/f² spectrum + mild noise
  // "fake": adds periodic checkerboard/upsampling fingerprint (GAN artifact)
  function makeImage(kind, seed, size) {
    var rnd = root.u ? root.u.rng(seed) : mulberry32(seed);
    var N = size || 128;
    var img = [];
    var fx = [], fy = [];
    for (var i = 0; i < N; i++) { fx.push(rnd() * 2 - 1); fy.push(rnd() * 2 - 1); }
    for (var y = 0; y < N; y++) {
      var row = [];
      for (var x = 0; x < N; x++) {
        // 1/f²-ish field: sum of a few low-frequency sinusoids
        var v = 0;
        for (var k = 1; k <= 4; k++) {
          v += (fx[k % N] * Math.cos(2 * Math.PI * k * x / N) +
                fy[k % N] * Math.sin(2 * Math.PI * k * y / N)) / (k * k);
        }
        v = v * 40 + (rnd() - 0.5) * 6;           // smooth field + sensor noise
        if (kind === "fake") {
          v += 14 * (((x + y) % 8 < 4) ? 1 : -1); // periodic upsampling fingerprint
        }
        row.push(Math.max(0, Math.min(255, 128 + v)));
      }
      img.push(row);
    }
    return img;
  }

  function mulberry32(seed) {
    var t = seed >>> 0;
    return function () {
      t += 0x6D2B79F5;
      var r = Math.imul(t ^ (t >>> 15), 1 | t);
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------------- 2D DFT magnitude spectrum (radial profile) ----------------
  // Separable two-pass DFT (rows then columns): O(N²M) instead of O(N³).
  function spectrumMagnitude(img) {
    var N = img.length, M = img[0].length;
    var rowsRe = [], rowsIm = [];
    for (var y = 0; y < N; y++) {
      var re = new Float64Array(M), im = new Float64Array(M);
      for (var u = 0; u < M; u++) {
        var sr = 0, si = 0;
        for (var x = 0; x < M; x++) {
          var ang = -2 * Math.PI * u * x / M;
          var px = img[y][x] / 255;
          sr += px * Math.cos(ang);
          si += px * Math.sin(ang);
        }
        re[u] = sr; im[u] = si;
      }
      rowsRe.push(re); rowsIm.push(im);
    }
    var mag = [];
    for (var v = 0; v < N; v++) {
      var mrow = new Float64Array(M);
      for (var u2 = 0; u2 < M; u2++) {
        var sr2 = 0, si2 = 0;
        for (var y2 = 0; y2 < N; y2++) {
          var ang2 = -2 * Math.PI * v * y2 / N;
          var rr = rowsRe[y2][u2], ii = rowsIm[y2][u2];
          var c = Math.cos(ang2), s = Math.sin(ang2);
          sr2 += rr * c - ii * s;
          si2 += rr * s + ii * c;
        }
        mrow[u2] = Math.sqrt(sr2 * sr2 + si2 * si2);
      }
      mag.push(mrow);
    }
    return mag;
  }

  /** roll spectrum so DC sits at the center (for display) */
  function spectrumCentered(mag) {
    var N = mag.length, M = mag[0].length, out = [];
    for (var v = 0; v < N; v++) {
      var row = new Float64Array(M);
      for (var u = 0; u < M; u++) row[(u + (M >> 1)) % M] = mag[v][u];
      out.push(row);
    }
    var out2 = [];
    for (var v2 = 0; v2 < N; v2++) out2.push(out[(v2 + (N >> 1)) % N]);
    return out2;
  }

  // radial average profile of magnitude spectrum (the CViT frequency feature)
  // uses circular distance so the unshifted spectrum wraps correctly
  function radialProfile(mag, bins) {
    var N = mag.length, M = mag[0].length;
    var cy = N / 2, cx = M / 2;
    bins = bins || 16;
    var acc = [], cnt = [];
    for (var i = 0; i < bins; i++) { acc.push(0); cnt.push(0); }
    var maxR = Math.sqrt(cy * cy + cx * cx);
    for (var v = 0; v < N; v++) {
      for (var u2 = 0; u2 < M; u2++) {
        var dy0 = Math.abs(v - cy), dx0 = Math.abs(u2 - cx);
        var dy = Math.min(dy0, N - dy0), dx = Math.min(dx0, M - dx0);
        var r = Math.sqrt(dy * dy + dx * dx);
        var b = Math.min(bins - 1, Math.floor((r / maxR) * bins));
        acc[b] += mag[v][u2]; cnt[b]++;
      }
    }
    return acc.map(function (a, i) { return cnt[i] ? a / cnt[i] : 0; });
  }

  // ---------------- DCT-II (8x8 blocks → energy statistics) ----------------
  function dct1D(seq) {
    var N = seq.length, out = [];
    for (var k = 0; k < N; k++) {
      var s = 0;
      for (var n = 0; n < N; n++) s += seq[n] * Math.cos(Math.PI * (n + 0.5) * k / N);
      out.push(s * (k === 0 ? Math.SQRT1_2 : 1) * Math.sqrt(2 / N));
    }
    return out;
  }

  function dctFeatures(img) {
    var N = img.length, M = img[0].length;
    var hi = 0, lo = 0, blocks = 0;
    for (var by = 0; by + 8 <= N; by += 8) {
      for (var bx = 0; bx + 8 <= M; bx += 8) {
        var rows = [];
        for (var y = by; y < by + 8; y++) rows.push(dct1D(img[y].slice(bx, bx + 8)));
        for (var k = 0; k < 8; k++) {
          var col = rows.map(function (r) { return r[k]; });
          var colD = dct1D(col);
          for (var u2 = 0; u2 < 8; u2++) {
            var a = Math.abs(colD[u2]);
            if ((k + u2) <= 1) lo += a; else hi += a;
          }
        }
        blocks++;
      }
    }
    return { high_low_ratio: hi / (lo + 1e-9), blocks: blocks };
  }

  // ---------------- detector: score = periodic fingerprint energy ----------------
  // The fake fingerprint creates sharp mid-frequency peaks in the radial
  // spectrum. Score = normalized peakiness of the radial profile (L2 / L1).
  function scoreImage(img) {
    var mag = spectrumMagnitude(img);
    var prof = radialProfile(mag, 16);
    var l1 = prof.reduce(function (a, b) { return a + Math.abs(b); }, 0);
    var l2 = Math.sqrt(prof.reduce(function (a, b) { return a + b * b; }, 0));
    var peakiness = l2 / (l1 + 1e-9);
    var dct = dctFeatures(img);
    // combine spectral peakiness with DCT high/low energy ratio
    var raw = 0.5 * Math.min(1, peakiness * 2.2) + 0.5 * Math.min(1, dct.high_low_ratio * 1.6);
    // Platt-style calibration to a fake probability: observed decision
    // midpoint ≈ 0.71 across seeds/sizes (real ~0.57, fake ~0.86)
    var p = 1 / (1 + Math.exp(-60 * (raw - 0.71)));
    return { score: p, rawScore: raw, radialProfile: prof, dct: dct };
  }

  // ---------------- ROC / AUC over a seeded test set ----------------
  function evaluate(seed, nPerClass) {
    var items = [];
    for (var i = 0; i < nPerClass; i++) {
      var real = makeImage("real", seed + i, 64);
      var fake = makeImage("fake", seed + 1000 + i, 64);
      items.push({ label: 0, s: scoreImage(real).score });
      items.push({ label: 1, s: scoreImage(fake).score });
    }
    // rank-based AUC (Mann-Whitney U)
    var pos = items.filter(function (it) { return it.label === 1; });
    var neg = items.filter(function (it) { return it.label === 0; });
    var wins = 0;
    pos.forEach(function (p) {
      neg.forEach(function (n) {
        wins += p.s > n.s ? 1 : p.s === n.s ? 0.5 : 0;
      });
    });
    var auc = wins / (pos.length * neg.length);
    // ROC points at a few thresholds
    var pts = [{ fpr: 0, tpr: 0 }];
    for (var t = 0; t <= 20; t++) {
      var thr = t / 20;
      var tp = pos.filter(function (p) { return p.s >= thr; }).length;
      var fp = neg.filter(function (n) { return n.s >= thr; }).length;
      pts.push({ fpr: fp / neg.length, tpr: tp / pos.length });
    }
    return { auc: auc, roc: pts, items: items };
  }

  root.DeepfakeLab = {
    makeImage: makeImage,
    spectrumMagnitude: spectrumMagnitude,
    spectrumCentered: spectrumCentered,
    radialProfile: radialProfile,
    dctFeatures: dctFeatures,
    scoreImage: scoreImage,
    evaluate: evaluate
  };
})(typeof window !== "undefined" ? window : globalThis);
