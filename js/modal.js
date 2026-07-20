var _pkgModalResultIdx = null;
var _pkgModalManualPartIdx = null;
var _pkgModalOptions = null;
var _pkgSelectedMethod = null;
var _pkgModalMaterial = 'carton';
var _pkgPreviewAlpha = Math.PI / 5;
var _pkgPreviewBeta = Math.PI / 4;
var _pkgPreviewDrag = { active: false, startX: 0, startY: 0, startAlpha: 0, startBeta: 0 };

function openPkgMethodModal(resultIdx) {
  _pkgModalResultIdx = resultIdx;
  _pkgModalManualPartIdx = null;
  var result = state._batchResults[resultIdx];
  if (!result) return;
  var predictedClass = result.predictedClass || '';
  var material = _classToMaterial(predictedClass);
  _pkgModalMaterial = material;
  var matSelect = document.getElementById('pkgModalMaterial');
  if (matSelect) matSelect.value = material;
  _openModalForParts(result.rawFeatures || {}, result.totalParts || 1, material, false);
}

function openManualPartPkgModal(partIdx) {
  _pkgModalResultIdx = null;
  _pkgModalManualPartIdx = partIdx;
  var item = state.manualPartsList[partIdx];
  if (!item) return;
  var pack = item.pack;
  var predClass = item.prediction ? item.prediction.predicted_class : '';
  var material = _classToMaterial(predClass);
  _pkgModalMaterial = material;
  var matSelect = document.getElementById('pkgModalMaterial');
  if (matSelect) matSelect.value = material;
  var raw = { '长': pack.l, '宽': pack.w, '高': pack.h, '零件尺寸L': pack.l, '零件尺寸W': pack.w, '零件尺寸H': pack.h };
  _openModalForParts(raw, item.totalParts || 1, material, true);
}

function _classToMaterial(cls) {
  if (cls.includes('木箱')) return 'wood';
  if (cls.includes('铁') || cls.includes('铁架') || cls.includes('轻钢')) return 'iron';
  if (cls.includes('天地盖')) return 'tdg';
  if (cls === 'STD') return 'STD';
  if (cls.includes('纸箱')) return 'carton';
  return 'carton';
}

function _openModalForParts(raw, totalParts, material, isManual) {
  var partL = parseFloat(raw['长'] || raw['零件尺寸L'] || 100);
  var partW = parseFloat(raw['宽'] || raw['零件尺寸W'] || 100);
  var partH = parseFloat(raw['高'] || raw['零件尺寸H'] || 100);
  var ctnSpec = CONTAINER_SPECS[state.selectedContainer] || CONTAINER_SPECS['20gp'];
  var matInfo = MATERIAL_PRICES[material];
  var sizes = MODAL_BOX_SIZES[material] || [];
  var options = sizes.map(function(s) {
    var orientations = [{d:[partL,partW,partH]},{d:[partW,partL,partH]},{d:[partH,partW,partL]},{d:[partH,partL,partW]},{d:[partL,partH,partW]},{d:[partW,partH,partL]}];
    var viable = orientations.filter(function(o) { return o.d[0] <= s.l && o.d[1] <= s.w && o.d[2] <= s.h; });
    if (!viable.length) return null;
    var best = viable[0], bestSnp = 0;
    viable.forEach(function(o) {
      var snp = Math.max(1, Math.floor(s.l / o.d[0])) * Math.max(1, Math.floor(s.w / o.d[1])) * Math.max(1, Math.floor(s.h / o.d[2]));
      if (snp > bestSnp) { bestSnp = snp; best = o; }
    });
    var boxesNeeded = Math.ceil(totalParts / bestSnp);
    var partVol = partL * partW * partH;
    var boxVol = s.l * s.w * s.h;
    var fillRate = (bestSnp * partVol) / boxVol;
    var orderFillRate = (totalParts * partVol) / (boxesNeeded * boxVol);
    return { l: s.l, w: s.w, h: s.h, snp: bestSnp, orientation: '平放', boxesNeeded: boxesNeeded, fillRate: fillRate, orderFillRate: orderFillRate, name: matInfo.name + ' ' + s.l + '×' + s.w + '×' + s.h, material: material };
  }).filter(function(o) { return o !== null; });
  options.sort(function(a, b) { if (a.boxesNeeded !== b.boxesNeeded) return a.boxesNeeded - b.boxesNeeded; return b.orderFillRate - a.orderFillRate; });
  _pkgModalOptions = options;
  if (isManual && state.manualPartsList[_pkgModalManualPartIdx] && state.manualPartsList[_pkgModalManualPartIdx].pkgMethod) {
    var pm = state.manualPartsList[_pkgModalManualPartIdx].pkgMethod;
    _pkgSelectedMethod = { l: pm.pkgL, w: pm.pkgW, h: pm.pkgH, snp: pm.snp, boxesNeeded: pm.boxesPerCtn, fillRate: pm.fillRate, orderFillRate: pm.fillRate, name: pm.name, material: pm.material };
  } else if (state._batchResults && state._batchResults[_pkgModalResultIdx] && state._batchResults[_pkgModalResultIdx].pkgMethod) {
    _pkgSelectedMethod = state._batchResults[_pkgModalResultIdx].pkgMethod;
  } else {
    _pkgSelectedMethod = options.length > 0 ? options[0] : null;
  }
  document.getElementById('pkgModalTitle').textContent = '选择包装方式 - ' + matInfo.name;
  document.getElementById('pkgModalHeader').innerHTML = '零件总量: ' + totalParts + ' 件 | 集装箱: ' + ctnSpec.label;
  renderPkgModalGrid(options);
  document.getElementById('pkgModalOverlay').classList.remove('hidden');
  setTimeout(redrawPkgPreviewInModal, 50);
}

function renderPkgModalGrid(options) {
  var grid = document.getElementById('pkgModalGrid');
  if (!options || options.length === 0) { grid.innerHTML = '<div style="padding:20px;text-align:center;color:var(--gray-2);">暂无数值可显示</div>'; return; }
  grid.innerHTML = options.map(function(o, idx) {
    var sel = _pkgSelectedMethod && _pkgSelectedMethod.l === o.l && _pkgSelectedMethod.w === o.w && _pkgSelectedMethod.h === o.h;
    return '<div class="pkg-modal-card' + (sel ? ' selected' : '') + '" onclick="selectPkgModalCard(this, ' + idx + ')">' +
      '<span class="check-mark">✓</span>' +
      '<div class="card-size">' + o.l + '×' + o.w + '×' + o.h + ' mm</div>' +
      '<div class="card-mat">需 <b>' + o.boxesNeeded + '</b> 箱 | 每箱 <b>' + o.snp + '</b> 件 | 订单空间利用率 ' + (o.orderFillRate * 100).toFixed(1) + '%</div>' +
      '</div>';
  }).join('');
}

function selectPkgModalCard(el, idx) {
  document.querySelectorAll('.pkg-modal-card').forEach(function(c) { c.classList.remove('selected'); });
  el.classList.add('selected');
  if (_pkgModalOptions && _pkgModalOptions[idx]) {
    _pkgSelectedMethod = _pkgModalOptions[idx];
    redrawPkgPreviewInModal();
  }
}

function confirmPkgSelection() {
  if (!_pkgSelectedMethod) { closePkgModal(); return; }
  if (_pkgModalManualPartIdx !== null) {
    var item = state.manualPartsList[_pkgModalManualPartIdx];
    if (item) {
      var matchedMethod = item.pkgMethods.find(function(m) { return m.pkgL === _pkgSelectedMethod.l && m.pkgW === _pkgSelectedMethod.w && m.pkgH === _pkgSelectedMethod.h; });
      state.manualPartsList[_pkgModalManualPartIdx].pkgMethod = matchedMethod || { pkgL: _pkgSelectedMethod.l, pkgW: _pkgSelectedMethod.w, pkgH: _pkgSelectedMethod.h, snp: _pkgSelectedMethod.snp, boxesPerCtn: _pkgSelectedMethod.boxesNeeded, fillRate: _pkgSelectedMethod.fillRate, name: _pkgSelectedMethod.name, material: _pkgSelectedMethod.material };
      state.manualPartsList[_pkgModalManualPartIdx].selectedPkgMethod = matchedMethod ? matchedMethod.id : null;
      persistState();
      renderManualPartsList();
    }
  } else if (_pkgModalResultIdx !== null) {
    state._batchResults[_pkgModalResultIdx].pkgMethod = _pkgSelectedMethod;
    persistState();
    renderBatchTable(state._batchResults, state._batchCsvData.header);
  }
  closePkgModal();
}

function closePkgModal() {
  document.getElementById('pkgModalOverlay').classList.add('hidden');
  _pkgModalResultIdx = null;
  _pkgModalManualPartIdx = null;
  _pkgModalOptions = null;
  _pkgSelectedMethod = null;
  _pkgModalMaterial = 'carton';
}

function changeModalMaterial(newMaterial) {
  _pkgModalMaterial = newMaterial;
  var raw = {};
  var totalParts = 1;
  if (_pkgModalResultIdx !== null && state._batchResults[_pkgModalResultIdx]) {
    raw = state._batchResults[_pkgModalResultIdx].rawFeatures || {};
    totalParts = state._batchResults[_pkgModalResultIdx].totalParts || 1;
  } else if (_pkgModalManualPartIdx !== null && state.manualPartsList[_pkgModalManualPartIdx]) {
    var pack = state.manualPartsList[_pkgModalManualPartIdx].pack;
    raw = { '长': pack.l, '宽': pack.w, '高': pack.h, '零件尺寸L': pack.l, '零件尺寸W': pack.w, '零件尺寸H': pack.h };
    totalParts = state.manualPartsList[_pkgModalManualPartIdx].totalParts || 1;
  }
  _openModalForParts(raw, totalParts, newMaterial, _pkgModalManualPartIdx !== null);
}

function redrawPkgPreviewInModal() {
  if (!_pkgSelectedMethod) return;
  var raw = {};
  if (_pkgModalResultIdx !== null && state._batchResults[_pkgModalResultIdx]) raw = state._batchResults[_pkgModalResultIdx].rawFeatures || {};
  else if (_pkgModalManualPartIdx !== null && state.manualPartsList[_pkgModalManualPartIdx]) { var p = state.manualPartsList[_pkgModalManualPartIdx].pack; raw = { '长': p.l, '宽': p.w, '高': p.h, '零件尺寸L': p.l, '零件尺寸W': p.w, '零件尺寸H': p.h }; }
  var partL = parseFloat(raw['长'] || raw['零件尺寸L'] || 100);
  var partW = parseFloat(raw['宽'] || raw['零件尺寸W'] || 100);
  var partH = parseFloat(raw['高'] || raw['零件尺寸H'] || 100);
  var snp = _pkgSelectedMethod.snp;
  var gX = 1, gY = 1, gZ = 1;
  if (snp > 1) {
    var cubeRoot = Math.round(Math.pow(snp, 1/3));
    for (var dx = Math.max(1, cubeRoot - 2); dx <= cubeRoot + 2; dx++) {
      for (var dy = Math.max(1, cubeRoot - 2); dy <= cubeRoot + 2; dy++) {
        gZ = Math.ceil(snp / (dx * dy));
        if (dx * dy * gZ >= snp) { gX = dx; gY = dy; break; }
      }
      if (gX * gY * gZ >= snp) break;
    }
  }
  drawPkgPreviewForModal({ pkgL: _pkgSelectedMethod.l, pkgW: _pkgSelectedMethod.w, pkgH: _pkgSelectedMethod.h, snp: snp, nx: gX, ny: gY, nz: gZ }, { l: partL, w: partW, h: partH, buffer: 0 });
}

function drawPkgPreviewForModal(method, part) {
  var canvas = document.getElementById('pkgPreviewCanvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  var oL = method.pkgL, oW = method.pkgW, oH = method.pkgH;
  var off = part.buffer || 0;
  var pL = part.l + 2 * off, pW = part.w + 2 * off, pH = part.h + 2 * off;
  var scaleDownX = pL > oL ? (oL - 1) / pL : 1;
  var scaleDownY = pH > oH ? (oH - 1) / pH : 1;
  var scaleDownZ = pW > oW ? (oW - 1) / pW : 1;
  var scaleDown = Math.min(scaleDownX, scaleDownY, scaleDownZ, 1);
  var iL = pL * scaleDown, iW = pW * scaleDown, iH = pH * scaleDown;
  var alpha = _pkgPreviewAlpha, beta = _pkgPreviewBeta;
  var approxDiag = Math.sqrt(oL * oL + oW * oW + oH * oH);
  var scale = Math.min(W, H) * 0.78 / approxDiag;
  var OX = W / 2, OY = H / 2;

  function pt(x, y, z) {
    var xs = x - oL / 2, ys = y - oH / 2, zs = z - oW / 2;
    var x1 = xs * Math.cos(alpha) - zs * Math.sin(alpha);
    var z1 = xs * Math.sin(alpha) + zs * Math.cos(alpha);
    var y2 = ys * Math.cos(beta) - z1 * Math.sin(beta);
    var z2 = ys * Math.sin(beta) + z1 * Math.cos(beta);
    return { px: OX + x1 * scale, py: OY - y2 * scale, depth: z2 };
  }

  function drawPoly(pts, fillColor, strokeColor, lw) {
    if (!pts || pts.length < 3) return;
    ctx.beginPath();
    ctx.moveTo(pts[0].px, pts[0].py);
    for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i].px, pts[i].py);
    ctx.closePath();
    if (fillColor) { ctx.fillStyle = fillColor; ctx.fill(); }
    if (strokeColor) { ctx.strokeStyle = strokeColor; ctx.lineWidth = lw || 0.8; ctx.stroke(); }
  }

  function rotNormal(n) {
    var x1 = n[0] * Math.cos(alpha) - n[2] * Math.sin(alpha);
    var z1 = n[0] * Math.sin(alpha) + n[2] * Math.cos(alpha);
    var y2 = n[1] * Math.cos(beta) - z1 * Math.sin(beta);
    var z2 = n[1] * Math.sin(beta) + z1 * Math.cos(beta);
    return [x1, y2, z2];
  }

  var pkgTop = 'rgba(210,180,140,0.35)', pkgRight = 'rgba(180,140,100,0.55)';
  var pkgFront = 'rgba(160,120,80,0.60)', pkgLeft = 'rgba(140,100,70,0.45)', pkgBottom = 'rgba(120,85,60,0.30)', pkgEdge = '#8B6914';
  var partTop = 'rgba(52,152,219,0.80)', partRight = 'rgba(41,128,185,0.92)';
  var partFront = 'rgba(30,100,160,0.88)', partLeft = 'rgba(25,80,130,0.75)', partBot = 'rgba(20,60,100,0.60)', partEdge = '#1a5276';
  var pkgFaceColors = { 0: pkgFront, 1: pkgFront, 2: pkgRight, 3: pkgLeft, 4: pkgTop, 5: pkgBottom };

  function getBoxFaces(L, Wg, H, offset) {
    var ox = offset, oy = offset, oz = offset;
    var normals = [
      { n: [0, 0, 1], fi: 0, verts: [[ox, oy, oz + Wg], [ox + L, oy, oz + Wg], [ox + L, oy + H, oz + Wg], [ox, oy + H, oz + Wg]] },
      { n: [0, 0, -1], fi: 1, verts: [[ox + L, oy, oz], [ox, oy, oz], [ox, oy + H, oz], [ox + L, oy + H, oz]] },
      { n: [1, 0, 0], fi: 2, verts: [[ox + L, oy, oz], [ox + L, oy, oz + Wg], [ox + L, oy + H, oz + Wg], [ox + L, oy + H, oz]] },
      { n: [-1, 0, 0], fi: 3, verts: [[ox, oy, oz + Wg], [ox, oy, oz], [ox, oy + H, oz], [ox, oy + H, oz + Wg]] },
      { n: [0, 1, 0], fi: 4, verts: [[ox, oy + H, oz], [ox + L, oy + H, oz], [ox + L, oy + H, oz + Wg], [ox, oy + H, oz + Wg]] },
      { n: [0, -1, 0], fi: 5, verts: [[ox, oy, oz + Wg], [ox + L, oy, oz + Wg], [ox + L, oy, oz], [ox, oy, oz]] },
    ];
    var visFaces = [];
    normals.forEach(function(f) {
      var rn = rotNormal(f.n);
      if (rn[2] < 0) {
        var proj = f.verts.map(function(v) { return pt.apply(null, v); });
        var depth = proj.reduce(function(s, p) { return s + p.depth; }, 0) / proj.length;
        visFaces.push({ fi: f.fi, pts: proj, depth: depth });
      }
    });
    visFaces.sort(function(a, b) { return a.depth - b.depth; });
    return visFaces;
  }

  var visPkg = getBoxFaces(oL, oW, oH, 0);
  visPkg.forEach(function(f) { drawPoly(f.pts, pkgFaceColors[f.fi], pkgEdge, 1.2); });

  var snp = method.snp;
  if (snp === 1) {
    var partFaceColors = { 0: partFront, 1: partFront, 2: partRight, 3: partLeft, 4: partTop, 5: partBot };
    var visPart = getBoxFaces(iL, iW, iH, off);
    visPart.forEach(function(f) { drawPoly(f.pts, partFaceColors[f.fi], partEdge, 0.9); });
  } else {
    var mNX = method.nx || 1, mNY = method.ny || 1, mNZ = method.nz || 1;
    var gX = mNX, gY = mNY, gZ = mNZ;
    var partGap = 3;
    var partVisL = (oL - 2 * off) / gX - partGap;
    var partVisW = (oW - 2 * off) / gZ - partGap;
    var partVisH = (oH - 2 * off) / gY - partGap;
    var partPalettes2 = [
      { fill: '#3498db', dark: '#1f6aa5', light: '#85c1e9' },
      { fill: '#27ae60', dark: '#1a7a42', light: '#82e0aa' },
      { fill: '#e67e22', dark: '#a0510e', light: '#f5b041' },
      { fill: '#9b59b6', dark: '#6c3483', light: '#c39bd3' },
      { fill: '#e74c3c', dark: '#a93226', light: '#f1948a' },
      { fill: '#1abc9c', dark: '#0e7868', light: '#76d7c4' },
    ];

    var visParts = [];
    for (var pz = 0; pz < gZ; pz++) {
      for (var py = 0; py < gY; py++) {
        for (var px = 0; px < gX; px++) {
          var partIdx = pz * gX * gY + py * gX + px;
          if (partIdx >= snp) break;
          var pal = partPalettes2[partIdx % partPalettes2.length];
          var ppx = off + px * (partVisL + partGap);
          var ppy = off + py * (partVisH + partGap);
          var ppz = off + pz * (partVisW + partGap);
          var pFaceColors2 = { 0: pal.fill + 'dd', 1: pal.fill + '60', 2: pal.fill + '99', 3: pal.fill + '70', 4: pal.light + 'dd', 5: pal.dark + '60' };
          var normals2 = [
            { n: [0, 0, 1], fi: 0, verts: [[ppx, ppy, ppz + partVisW], [ppx + partVisL, ppy, ppz + partVisW], [ppx + partVisL, ppy + partVisH, ppz + partVisW], [ppx, ppy + partVisH, ppz + partVisW]] },
            { n: [0, 0, -1], fi: 1, verts: [[ppx + partVisL, ppy, ppz], [ppx, ppy, ppz], [ppx, ppy + partVisH, ppz], [ppx + partVisL, ppy + partVisH, ppz]] },
            { n: [1, 0, 0], fi: 2, verts: [[ppx + partVisL, ppy, ppz], [ppx + partVisL, ppy, ppz + partVisW], [ppx + partVisL, ppy + partVisH, ppz + partVisW], [ppx + partVisL, ppy + partVisH, ppz]] },
            { n: [-1, 0, 0], fi: 3, verts: [[ppx, ppy, ppz + partVisW], [ppx, ppy, ppz], [ppx, ppy + partVisH, ppz], [ppx, ppy + partVisH, ppz + partVisW]] },
            { n: [0, 1, 0], fi: 4, verts: [[ppx, ppy + partVisH, ppz], [ppx + partVisL, ppy + partVisH, ppz], [ppx + partVisL, ppy + partVisH, ppz + partVisW], [ppx, ppy + partVisH, ppz + partVisW]] },
            { n: [0, -1, 0], fi: 5, verts: [[ppx, ppy, ppz + partVisW], [ppx + partVisL, ppy, ppz + partVisW], [ppx + partVisL, ppy, ppz], [ppx, ppy, ppz]] },
          ];
          normals2.forEach(function(f) {
            var rn = rotNormal(f.n);
            if (rn[2] < 0) {
              var proj = f.verts.map(function(v) { return pt.apply(null, v); });
              var depth = proj.reduce(function(s, p) { return s + p.depth; }, 0) / proj.length;
              visParts.push({ pts: proj, fill: pFaceColors2[f.fi], depth: depth });
            }
          });
        }
      }
    }
    visParts.sort(function(a, b) { return a.depth - b.depth; });
    visParts.forEach(function(f) { drawPoly(f.pts, f.fill, null, 0.3); });
  }
}

function initPkgPreviewDrag() {
  var canvas = document.getElementById('pkgPreviewCanvas');
  if (!canvas) return;
  canvas.addEventListener('mousedown', function(e) {
    _pkgPreviewDrag.active = true;
    _pkgPreviewDrag.startX = e.clientX;
    _pkgPreviewDrag.startY = e.clientY;
    _pkgPreviewDrag.startAlpha = _pkgPreviewAlpha;
    _pkgPreviewDrag.startBeta = _pkgPreviewBeta;
  });
  document.addEventListener('mousemove', function(e) {
    if (!_pkgPreviewDrag.active) return;
    var dx = e.clientX - _pkgPreviewDrag.startX;
    var dy = e.clientY - _pkgPreviewDrag.startY;
    _pkgPreviewAlpha = _pkgPreviewDrag.startAlpha + dx * 0.012;
    _pkgPreviewBeta = Math.max(-Math.PI / 3, Math.min(Math.PI / 2.2, _pkgPreviewDrag.startBeta + dy * 0.010));
    redrawPkgPreviewInModal();
  });
  document.addEventListener('mouseup', function() { _pkgPreviewDrag.active = false; });
  canvas.addEventListener('touchstart', function(e) {
    var t = e.touches[0];
    _pkgPreviewDrag.active = true;
    _pkgPreviewDrag.startX = t.clientX;
    _pkgPreviewDrag.startY = t.clientY;
    _pkgPreviewDrag.startAlpha = _pkgPreviewAlpha;
    _pkgPreviewDrag.startBeta = _pkgPreviewBeta;
  }, { passive: true });
  document.addEventListener('touchmove', function(e) {
    if (!_pkgPreviewDrag.active) return;
    var t = e.touches[0];
    var dx = t.clientX - _pkgPreviewDrag.startX;
    var dy = t.clientY - _pkgPreviewDrag.startY;
    _pkgPreviewAlpha = _pkgPreviewDrag.startAlpha + dx * 0.012;
    _pkgPreviewBeta = Math.max(-Math.PI / 3, Math.min(Math.PI / 2.2, _pkgPreviewDrag.startBeta + dy * 0.010));
    redrawPkgPreviewInModal();
  }, { passive: true });
  document.addEventListener('touchend', function() { _pkgPreviewDrag.active = false; });
}

document.addEventListener('DOMContentLoaded', function() { setTimeout(initPkgPreviewDrag, 100); });
