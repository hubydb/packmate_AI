function initStep2() {
  if (!state.model) { alert('模型尚未加载，请先训练模型'); location.href = '/train_llm'; return; }

  var hasManual = state.manualPartsList && state.manualPartsList.length > 0;
  var hasBatch = state._batchResults && state._batchResults.length > 0;
  var hasSingle = state.part && state.predicted;

  // 优先判断显式模式：批量 > 单零件预测
  if (hasManual || hasBatch) {
    document.getElementById('pkgOptions').style.display = 'none';
    document.getElementById('palletStackingSection').style.display = 'none';
    var count = hasManual ? state.manualPartsList.length : state._batchResults.length;
    document.getElementById('partInfoSummary').innerHTML = '<b>批量预测模式</b> | 共 ' + count + ' 条';
    document.getElementById('step2BatchSummary').style.display = 'block';
    renderStep2BatchSummary();
  } else if (hasSingle) {
    document.getElementById('partInfoSummary').innerHTML = `<b>${escapeHtml(state.part.partNo)}</b>${state.part.partName ? ' ' + escapeHtml(state.part.partName) : ''} | ${state.part.l}×${state.part.w}×${state.part.h} mm | ${state.part.weight || 0} kg | ${state.part.qty} 件 | 分类 ${state.part.category} | 集装箱 ${state.part.ctnSpec.label} | 模型推荐 <b>${escapeHtml(state.predicted.predicted_class)}</b>`;
    document.getElementById('step2BatchSummary').style.display = 'none';
    if (!state.pkgMethods || state.pkgMethods.length === 0) {
      state.pkgMethods = computePkgMethods(state.part);
      sortPkgMethodsByConfidence(state.predicted);
      persistState();
    }
    document.getElementById('pkgOptions').style.display = 'grid';
    console.log('step2 single mode, pkgMethods count:', state.pkgMethods ? state.pkgMethods.length : 0);
    renderPkgOptions();
  } else {
    document.getElementById('partInfoSummary').innerHTML = '请先返回 step1 完成零件输入或批量预测。';
    document.getElementById('pkgOptions').style.display = 'none';
    document.getElementById('step2BatchSummary').style.display = 'none';
  }
  initPkgPreviewDrag();
}

function selectPkgMethod(el, id) {
  document.querySelectorAll('.pkg-option').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  state.selectedPkgMethod = id;
  persistState();
  updatePalletStackingSection();
}

function goStep3() {
  const method = state.pkgMethods.find(m => m.id === state.selectedPkgMethod);
  if (!method) { alert('请先选择一种包装方式'); return; }
  location.href = '/step3.html';
}

function renderPkgOptions() {
  const container = document.getElementById('pkgOptions');
  console.log('renderPkgOptions called, container:', !!container, 'pkgMethods:', state.pkgMethods ? state.pkgMethods.length : 0);
  if (!container) return;
  if (!state.pkgMethods || state.pkgMethods.length === 0) {
    container.innerHTML = '<div class="alert alert-warning">无法推荐包装方式，请返回修改零件参数。</div>';
    return;
  }
  container.innerHTML = state.pkgMethods.map((m, i) => `
    <div class="pkg-option ${m.id === state.selectedPkgMethod ? 'selected' : ''}" data-id="${m.id}" data-index="${i}" data-name="${escapeHtml(m.name)}" data-l="${m.pkgL}" data-w="${m.pkgW}" data-h="${m.pkgH}" data-snp="${m.snp}" data-boxes="${m.boxesPerCtn}" data-total="${m.totalParts}" data-fill="${(m.fillRate * 100).toFixed(1)}" data-ctnfill="${(m.containerFillRate * 100).toFixed(1)}" data-weight="${m.perBoxWeight.toFixed(1)}" data-match="${m.matchScore > 0 ? '1' : '0'}" onclick="selectPkgMethod(this, '${m.id}')">
      <span class="check-mark">✓</span>
      <div class="opt-title">${m.matchScore > 0 ? '⭐ ' : ''}${escapeHtml(m.name)}</div>
      <div class="opt-desc">
        ${escapeHtml(m.desc)}<br/>
        包箱尺寸：${m.pkgL}×${m.pkgW}×${m.pkgH} mm | 单箱重量：<b>${m.perBoxWeight.toFixed(1)} kg</b><br/>
        包箱空间利用率：<b style="color:var(--primary)">${(m.fillRate * 100).toFixed(1)}%</b> | 集装箱空间利用率：${(m.containerFillRate * 100).toFixed(1)}%<br/>
        每集装箱 <b>${m.totalParts}</b> 件（${m.boxesPerCtn} 箱 × ${m.snp} 件/箱）
        ${m.matchScore > 0 ? `<br/><span style="color:var(--success)">模型推荐匹配</span>` : ''}
      </div>
    </div>
  `).join('');
  updatePalletStackingSection();
}

function updatePalletStackingSection() {
  var section = document.getElementById('palletStackingSection');
  var list = document.getElementById('palletStackingList');
  if (!section || !list) return;
  var method = state.pkgMethods.find(function(m) { return m.id === state.selectedPkgMethod; });
  if (!method || !method.name || method.name.indexOf('纸箱') === -1) { section.style.display = 'none'; return; }
  var result = computePalletStacking(method.pkgL, method.pkgW);
  if (!result.fits) {
    section.style.display = 'block';
    list.innerHTML = '<div style="padding:12px;color:var(--gray-2);font-size:13px;">该纸箱尺寸无法使用 148A 托盘码放</div>';
    return;
  }
  var s = result.spec;
  list.innerHTML = '<div style="background:#f7f8fa;border:1px solid #e8ecf0;border-radius:6px;padding:10px 12px;font-size:12px;line-height:1.8;">' +
    '<b style="font-size:13px;">' + s.lDiv + '×' + s.wDiv + ' 等分</b>（格 ' + s.cellL + '×' + s.cellW + ' mm）<br/>' +
    '每层 <b>' + s.alongL + '×' + s.alongW + ' = ' + s.total + '</b> 箱/托盘' +
  '</div>';
  section.style.display = 'block';
}

function computePalletStacking(cL, cW) {
  var PACK_PALLET_DIVS = [[2,2,700,530],[2,4,700,265],[4,2,350,530],[4,4,350,265],[8,2,175,530],[8,4,175,265]];
  var best = null;
  for (var i = 0; i < PACK_PALLET_DIVS.length; i++) {
    var spec = PACK_PALLET_DIVS[i];
    var cellL = spec[2], cellW = spec[3];
    var fit1 = (cL <= cellL && cW <= cellW);
    var fit2 = (cW <= cellL && cL <= cellW);
    if (fit1 || fit2) {
      var alongL = Math.floor(cellL / cL);
      var alongW = Math.floor(cellW / cW);
      var total = alongL * alongW;
      if (!best || total > best.total) {
        best = { index: i, lDiv: spec[0], wDiv: spec[1], cellL: cellL, cellW: cellW, alongL: alongL, alongW: alongW, total: total };
      }
    }
  }
  if (!best) return { fits: false, spec: null, boxesPerPallet: 0 };
  return { fits: true, spec: best, boxesPerPallet: best.total };
}

function canFitOnPallet(material, pkgL, pkgW, pkgH) {
  if (material !== 'STD' && material !== 'carton' && material !== 'tdg') return false;
  return computePalletStacking(pkgL, pkgW).fits;
}

function computeBatchPalletAllocation(results) {
  var pallets = [];
  results.forEach(function(r, idx) {
    if (r.error || !r.pkgMethod) return;
    var pkg = r.pkgMethod;
    var boxesNeeded = pkg.boxesNeeded || 1;
    var pIdx = getPalletSpecIndex(pkg.material, pkg.l, pkg.w, pkg.h);
    if (pIdx < 0) return;
    var spec = computePalletStacking(pkg.l, pkg.w).spec;
    var remainingBoxes = boxesNeeded;
    while (remainingBoxes > 0) {
      var foundPallet = null;
      for (var pi = 0; pi < pallets.length; pi++) {
        var p = pallets[pi];
        if (p.spec.index !== pIdx) continue;
        if (p.totalBoxes < p.totalCells) { foundPallet = p; break; }
      }
      if (!foundPallet) {
        var totalCells = spec.lDiv * spec.wDiv;
        var cells = [];
        for (var c = 0; c < totalCells; c++) cells.push(-1);
        foundPallet = { seq: [], spec: spec, cells: cells, totalCells: totalCells, totalBoxes: 0 };
        pallets.push(foundPallet);
      }
      for (var ci = 0; ci < foundPallet.totalCells && remainingBoxes > 0; ci++) {
        if (foundPallet.cells[ci] === -1) {
          foundPallet.cells[ci] = idx + 1;
          foundPallet.totalBoxes++;
          if (foundPallet.seq.indexOf(idx + 1) < 0) foundPallet.seq.push(idx + 1);
          remainingBoxes--;
        }
      }
    }
  });
  pallets.forEach(function(p) {
    var cnt = {};
    for (var c = 0; c < p.totalCells; c++) {
      var s = p.cells[c];
      if (s > 0) cnt[s] = (cnt[s] || 0) + 1;
    }
    p.totalBoxes = Object.keys(cnt).length > 0 ? Object.values(cnt).reduce(function(a, b) { return a + b; }, 0) : 0;
    var seen = {};
    p.seq = p.seq.filter(function(s) { if (seen[s]) return false; seen[s] = true; return true; });
  });
  var seqToPallet = {};
  pallets.forEach(function(p, pi) {
    p.seq.forEach(function(s) { seqToPallet[s] = pi + 1; });
  });
  return { pallets: pallets, seqToPallet: seqToPallet };
}

function getPalletSpecIndex(material, pkgL, pkgW, pkgH) {
  if (!canFitOnPallet(material, pkgL, pkgW, pkgH)) return -1;
  return computePalletStacking(pkgL, pkgW).spec ? computePalletStacking(pkgL, pkgW).spec.index : -1;
}

function renderStep2BatchSummary() {
  var section = document.getElementById('step2BatchSummary');
  var head = document.getElementById('step2BatchSummaryHead');
  var body = document.getElementById('step2BatchSummaryBody');
  if (!section || !head || !body) return;
  var results = (state._batchResults && state._batchResults.length > 0) ? state._batchResults : (state.manualPartsList ? convertManualPartsToBatchResults() : []);
  if (results.length === 0) { section.style.display = 'none'; return; }

  var palletL = 1400, palletW = 1060;
  var alloc = computeBatchPalletAllocation(results);
  var pallets = alloc.pallets;
  var seqToPallet = alloc.seqToPallet;

  var ths = ['序号', '零件号', '零件名称', '单车用量', '总数量', '一级包装类型及尺寸', '批组箱数', 'CKD SNP', 'CKD重量（KG）', '一级包装成本（元）', '是否需要装托盘', '托盘号'];
  head.innerHTML = '<tr>' + ths.map(function(h) { return '<th>' + escapeHtml(h) + '</th>'; }).join('') + '</tr>';

  body.innerHTML = results.map(function(r, idx) {
    var seqNum = idx + 1;
    if (r.error) return '<tr><td colspan="12" style="color:var(--danger);font-size:12px;">行 ' + r.rowIndex + ' 预测错误：' + escapeHtml(r.error) + '</td></tr>';
    var pkg = r.pkgMethod;
    if (!pkg) return '<tr><td colspan="12" style="color:var(--gray-2);font-size:12px;">行 ' + r.rowIndex + ' 无推荐包装方案</td></tr>';
    var partNo = r.rawFeatures ? (r.rawFeatures['零件号'] || '') : (state.manualPartsList[idx] ? state.manualPartsList[idx].partNo : '');
    var partName = r.rawFeatures ? (r.rawFeatures['零件名称'] || '') : (state.manualPartsList[idx] ? state.manualPartsList[idx].partName : '');
    var usage = r.rawFeatures ? parseFloat(r.rawFeatures['单车用量']) : (state.manualPartsList[idx] ? state.manualPartsList[idx].usage : 0);
    var totalQty = r.totalParts !== null && r.totalParts !== undefined ? r.totalParts : 1;
    var boxesNeeded = pkg.boxesNeeded;
    var snp = pkg.snp;
    var displaySnp = totalQty < snp ? totalQty : snp;
    var partWeight = 0;
    if (r.rawFeatures) {
      partWeight = parseFloat(r.rawFeatures['零件重量（KG）'] !== undefined ? r.rawFeatures['零件重量（KG）'] : (r.rawFeatures['零件重量(KG)'] !== undefined ? r.rawFeatures['零件重量(KG)'] : (r.rawFeatures['零件重量KG'] !== undefined ? r.rawFeatures['零件重量KG'] : '0'))) || 0;
    } else if (state.manualPartsList[idx]) {
      partWeight = state.manualPartsList[idx].pack.weight || 0;
    }
    var ckdWeight = Math.round(displaySnp * partWeight * 100) / 100;
    var cost = computePkgCost(pkg.l, pkg.w, pkg.h, boxesNeeded, pkg.material);
    var palletNeed = seqToPallet[seqNum] !== undefined;
    var palletText = palletNeed ? '<span style="color:var(--success);font-weight:600;">是</span>' : '否';
    var palletNum = palletNeed ? ('托盘' + seqToPallet[seqNum]) : '';
    return '<tr>' +
      '<td style="text-align:center;">' + seqNum + '</td>' +
      '<td>' + escapeHtml(String(partNo)) + '</td>' +
      '<td>' + escapeHtml(String(partName)) + '</td>' +
      '<td>' + (isNaN(usage) ? '—' : usage) + '</td>' +
      '<td>' + totalQty + '</td>' +
      '<td>' + escapeHtml(pkg.name) + '</td>' +
      '<td>' + boxesNeeded + '</td>' +
      '<td>' + displaySnp + '</td>' +
      '<td>' + ckdWeight.toFixed(2) + '</td>' +
      '<td>' + cost.toFixed(2) + '</td>' +
      '<td>' + palletText + '</td>' +
      '<td>' + (palletNum ? '<b style="color:var(--primary);">' + palletNum + '</b>' : '—') + '</td>' +
    '</tr>';
  }).join('');

  renderPalletDiagrams(pallets, results, palletL, palletW);
  section.style.display = 'block';
}

function convertManualPartsToBatchResults() {
  var results = [];
  state.manualPartsList.forEach(function(item, idx) {
    var pack = item.pack;
    var usage = item.usage || 0;
    var batchQty = pack.batchQty || 48;
    var totalParts = usage > 0 ? batchQty * usage : pack.qty;
    var pkg = item.pkgMethod;
    var mfVals = item.modelFeatureValues || {};
    var batchPkg = pkg ? { l: pkg.pkgL, w: pkg.pkgW, h: pkg.pkgH, snp: pkg.snp, boxesNeeded: pkg.boxesPerCtn, fillRate: pkg.fillRate, name: pkg.name, material: pkg.material } : null;
    results.push({
      rowIndex: idx + 1,
      rawFeatures: { '零件号': pack.partNo, '零件名称': pack.partName, '零件重量（KG）': pack.weight || 0, '零件分类': pack.category, '包装等级分类': mfVals['包装等级分类'] || '', '零件种类': mfVals['零件种类'] || '', '包装袋情况': mfVals['包装袋情况'] || '', '干燥剂情况': mfVals['干燥剂情况'] || '', '单车用量': usage },
      totalParts: totalParts,
      pkgMethod: batchPkg,
      error: null
    });
  });
  return results;
}

function computePkgCost(pkgL, pkgW, pkgH, boxesNeeded, material) {
  var surfaceM2 = 2 * (pkgL * pkgW + pkgL * pkgH + pkgW * pkgH) / 1e6;
  var costPerBox = 0;
  if (material === 'iron') {
    var weight = estimatePkgWeight(pkgL, pkgW, pkgH, null);
    costPerBox = weight * 7.8;
  } else if (material === 'tdg') {
    costPerBox = surfaceM2 * 5.4;
  } else if (material === 'wood') {
    costPerBox = surfaceM2 * 36;
  } else {
    costPerBox = surfaceM2 * 4.6;
  }
  return Math.round(costPerBox * boxesNeeded * 100) / 100;
}

function renderPalletDiagrams(pallets, results, palletL, palletW) {
  var container = document.getElementById('palletDiagramsContainer');
  if (!container) return;
  if (pallets.length === 0) { container.innerHTML = ''; return; }

  var COLORS = ['#4a90e2','#e94b4b','#50c875','#f5a623','#9b59b6','#1abc9c','#e67e22','#34495e','#e91e63','#00bcd4','#8bc34a','#ff5722'];
  var svgW = 320, svgH = 220;
  var padX = 24, padY = 16;
  var innerW = svgW - padX * 2, innerH = svgH - padY * 2 - 20;

  var seqToResult = {};
  results.forEach(function(r, idx) { seqToResult[idx + 1] = r; });

  container.innerHTML = '<div style="margin-top:20px;"><div class="card-title" style="font-size:14px;margin-bottom:12px;"><span class="icon">🚢</span> 托盘码放示意图（俯视图）</div>' +
    '<div style="font-size:12px;color:#888;margin-bottom:12px;">托盘规格：' + palletL + ' mm × ' + palletW + ' mm × 1100 mm（高）</div>' +
    '<div style="display:flex;flex-wrap:wrap;gap:16px;">' +
    pallets.map(function(p, pi) {
      var spec = p.spec;
      var lDiv = spec.lDiv, wDiv = spec.wDiv;
      var cellL = spec.cellL, cellW = spec.cellW;
      var scaleX = innerW / palletL;
      var scaleY = innerH / palletW;
      var scale = Math.min(scaleX, scaleY);
      var pw = palletL * scale;
      var ph = palletW * scale;
      var ox = padX + (innerW - pw) / 2;
      var oy = padY + (innerH - ph) / 2;
      var cellPxW = pw / lDiv;
      var cellPxH = ph / wDiv;

      var rects = '';
      for (var pos = 0; pos < p.totalCells; pos++) {
        var seqNo = p.cells[pos];
        var row = Math.floor(pos / lDiv);
        var col = pos % lDiv;
        var x = ox + col * cellPxW;
        var y = oy + row * cellPxH;

        if (seqNo > 0) {
          var color = COLORS[(seqNo - 1) % COLORS.length];
          var r2 = seqToResult[seqNo];
          var pkgL = r2 && r2.pkgMethod ? r2.pkgMethod.l : cellL;
          var pkgW2 = r2 && r2.pkgMethod ? r2.pkgMethod.w : cellW;
          var fillL = Math.min(1, pkgL / cellL);
          var fillW = Math.min(1, pkgW2 / cellW);
          var boxPxW = cellPxW * fillL;
          var boxPxH = cellPxH * fillW;
          var boxX = x + (cellPxW - boxPxW) / 2;
          var boxY = y + (cellPxH - boxPxH) / 2;
          rects += '<rect x="' + boxX + '" y="' + boxY + '" width="' + boxPxW + '" height="' + boxPxH + '" fill="' + color + '" stroke="#fff" stroke-width="1"/>';
          var fs = Math.min(12, boxPxW / 2, boxPxH / 2);
          if (fs >= 6) {
            rects += '<text x="' + (boxX + boxPxW / 2) + '" y="' + (boxY + boxPxH / 2) + '" text-anchor="middle" dominant-baseline="middle" fill="#fff" font-size="' + fs + '" font-weight="bold">' + seqNo + '</text>';
          }
        } else {
          rects += '<rect x="' + x + '" y="' + y + '" width="' + cellPxW + '" height="' + cellPxH + '" fill="#f0f0f0" stroke="#ccc" stroke-width="0.5"/>';
        }
      }

      rects = '<rect x="' + ox + '" y="' + oy + '" width="' + pw + '" height="' + ph + '" fill="#fff" stroke="#888" stroke-width="1"/>' + rects;
      rects += '<rect x="' + ox + '" y="' + oy + '" width="' + pw + '" height="' + ph + '" fill="none" stroke="#333" stroke-width="2"/>';

      var svg = '<svg width="' + svgW + '" height="' + svgH + '" style="display:block;background:#fafafa;">' + rects + '</svg>';

      var totalBoxArea = 0;
      p.seq.forEach(function(s) {
        var r2 = seqToResult[s];
        if (r2 && r2.pkgMethod) {
          totalBoxArea += (r2.pkgMethod.l * r2.pkgMethod.w);
        } else {
          totalBoxArea += (cellL * cellW);
        }
      });
      var utilPct = Math.round(totalBoxArea / (palletL * palletW) * 100);

      var partInfo = p.seq.map(function(s) {
        var r = seqToResult[s];
        var partNo = r && r.rawFeatures ? (r.rawFeatures['零件号'] || ('零件' + s)) : ('零件' + s);
        var boxes = r && r.pkgMethod ? r.pkgMethod.boxesNeeded : 1;
        var color = COLORS[(s - 1) % COLORS.length];
        return '<span style="display:inline-flex;align-items:center;gap:3px;margin:2px 4px;">' +
          '<span style="width:10px;height:10px;border-radius:2px;background:' + color + ';display:inline-block;flex-shrink:0;"></span>' +
          '<span>' + escapeHtml(String(partNo).slice(-10)) + '×' + boxes + '箱</span>' +
        '</span>';
      }).join('');

      return '<div style="background:#fff;border:1px solid #e0e6f0;border-radius:8px;padding:12px;min-width:260px;">' +
        '<div style="font-size:13px;font-weight:600;margin-bottom:6px;">' +
        '托盘 ' + (pi + 1) + ' ' +
        '<span style="font-weight:normal;color:#666;font-size:11px;">' + lDiv + '×' + wDiv + ' 等分（格 ' + cellL + '×' + cellW + ' mm）</span>' +
        ' <span style="font-weight:normal;color:var(--primary);font-size:12px;">占用率 ' + utilPct + '%</span>' +
        '</div>' +
        svg +
        '<div style="font-size:11px;color:#555;margin-top:6px;line-height:1.8;">' + partInfo + '</div>' +
      '</div>';
    }).join('') + '</div></div>';
}

function exportStep2Results() {
  var section = document.getElementById('step2BatchSummary');
  var tbody = document.getElementById('step2BatchSummaryBody');
  if (!section || section.style.display === 'none' || !tbody || !tbody.children.length) {
    alert('无预测包装方案汇总可导出，请先完成批量预测。');
    return;
  }
  var header = ['序号', '零件号', '零件名称', '单车用量', '总数量', '一级包装类型及尺寸', '批组箱数', 'CKD SNP', 'CKD重量（KG）', '一级包装成本（元）', '是否需要装托盘', '托盘号'];
  var lines = [header];
  tbody.querySelectorAll('tr').forEach(function(tr) {
    var cells = [];
    tr.querySelectorAll('td').forEach(function(td) { cells.push(td.textContent.trim()); });
    if (cells.length >= 12) lines.push(cells);
  });
  var csv = lines.map(function(row) { return row.map(function(v) { return '"' + String(v).replace(/"/g, '""') + '"'; }).join(','); }).join('\n');
  var BOM = '\uFEFF';
  var blob = new Blob([BOM + csv], { type: 'text/csv;charset=utf-8;' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = '预测包装方案汇总_' + new Date().toISOString().slice(0, 10) + '.csv';
  a.click();
  URL.revokeObjectURL(url);
}
