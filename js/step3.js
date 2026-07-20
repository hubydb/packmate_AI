function initStep3() {
  var hasBatch = (state._batchResults && state._batchResults.length > 0) || (state.manualPartsList && state.manualPartsList.length > 0);
  if (hasBatch) {
    document.getElementById('singleSchemePanel').style.display = 'none';
    document.getElementById('batchSchemePanel').style.display = 'block';
    renderBatchSchemeSummary();
  } else if (state.part && state.selectedPkgMethod && state.pkgMethods) {
    document.getElementById('singleSchemePanel').style.display = 'block';
    document.getElementById('batchSchemePanel').style.display = 'none';
    renderSchemeBySelected();
  } else {
    document.getElementById('schemeHeader').innerHTML = '请返回上一步选择包装方式。';
  }
}

function renderSchemeBySelected() {
  const method = state.pkgMethods.find(m => m.id === state.selectedPkgMethod);
  if (!method || !state.part) return;
  const part = state.part;
  const boxesPerCtn = calcBoxesPerContainer(method.pkgL, method.pkgW, method.pkgH, method.snp, part);
  const totalPartsPerCtn = boxesPerCtn * method.snp;
  const totalVolume = method.pkgL * method.pkgW * method.pkgH * boxesPerCtn;
  const ctnVol = part.ctnSpec.l * part.ctnSpec.w * part.ctnSpec.h;
  const fillRate = totalVolume / ctnVol;
  const pkgMatWeight = estimatePkgWeight(method.pkgL, method.pkgW, method.pkgH, part);
  const totalWeightPerCtn = boxesPerCtn * (part.weight * method.snp + pkgMatWeight);
  const batchesNeeded = Math.ceil((part.batchQty || 1) / totalPartsPerCtn);
  const totalPkgCost = method.singlePkgCost;
  const batchPkgCost = boxesPerCtn * batchesNeeded * totalPkgCost;
  state.scheme = { method, boxesPerCtn, totalPartsPerCtn, fillRate, totalWeightPerCtn, batchesNeeded, totalPkgCost, batchPkgCost };

  document.getElementById('schemeHeader').innerHTML = `已选 <b>${escapeHtml(method.name)}</b>，每箱 ${method.snp} 件，${method.pkgL}×${method.pkgW}×${method.pkgH} mm。`;
  const ctnMaxWeight = part.ctnSpec.maxWeight;
  const warnings = [];
  if (method.pkgL > part.ctnSpec.l || method.pkgW > part.ctnSpec.w || method.pkgH > part.ctnSpec.h) warnings.push('包装箱尺寸超出集装箱内径');
  if (method.perBoxWeight > (part.maxLoad || 0) && part.maxLoad > 0) warnings.push('单箱重量超过零件最大承重');
  if (totalWeightPerCtn > ctnMaxWeight) warnings.push('集装箱总重量超过最大载重');
  const rows = [
    ['零件号', escapeHtml(part.partNo), ''],
    ['零件数量', `${part.qty} 件`, '本次采购/生产件数'],
    ['零件尺寸', `${part.l} × ${part.w} × ${part.h} mm`, '长×宽×高'],
    ['预测包装类型', escapeHtml(state.predicted ? state.predicted.predicted_class : '—'), '来自模型'],
    ['包装方式', escapeHtml(method.name), ''],
    ['摆放方式', escapeHtml(method.orientation), ''],
    ['SNP（每箱件数）', `${method.snp} 件`, 'Standard Packing Quantity'],
    ['每集装箱箱数', `${boxesPerCtn} 箱`, '按尺寸+重量取小值'],
    ['每集装箱总件数', `${totalPartsPerCtn} 件`, '箱数 × SNP'],
    ['包箱空间利用率', `${(method.fillRate * 100).toFixed(1)}%`, '零件体积 / 包箱体积'],
    ['集装箱空间利用率', `${(method.containerFillRate * 100).toFixed(1)}%`, '已装包箱总体积 / 集装箱体积'],
    ['单箱总重量', `${method.perBoxWeight.toFixed(1)} kg`, part.maxLoad > 0 ? `零件最大承重 ${part.maxLoad}kg` : `集装箱最大载重 ${ctnMaxWeight}kg`],
    ['集装箱总重量', `${totalWeightPerCtn.toFixed(1)} kg`, `集装箱最大载重 ${ctnMaxWeight}kg`],
    ['经济批次所需箱数', `${batchesNeeded} 箱`, `按批次量 ${part.batchQty} 件计算`],
    ['单个包装箱成本', `¥${totalPkgCost.toFixed(2)}`, '材料+人工+防锈+耗材'],
    ['约束校验', warnings.length ? `<span style="color:var(--danger)">${warnings.join('<br/>')}</span>` : '<span style="color:var(--success)">✅ 全部通过</span>', ''],
  ];
  if (part.loadChars.length > 0) rows.push(['装载特性', part.loadChars.join('、'), '已选择']);
  document.getElementById('schemeBody').innerHTML = rows.map(r => `<tr><td>${r[0]}</td><td><b>${r[1]}</b></td><td>${r[2]}</td></tr>`).join('');
  drawLayout(part, method, boxesPerCtn);
  renderCostSummary();
  renderContainerRecommendation(method, part);
}

function calcBoxesPerContainer(pkgL, pkgW, pkgH, snp, part) {
  const ctnSpec = part.ctnSpec;
  const nx = Math.floor(ctnSpec.l / pkgL);
  const ny = Math.floor(ctnSpec.w / pkgW);
  const nz = Math.floor(ctnSpec.h / pkgH);
  const byVolume = nx * ny * nz;
  const perBoxWeight = part.weight * snp + estimatePkgWeight(pkgL, pkgW, pkgH, part);
  const byWeight = perBoxWeight > 0 ? Math.floor(ctnSpec.maxWeight / perBoxWeight) : byVolume;
  return Math.max(1, Math.min(byVolume, byWeight));
}

function renderCostSummary() {
  if (!state.scheme) return;
  const { boxesPerCtn, totalPartsPerCtn, fillRate, totalWeightPerCtn, batchesNeeded, totalPkgCost, batchPkgCost } = state.scheme;
  document.getElementById('costSummary').innerHTML = [
    ['每箱件数', `${state.scheme.method.snp}`, '件'],
    ['每集装箱箱数', `${boxesPerCtn}`, '箱'],
    ['每集装箱总件数', `${totalPartsPerCtn}`, '件'],
    ['集装箱利用率', `${(fillRate * 100).toFixed(1)}%`, '空间'],
    ['集装箱总重量', `${totalWeightPerCtn.toFixed(1)}`, 'kg'],
    ['单箱成本', `¥${totalPkgCost.toFixed(2)}`, '元/箱'],
    ['批次包装成本', `¥${batchPkgCost.toFixed(2)}`, '元/批次'],
  ].map((item, idx) => `<div class="cost-item ${idx === 1 ? 'highlight' : ''}"><div class="cost-label">${item[0]}</div><div class="cost-value">${item[1]}</div><div class="cost-unit">${item[2]}</div></div>`).join('');
}

function renderContainerRecommendation(method, part) {
  const tbody = document.getElementById('containerBody');
  const rows = Object.entries(CONTAINER_SPECS).map(([key, spec]) => {
    const boxesPerCtn = Math.max(1, Math.min(
      Math.floor(spec.l / method.pkgL) * Math.floor(spec.w / method.pkgW) * Math.floor(spec.h / method.pkgH),
      method.perBoxWeight > 0 ? Math.floor(spec.maxWeight / method.perBoxWeight) : 9999
    ));
    const totalParts = boxesPerCtn * method.snp;
    const usedVol = boxesPerCtn * method.pkgL * method.pkgW * method.pkgH;
    const fillRate = usedVol / (spec.l * spec.w * spec.h);
    const ok = method.pkgL <= spec.l && method.pkgW <= spec.w && method.pkgH <= spec.h && method.perBoxWeight <= spec.maxWeight;
    return `<tr><td>${spec.label}</td><td>${boxesPerCtn} 箱</td><td>${totalParts} 件</td><td>${(fillRate * 100).toFixed(1)}%</td><td>${ok ? '<span style="color:var(--success)">可用</span>' : '<span style="color:var(--danger)">不可用</span>'}</td></tr>`;
  });
  tbody.innerHTML = rows.join('');
}

function drawLayout(part, method, boxes) {
  const canvas = document.getElementById('layoutCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const W = canvas.width, H = canvas.height;
  const ctn = part.ctnSpec;
  const margin = 40;
  const scaleX = (W - margin * 2) / ctn.l;
  const scaleY = (H - margin * 2) / ctn.w;
  const scale = Math.min(scaleX, scaleY);
  const boxW = method.pkgL * scale;
  const boxH = method.pkgW * scale;
  const containerW = ctn.l * scale;
  const containerH = ctn.w * scale;
  const startX = (W - containerW) / 2;
  const startY = (H - containerH) / 2;
  ctx.strokeStyle = '#5d7a96';
  ctx.lineWidth = 2;
  ctx.strokeRect(startX, startY, containerW, containerH);
  ctx.fillStyle = '#f3f5f7';
  ctx.fillRect(startX, startY, containerW, containerH);
  const cols = Math.max(1, Math.floor(ctn.l / method.pkgL));
  const rows = Math.max(1, Math.floor(ctn.w / method.pkgW));
  let drawn = 0;
  ctx.font = '12px sans-serif';
  for (let r = 0; r < rows && drawn < Math.min(boxes, cols * rows); r++) {
    for (let c = 0; c < cols && drawn < Math.min(boxes, cols * rows); c++) {
      const x = startX + c * boxW;
      const y = startY + r * boxH;
      ctx.fillStyle = `rgba(0,82,217,${0.12 + ((drawn % 6) * 0.12)})`;
      ctx.fillRect(x + 2, y + 2, boxW - 4, boxH - 4);
      ctx.strokeStyle = '#0f62fe';
      ctx.strokeRect(x + 2, y + 2, boxW - 4, boxH - 4);
      ctx.fillStyle = '#10233d';
      ctx.fillText(`${drawn + 1}`, x + 10, y + 18);
      drawn += 1;
    }
  }
  ctx.fillStyle = '#4f5d73';
  ctx.fillText(`${ctn.label}  ·  ${ctn.l}×${ctn.w}×${ctn.h}mm`, startX, startY + containerH + 22);
}

function renderBatchSchemeSummary() {
  var panel = document.getElementById('batchSchemePanel');
  var header = document.getElementById('batchSchemeHeader');
  var body = document.getElementById('batchSchemeBody');
  var palletSummary = document.getElementById('batchPalletSummary');
  var costSummary = document.getElementById('batchCostSummary');
  if (!panel || !body || !header || !palletSummary || !costSummary) return;

  var results = (state._batchResults && state._batchResults.length > 0) ? state._batchResults : (state.manualPartsList ? convertManualPartsToBatchResults() : []);
  if (results.length === 0) { panel.style.display = 'none'; return; }

  var specs = {};
  var totalPartCount = 0;
  results.forEach(function(r, idx) {
    if (r.error || !r.pkgMethod) return;
    var pkg = r.pkgMethod;
    totalPartCount += (r.totalParts || 1);
    var rowCost = computePkgCost(pkg.l, pkg.w, pkg.h, pkg.boxesNeeded, pkg.material);
    var key = pkg.material + '|' + pkg.l + '|' + pkg.w + '|' + pkg.h;
    if (!specs[key]) {
      specs[key] = { material: pkg.material, name: pkg.name, l: pkg.l, w: pkg.w, h: pkg.h, count: 0, rowCostSum: 0, snp: pkg.snp || 1, rows: [] };
    }
    specs[key].count += (pkg.boxesNeeded || 1);
    specs[key].rowCostSum += rowCost;
    specs[key].rows.push({ seq: idx + 1, partNo: r.rawFeatures['零件号'] || '', rowCost: rowCost });
  });

  var totalMaterialCost = 0;
  var totalAuxCost = 0;
  var MAT_ORDER = { carton: 0, STD: 1 };
  var specKeys = Object.keys(specs).sort(function(a, b) {
    var sa = specs[a], sb = specs[b];
    var oa = MAT_ORDER[sa.material] !== undefined ? MAT_ORDER[sa.material] : 2;
    var ob = MAT_ORDER[sb.material] !== undefined ? MAT_ORDER[sb.material] : 2;
    if (oa !== ob) return oa - ob;
    return sa.name.localeCompare(sb.name);
  });

  var rowsHtml = specKeys.map(function(key) {
    var s = specs[key];
    var matTotal = s.rowCostSum;
    var matPrice = s.count > 0 ? matTotal / s.count : 0;
    totalMaterialCost += matTotal;
    return '<tr>' +
      '<td>' + escapeHtml(s.name) + '</td>' +
      '<td>' + s.l + '×' + s.w + '×' + s.h + '</td>' +
      '<td style="text-align:center;">' + s.count + '</td>' +
      '<td style="text-align:right;">¥' + matPrice.toFixed(2) + '</td>' +
      '<td style="text-align:right;">¥' + matTotal.toFixed(2) + '</td>' +
      '<td style="text-align:right;">¥0.00</td>' +
      '<td></td>' +
    '</tr>';
  }).join('');

  if (!rowsHtml) rowsHtml = '<tr><td colspan="6" style="text-align:center;color:#888;">无有效数据</td></tr>';
  body.innerHTML = rowsHtml;

  var alloc = computeBatchPalletAllocation(results);
  var palletCount = alloc.pallets.length;
  palletSummary.innerHTML = '<div class="alert" style="background:#e8f0fe;color:#0052d9;">' +
    '<b>托盘需求：</b>共需 ' + palletCount + ' 个 148A 托盘' +
    '</div>';

  var grandTotal = totalMaterialCost;
  costSummary.innerHTML = '<div class="cost-summary" style="display:flex;flex-wrap:wrap;gap:10px;">' +
    '<div class="cost-item"><div class="cost-label">总零件数</div><div class="cost-value">' + totalPartCount + '</div><div class="cost-unit">件</div></div>' +
    '<div class="cost-item"><div class="cost-label">包装材料费</div><div class="cost-value">¥' + totalMaterialCost.toFixed(2) + '</div><div class="cost-unit">元</div></div>' +
    '<div class="cost-item"><div class="cost-label">辅料费</div><div class="cost-value">¥0.00</div><div class="cost-unit">元</div></div>' +
    '<div class="cost-item highlight"><div class="cost-label">合计</div><div class="cost-value">¥' + grandTotal.toFixed(2) + '</div><div class="cost-unit">元</div></div>' +
  '</div>';

  header.innerHTML = '共 ' + Object.keys(specs).length + ' 种包装规格，' + results.length + ' 个零件，' + totalPartCount + ' 件。';
  panel.style.display = 'block';
}

function exportStep3Results() {
  if (!state.part || !state.scheme) { alert('无方案数据可导出，请先生成包装方案。'); return; }
  var part = state.part;
  var scheme = state.scheme;
  var method = scheme.method;
  var detailHeader = ['项目', '结果', '说明'];
  var detailRows = [
    ['零件号', escapeHtml(part.partNo), ''],
    ['零件名称', escapeHtml(part.partName), ''],
    ['零件数量', part.qty + ' 件', '本次采购/生产件数'],
    ['零件尺寸', part.l + ' × ' + part.w + ' × ' + part.h + ' mm', '长×宽×高'],
    ['零件重量', part.weight + ' kg', ''],
    ['零件分类', part.category, ''],
    ['集装箱', part.ctnSpec.label, ''],
    ['预测包装类型', state.predicted ? state.predicted.predicted_class : '—', '来自模型'],
    ['包装方式', escapeHtml(method.name), ''],
    ['摆放方式', escapeHtml(method.orientation), ''],
    ['SNP（每箱件数）', method.snp + ' 件', 'Standard Packing Quantity'],
    ['每集装箱箱数', scheme.boxesPerCtn + ' 箱', '按尺寸+重量取小值'],
    ['每集装箱总件数', scheme.totalPartsPerCtn + ' 件', '箱数 × SNP'],
    ['包箱空间利用率', (method.fillRate * 100).toFixed(1) + '%', '零件体积 / 包箱体积'],
    ['集装箱空间利用率', (method.containerFillRate * 100).toFixed(1) + '%', '已装包箱总体积 / 集装箱体积'],
    ['单箱总重量', method.perBoxWeight.toFixed(1) + ' kg', part.maxLoad > 0 ? '零件最大承重 ' + part.maxLoad + 'kg' : '集装箱最大载重 ' + part.ctnSpec.maxWeight + 'kg'],
    ['集装箱总重量', scheme.totalWeightPerCtn.toFixed(1) + ' kg', '集装箱最大载重 ' + part.ctnSpec.maxWeight + 'kg'],
    ['经济批次所需箱数', scheme.batchesNeeded + ' 箱', '按批次量 ' + part.batchQty + ' 件计算'],
    ['单个包装箱成本', '¥' + scheme.totalPkgCost.toFixed(2), '材料+人工+防锈+耗材'],
    ['批次包装成本', '¥' + scheme.batchPkgCost.toFixed(2), '元/批次'],
  ];
  var ctnHeader = ['集装箱', '可装箱数', '总件数', '空间利用率', '状态'];
  var ctnRows = [ctnHeader];
  Object.entries(CONTAINER_SPECS).forEach(function(entry) {
    var key = entry[0], spec = entry[1];
    var boxesPerCtn = Math.max(1, Math.min(
      Math.floor(spec.l / method.pkgL) * Math.floor(spec.w / method.pkgW) * Math.floor(spec.h / method.pkgH),
      method.perBoxWeight > 0 ? Math.floor(spec.maxWeight / method.perBoxWeight) : 9999
    ));
    var totalParts = boxesPerCtn * method.snp;
    var usedVol = boxesPerCtn * method.pkgL * method.pkgW * method.pkgH;
    var fillRate = usedVol / (spec.l * spec.w * spec.h);
    var ok = method.pkgL <= spec.l && method.pkgW <= spec.w && method.pkgH <= spec.h && method.perBoxWeight <= spec.maxWeight;
    ctnRows.push([spec.label, boxesPerCtn + ' 箱', totalParts + ' 件', (fillRate * 100).toFixed(1) + '%', ok ? '可用' : '不可用']);
  });
  var costHeader = ['指标', '数值', '单位'];
  var costRows = [
    ['每箱件数', state.scheme.method.snp, '件'],
    ['每集装箱箱数', scheme.boxesPerCtn, '箱'],
    ['每集装箱总件数', scheme.totalPartsPerCtn, '件'],
    ['集装箱利用率', (scheme.fillRate * 100).toFixed(1), '%'],
    ['集装箱总重量', scheme.totalWeightPerCtn.toFixed(1), 'kg'],
    ['单箱成本', '¥' + scheme.totalPkgCost.toFixed(2), '元/箱'],
    ['批次包装成本', '¥' + scheme.batchPkgCost.toFixed(2), '元/批次'],
  ];
  var allLines = [detailHeader.map(function(h) { return '"' + h + '"'; }).join(',')];
  allLines = allLines.concat(detailRows.map(function(row) { return row.map(function(v) { return '"' + String(v).replace(/"/g, '""') + '"'; }).join(','); }));
  allLines.push('');
  allLines.push(ctnHeader.map(function(h) { return '"' + h + '"'; }).join(','));
  allLines = allLines.concat(ctnRows.slice(1).map(function(row) { return row.map(function(v) { return '"' + String(v).replace(/"/g, '""') + '"'; }).join(','); }));
  allLines.push('');
  allLines.push(costHeader.map(function(h) { return '"' + h + '"'; }).join(','));
  allLines = allLines.concat(costRows.map(function(row) { return row.map(function(v) { return '"' + String(v).replace(/"/g, '""') + '"'; }).join(','); }));
  var csv = allLines.join('\n');
  var BOM = '\uFEFF';
  var blob = new Blob([BOM + csv], { type: 'text/csv;charset=utf-8;' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'pkg_scheme_' + (part.partNo || 'export') + '_' + new Date().toISOString().slice(0, 10) + '.csv';
  a.click();
  URL.revokeObjectURL(url);
}

function exportBatchSchemeResults() {
  var batchPanel = document.getElementById('batchSchemePanel');
  var body = document.getElementById('batchSchemeBody');
  if (!batchPanel || batchPanel.style.display === 'none' || !body || !body.children.length) {
    alert('无批量成本数据可导出，请先进行批量预测并进入批量成本测算页面。');
    return;
  }
  var lines = [];
  lines.push(['包装材料', '规格(mm)', '数量', '单价(元)', '小计(元)', '辅料(元)']);
  body.querySelectorAll('tr').forEach(function(tr) {
    var cells = [];
    tr.querySelectorAll('td').forEach(function(td) { cells.push(td.textContent.trim().replace('¥', '').replace('元', '').trim()); });
    if (cells.length >= 5) lines.push([cells[0], cells[1], cells[2], cells[3], cells[4], cells[5] || '0']);
  });
  var palletSummary = document.getElementById('batchPalletSummary');
  if (palletSummary) { lines.push(['', '']); lines.push(['托盘需求', palletSummary.textContent.trim()]); }
  var costSummary = document.getElementById('batchCostSummary');
  if (costSummary) {
    lines.push(['', '']);
    lines.push(['指标', '数值', '单位']);
    costSummary.querySelectorAll('.cost-item').forEach(function(item) {
      var label = item.querySelector('.cost-label');
      var value = item.querySelector('.cost-value');
      var unit = item.querySelector('.cost-unit');
      if (label) lines.push([label.textContent.trim(), value ? value.textContent.trim() : '', unit ? unit.textContent.trim() : '']);
    });
  }
  var csv = lines.map(function(row) { return row.map(function(v) { return '"' + String(v).replace(/"/g, '""') + '"'; }).join(','); }).join('\n');
  var BOM = '\uFEFF';
  var blob = new Blob([BOM + csv], { type: 'text/csv;charset=utf-8;' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'batch_cost_' + new Date().toISOString().slice(0, 10) + '.csv';
  a.click();
  URL.revokeObjectURL(url);
}
