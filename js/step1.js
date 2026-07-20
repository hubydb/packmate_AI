function initStep1() {
  setupModelUI();
  initImageUpload();
  renderManualPartsList();
  updateContainerInputs();
}

function updateContainerInputs() {
  const spec = CONTAINER_SPECS[state.selectedContainer];
  if (document.getElementById('ctnL')) {
    document.getElementById('ctnL').value = spec.l;
    document.getElementById('ctnW').value = spec.w;
    document.getElementById('ctnH').value = spec.h;
    document.getElementById('ctnMaxW').value = spec.maxWeight;
  }
}

function selectContainer(el) {
  document.querySelectorAll('.container-chip').forEach(c => c.classList.remove('selected'));
  el.classList.add('selected');
  state.selectedContainer = el.dataset.type;
  updateContainerInputs();
  persistState();
}

function onLoadCharsChange() {
  const checkboxes = document.querySelectorAll('#loadCharsCheckboxes input[type="checkbox"]:checked');
  state.loadChars = Array.from(checkboxes).map(cb => cb.value);
  persistState();
}

function getPackInput() {
  const loadCharsCheckboxes = document.querySelectorAll('#loadCharsCheckboxes input[type="checkbox"]:checked');
  const loadChars = Array.from(loadCharsCheckboxes).map(cb => cb.value);
  return {
    partNo: document.getElementById('partNo').value.trim(),
    partName: document.getElementById('partName').value.trim(),
    category: document.getElementById('partCategory').value,
    l: parseFloat(document.getElementById('partL').value) || 0,
    w: parseFloat(document.getElementById('partW').value) || 0,
    h: parseFloat(document.getElementById('partH').value) || 0,
    weight: parseFloat(document.getElementById('partWeight').value) || 0,
    maxLoad: parseFloat(document.getElementById('partMaxLoad').value) || 0,
    loadChars: loadChars,
    qty: parseInt(document.getElementById('partQty').value) || 0,
    batchQty: parseInt(document.getElementById('batchQty').value) || 48,
    buffer: parseFloat(document.getElementById('bufferThickness').value) || 0,
    tolerance: parseFloat(document.getElementById('deformationTol').value) || 0,
    material: document.getElementById('pkgMaterial').value,
    antiRust: document.getElementById('antiRust').value === 'yes',
    ctnSpec: CONTAINER_SPECS[state.selectedContainer],
  };
}

function validatePackInput(p) {
  if (!p.partNo) return '请输入零件号';
  if (!p.partName) return '请输入零件名称';
  if (!p.l || !p.w || !p.h) return '请输入零部件的长、宽、高';
  return null;
}

function syncPackInputsFromModel() {
  state.featureMeta.forEach(meta => {
    const input = document.getElementById('feat_' + meta.name);
    if (!input) return;
    let value = input.value;
    if (value === undefined || value === null) value = '';
    value = String(value).trim();
    if (meta.name === '零件号') {
      document.getElementById('partNo').value = value || document.getElementById('partNo').value || '';
    } else if (meta.name === '零件名称') {
      document.getElementById('partName').value = value || document.getElementById('partName').value || '';
    } else if (meta.name === '零件重量（KG）') {
      const num = parseFloat(value);
      if (Number.isFinite(num)) document.getElementById('partWeight').value = num;
    } else if (meta.name === '包装等级分类') {
      const el = document.getElementById('feat_包装等级分类');
      if (el) el.value = value;
    } else if (meta.name === '零件分类' && /^[ABCabc]$/.test(value)) {
      document.getElementById('partCategory').value = value.toUpperCase();
    }
  });
  autoFillPackageInputs();
}

function collectModelFeatures() {
  var qualitySet = {};
  var keywords = (state.model && state.model.quality_keywords) || [];
  if (keywords.length === 0) {
    var fallbackInput = document.getElementById('loadCharsTextFallback');
    var text = fallbackInput ? (fallbackInput.value || '') : '';
    _parseQualityText(text).forEach(function(kw) { qualitySet[kw] = true; });
  }

  return state.featureMeta.map(meta => {
    if (meta.name.startsWith('质量_')) {
      var kw = meta.name.slice(2);
      if (keywords.length > 0) {
        var chk = document.getElementById('load_char_' + kw);
        return (chk && chk.checked) ? 1 : 0;
      } else {
        return qualitySet[kw] ? 1 : 0;
      }
    }
    let input = document.getElementById('feat_' + meta.name);
    let raw = '';
    if (input) {
      raw = String(input.value || '').trim();
    } else {
      switch (meta.name) {
        case '零件号': raw = String(document.getElementById('partNo').value || '').trim(); break;
        case '零件名称': raw = String(document.getElementById('partName').value || '').trim(); break;
        case '包装等级分类': raw = String(document.getElementById('feat_包装等级分类').value || '').trim(); break;
        case '零件重量（KG）': raw = String(document.getElementById('partWeight').value || '').trim(); break;
        case '零件分类': raw = String(document.getElementById('partCategory').value || '').trim(); break;
      }
    }
    if (!raw) return NaN;
    if (meta.kind === 'numeric') {
      const num = parseFloat(raw);
      return Number.isFinite(num) ? num : NaN;
    }
    const mapping = state.model.feature_cat_encoding && state.model.feature_cat_encoding[meta.name];
    if (!mapping || !Object.prototype.hasOwnProperty.call(mapping, raw)) return NaN;
    return mapping[raw];
  });
}

function applyPredictionToPackParams(result) {
  const cls = String(result.predicted_class || '');
  const materialEl = document.getElementById('pkgMaterial');
  if (cls.includes('纸箱') || cls.includes('天地盖') || cls === 'STD') {
    materialEl.value = 'carton';
  } else if (cls.includes('木箱')) {
    materialEl.value = 'wood';
  } else if (cls.includes('铁箱')) {
    materialEl.value = 'iron';
  }
}

function runPrediction() {
  if (!state.model) { alert('模型尚未加载，请先在训练页面完成模型训练。'); return; }
  if (state.manualPartsList && state.manualPartsList.length > 0) {
    state.part = null;
    state.predicted = null;
    state.pkgMethods = null;
    state.selectedPkgMethod = null;
    persistState();
    location.href = '/step2.html';
    return;
  }
  syncPackInputsFromModel();
  const modelFeatures = collectModelFeatures();
  const result = predictLGB(state.model, modelFeatures);
  state.predicted = result;
  applyPredictionToPackParams(result);

  const pack = getPackInput();
  const err = validatePackInput(pack);
  if (err) { alert(err); return; }
  state.part = pack;
  persistState();
  location.href = '/step2.html';
}

function autoFillPackageInputs() {
  document.getElementById('partNo').value = document.getElementById('partNo').value || 'AUTO-' + Date.now().toString().slice(-6);
  document.getElementById('partName').value = document.getElementById('partName').value || '自动示例零件';
  document.getElementById('partL').value = document.getElementById('partL').value || 280;
  document.getElementById('partW').value = document.getElementById('partW').value || 180;
  document.getElementById('partH').value = document.getElementById('partH').value || 120;
  document.getElementById('partWeight').value = document.getElementById('partWeight').value || 1.2;
  document.getElementById('partMaxLoad').value = document.getElementById('partMaxLoad').value || 40;
  document.getElementById('partQty').value = document.getElementById('partQty').value || 96;
}

function clearPackageInputs() {
  ['partNo','partName','partL','partW','partH','partWeight','partMaxLoad','partQty','operator','remark'].forEach(id => {
    const el = document.getElementById(id);
    if (el && !el.readOnly) el.value = '';
  });
  ['bufferThickness','deformationTol'].forEach(id => { document.getElementById(id).value = 0; });
  document.getElementById('batchQty').value = 48;
  document.getElementById('pkgMaterial').value = 'wood';
  document.getElementById('antiRust').value = 'no';
  document.querySelectorAll('#loadCharsCheckboxes input[type="checkbox"]').forEach(cb => cb.checked = false);
  document.getElementById('feat_包装等级分类').value = '';
  document.getElementById('partCategory').value = 'B';
  state.loadChars = [];
  persistState();
}

function initImageUpload() {
  const imgUpload = document.getElementById('imgUpload');
  const imgFile = document.getElementById('imgFile');
  if (!imgUpload || !imgFile) return;
  imgUpload.addEventListener('click', () => imgFile.click());
  imgUpload.addEventListener('dragover', (e) => { e.preventDefault(); imgUpload.style.borderColor = 'var(--primary)'; });
  imgUpload.addEventListener('dragleave', () => { imgUpload.style.borderColor = ''; });
  imgUpload.addEventListener('drop', (e) => {
    e.preventDefault();
    imgUpload.style.borderColor = '';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleImageFile(file);
  });
  imgFile.addEventListener('change', () => {
    if (imgFile.files[0]) handleImageFile(imgFile.files[0]);
  });
  if (state.imageData) {
    document.getElementById('imgPreview').src = state.imageData;
    document.getElementById('imgPreviewWrap').style.display = 'block';
    imgUpload.style.display = 'none';
  }
}

function handleImageFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('imgPreview').src = e.target.result;
    document.getElementById('imgPreviewWrap').style.display = 'block';
    document.getElementById('imgUpload').style.display = 'none';
    state.imageData = e.target.result;
    persistState();
  };
  reader.readAsDataURL(file);
}

function removeImage() {
  document.getElementById('imgPreview').src = '';
  document.getElementById('imgPreviewWrap').style.display = 'none';
  document.getElementById('imgUpload').style.display = 'flex';
  document.getElementById('imgFile').value = '';
  state.imageData = null;
  persistState();
}

function initFeatureSelect(id, featureName) {
  const select = document.getElementById(id);
  if (!select) return;
  const meta = state.featureMeta.find(m => m.name === featureName);
  let cats = [];
  if (meta && meta.categories) cats = meta.categories.filter(v => v !== '__MISSING__');
  select.innerHTML = '<option value="">-- 请选择 --</option>';
  cats.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    select.appendChild(opt);
  });
}

function initLoadCharsCheckboxes() {
  var container = document.getElementById('loadCharsCheckboxes');
  if (!container) return;
  var keywords = (state.model && state.model.quality_keywords) || [];
  container.innerHTML = '';
  if (keywords.length === 0) {
    var hint = document.createElement('span');
    hint.style.cssText = 'font-size:12px;color:#e37318;margin-right:8px;';
    hint.textContent = '（请重新训练模型）';
    container.appendChild(hint);
    var input = document.createElement('input');
    input.type = 'text';
    input.id = 'loadCharsTextFallback';
    input.placeholder = '粘贴零件质量要求文本，系统自动解析防XX关键词';
    input.style.cssText = 'border:1px solid #c0c6d1;border-radius:4px;padding:2px 8px;font-size:13px;width:320px;';
    container.appendChild(input);
    return;
  }
  keywords.forEach(function(kw) {
    var label = document.createElement('label');
    label.style.cssText = 'display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border:1px solid #c0c6d1;border-radius:12px;cursor:pointer;font-size:13px;user-select:none;white-space:nowrap;';
    var checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = 'load_char_' + kw;
    checkbox.value = kw;
    checkbox.style.margin = '0';
    checkbox.addEventListener('change', onLoadCharsChange);
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(kw));
    container.appendChild(label);
  });
}

function setupModelUI() {
  state.featureMeta = state.model ? (state.model.feature_meta || []) : [];
  const badge = document.getElementById('statusBadge');
  const modeTag = document.getElementById('modeTag');
  if (!state.model) {
    if (badge) { badge.textContent = '未加载模型'; badge.className = 'badge'; badge.style.background = '#fef0ef'; badge.style.color = '#d54941'; }
    return;
  }
  if (badge) { badge.textContent = '模型已加载'; badge.style.background = '#e8f8ef'; badge.style.color = '#2ba471'; }
  if (modeTag) { modeTag.textContent = '在线预测模式'; modeTag.className = 'mode-tag online'; }
  initFeatureSelect('feat_包装袋情况', '包装袋情况');
  initFeatureSelect('feat_干燥剂情况', '干燥剂情况');
  initLoadCharsCheckboxes();
  initFeatureSelect('feat_零件种类', '零件种类');
  initFeatureSelect('feat_包装等级分类', '包装等级分类');
}

function _parseQualityText(text) {
  if (!text || typeof text !== 'string') return [];
  var matches = text.match(/防[\u4e00-\u9fa5a-zA-Z0-9]+/g) || [];
  return matches;
}

// 手动录入零件
function addManualPart() {
  if (!state.model) { alert('模型尚未加载，请先在训练页面完成模型训练。'); return; }
  syncPackInputsFromModel();
  const pack = getPackInput();
  const err = validatePackInput(pack);
  if (err) { alert(err); return; }

  const modelFeatures = collectModelFeatures();
  const result = predictLGB(state.model, modelFeatures);
  state.predicted = result;
  applyPredictionToPackParams(result);
  state.part = pack;
  state.pkgMethods = computePkgMethods(pack);
  sortPkgMethodsByConfidence(result);

  if (!state.pkgMethods.length) {
    var constraintErr = checkPkgConstraints(pack);
    alert(constraintErr ? '无法推荐包装方式，原因如下：\n\n' + constraintErr : '无法推荐包装方式，请检查输入参数是否超出包箱限制。');
    return;
  }

  const bestMethod = state.pkgMethods[0];
  const usage = parseFloat(document.getElementById('feat_单车用量').value) || 0;
  const batchQty = parseInt(document.getElementById('batchQty').value) || 48;
  const totalParts = usage > 0 ? batchQty * usage : pack.qty;

  var modelFeatureValues = {};
  state.featureMeta.forEach(function(meta) {
    var input = document.getElementById('feat_' + meta.name);
    if (input) modelFeatureValues[meta.name] = input.value || '';
  });

  state.manualPartsList.push({
    partNo: pack.partNo,
    partName: pack.partName,
    pack: pack,
    features: modelFeatures,
    prediction: result,
    pkgMethod: bestMethod,
    totalParts: totalParts,
    usage: usage,
    modelFeatureValues: modelFeatureValues,
    pkgMethods: state.pkgMethods.slice(),
    selectedPkgMethod: bestMethod.id,
  });
  persistState();

  document.getElementById('partNo').value = '';
  document.getElementById('partName').value = '';
  document.getElementById('partL').value = '';
  document.getElementById('partW').value = '';
  document.getElementById('partH').value = '';
  document.getElementById('partWeight').value = '';
  document.getElementById('partMaxLoad').value = '';
  document.getElementById('partQty').value = '';
  document.getElementById('bufferThickness').value = '0';
  document.getElementById('deformationTol').value = '0';
  document.getElementById('antiRust').value = 'no';
  document.getElementById('remark').value = '';
  document.querySelectorAll('#loadCharsCheckboxes input[type="checkbox"]').forEach(function(cb) { cb.checked = false; });
  state.loadChars = [];
  clearModelInputs();
  renderManualPartsList();
}

function renderManualPartsList() {
  var card = document.getElementById('manualPartsCard');
  var head = document.getElementById('manualPartsHead');
  var body = document.getElementById('manualPartsBody');
  var countEl = document.getElementById('manualPartsCount');
  var alertEl = document.getElementById('manualPartsAlert');
  var toStep2Btn = document.getElementById('toStep2Btn');
  if (!card || !head || !body) return;

  var list = state.manualPartsList;
  if (list.length === 0) { card.style.display = 'none'; return; }
  card.style.display = 'block';
  if (alertEl) alertEl.style.display = 'block';
  if (countEl) countEl.textContent = '共 ' + list.length + ' 个零件';
  if (toStep2Btn) toStep2Btn.style.display = 'inline-flex';

  var ths = ['#', '零件号', '零件名称', '零件尺寸(mm)', '单车用量', '零件重量(KG)', '预测类型', '置信度', '零件总量', '包装方式', '操作'];
  head.innerHTML = '<tr>' + ths.map(function(h) { return '<th>' + escapeHtml(h) + '</th>'; }).join('') + '</tr>';

  body.innerHTML = list.map(function(item, idx) {
    var pkg = item.pkgMethod;
    var predClass = item.prediction ? item.prediction.predicted_class : '—';
    var confPct = item.prediction && item.prediction.confidence !== undefined ? (item.prediction.confidence * 100).toFixed(2) + '%' : '—';
    var pkgBtnText = pkg ? pkg.name : '—';
    var pkgBtnStyle = pkg ? 'background:#0052d9;color:#fff;border-color:#0052d9;' : '';
    var pkgBtn = '<button class="btn" style="height:26px;padding:0 10px;font-size:12px;cursor:pointer;' + pkgBtnStyle + '" onclick="openManualPartPkgModal(' + idx + ')">' + escapeHtml(pkgBtnText) + '</button>';
    return '<tr>' +
      '<td style="text-align:center;">' + (idx + 1) + '</td>' +
      '<td>' + escapeHtml(String(item.partNo)) + '</td>' +
      '<td>' + escapeHtml(String(item.partName)) + '</td>' +
      '<td>' + item.pack.l + '×' + item.pack.w + '×' + item.pack.h + '</td>' +
      '<td>' + (isNaN(item.usage) ? '—' : item.usage) + '</td>' +
      '<td>' + (item.pack.weight || '—') + '</td>' +
      '<td style="font-weight:600;color:var(--primary);">' + escapeHtml(predClass) + '</td>' +
      '<td>' + confPct + '</td>' +
      '<td>' + (item.totalParts !== null && item.totalParts !== undefined ? item.totalParts.toFixed(0) : '—') + '</td>' +
      '<td>' + pkgBtn + '</td>' +
      '<td><button class="btn btn-danger" style="height:26px;padding:0 10px;font-size:12px;" onclick="deleteManualPart(' + idx + ')">删除</button></td>' +
    '</tr>';
  }).join('');
}

function deleteManualPart(idx) {
  if (idx >= 0 && idx < state.manualPartsList.length) {
    state.manualPartsList.splice(idx, 1);
    persistState();
    renderManualPartsList();
  }
}

function clearManualPartsList() {
  state.manualPartsList = [];
  persistState();
  renderManualPartsList();
}

function goStep2FromManual() {
  if (state.manualPartsList.length === 0) { alert('请先添加零件'); return; }
  state.part = null;
  state.predicted = null;
  state.pkgMethods = null;
  state.selectedPkgMethod = null;
  persistState();
  location.href = '/step2.html';
}

function clearModelInputs() {
  state.featureMeta.forEach(meta => {
    const input = document.getElementById('feat_' + meta.name);
    if (input) input.value = '';
  });
}

// 批量预测
function handleBatchCsv(file) {
  if (!file) return;
  var nameEl = document.getElementById('batchFileName');
  var btn = document.getElementById('batchPredictBtn');
  nameEl.textContent = '读取中...';
  state._batchCsvData = null;
  state._batchResults = null;
  document.getElementById('batchResultsArea').style.display = 'none';
  document.getElementById('batchExportBtn').disabled = true;
  document.getElementById('batchProgressArea').style.display = 'none';

  var reader = new FileReader();
  reader.onload = function(e) {
    try {
      var text = e.target.result;
      var lines = text.trim().split('\n');
      if (lines.length < 2) { alert('CSV 文件内容不足'); nameEl.textContent = ''; return; }
      function parseCSVLine(line) {
        var result = [], current = '', inQuote = false;
        for (var i = 0; i < line.length; i++) {
          var ch = line[i];
          if (ch === '"') {
            if (inQuote && line[i + 1] === '"') { current += '"'; i++; }
            else inQuote = !inQuote;
          } else if (ch === ',' && !inQuote) { result.push(current.trim()); current = ''; }
          else { current += ch; }
        }
        result.push(current.trim());
        return result;
      }
      var header = parseCSVLine(lines[0]);
      var rows = lines.slice(1).map(function(line) { return parseCSVLine(line); });
      state._batchCsvData = { header: header, rows: rows };
      nameEl.textContent = '已选: ' + file.name + ' (' + rows.length + ' 条)';
      btn.disabled = false;
      persistState();
    } catch (err) { alert('CSV 解析失败: ' + err.message); nameEl.textContent = ''; btn.disabled = true; }
  };
  reader.onerror = function() { alert('文件读取失败'); nameEl.textContent = ''; };
  reader.readAsText(file);
}

function _buildFeaturesFromRow(rowObj) {
  var model = state.model;
  if (!model || !model.feature_names) return [];
  var names = model.feature_names;
  var catEnc = model.feature_cat_encoding || {};
  var rawQualityText = rowObj['零件质量要求'] || '';
  var qualitySet = {};
  _parseQualityText(rawQualityText).forEach(function(kw) { qualitySet[kw] = true; });

  return names.map(function(name) {
    if (name.startsWith('质量_')) {
      var kw = name.slice(2);
      return qualitySet[kw] ? 1 : 0;
    }
    var v = rowObj[name];
    if (v === undefined || v === null || v === '') return NaN;
    v = String(v).trim();
    if (v === '') return NaN;
    if (catEnc[name]) {
      var mapping = catEnc[name];
      if (mapping.hasOwnProperty(v)) return mapping[v];
      return NaN;
    }
    var n = parseFloat(v);
    return isNaN(n) ? NaN : n;
  });
}

function computeBestPkgMethodForRow(rowObj, predictedClass, totalParts) {
  var material = 'carton';
  if (predictedClass.includes('木箱')) material = 'wood';
  else if (predictedClass.includes('铁') || predictedClass.includes('铁架') || predictedClass.includes('轻钢')) material = 'iron';
  else if (predictedClass.includes('天地盖')) material = 'tdg';
  else if (predictedClass === 'STD') material = 'STD';
  else if (predictedClass.includes('纸箱')) material = 'carton';

  var partL = parseFloat(rowObj['长'] || rowObj['零件尺寸L'] || 100);
  var partW = parseFloat(rowObj['宽'] || rowObj['零件尺寸W'] || 100);
  var partH = parseFloat(rowObj['高'] || rowObj['零件尺寸H'] || 100);
  var totalQty = !isNaN(totalParts) && totalParts > 0 ? totalParts : 1;
  var ctnSpec = CONTAINER_SPECS[state.selectedContainer] || CONTAINER_SPECS['20gp'];
  var matInfo = MATERIAL_PRICES[material];
  var sizes = MODAL_BOX_SIZES[material] || [];

  var options = sizes.map(function(s) {
    var pL = partL, pW = partW, pH = partH;
    var orientations = [{d:[pL,pW,pH]},{d:[pW,pL,pH]},{d:[pH,pW,pL]},{d:[pH,pL,pW]},{d:[pL,pH,pW]},{d:[pW,pH,pL]}];
    var viable = orientations.filter(function(o) { return o.d[0] <= s.l && o.d[1] <= s.w && o.d[2] <= s.h; });
    if (!viable.length) return null;
    var best = viable[0], bestSnp = 0;
    viable.forEach(function(o) {
      var snp = Math.max(1, Math.floor(s.l / o.d[0])) * Math.max(1, Math.floor(s.w / o.d[1])) * Math.max(1, Math.floor(s.h / o.d[2]));
      if (snp > bestSnp) { bestSnp = snp; best = o; }
    });
    var boxesNeeded = Math.ceil(totalQty / bestSnp);
    var partVol = pL * pW * pH;
    var boxVol = s.l * s.w * s.h;
    var fillRate = (bestSnp * partVol) / boxVol;
    var orderFillRate = (totalQty * partVol) / (boxesNeeded * boxVol);
    return { l: s.l, w: s.w, h: s.h, snp: bestSnp, boxesNeeded: boxesNeeded, fillRate: fillRate, orderFillRate: orderFillRate, name: matInfo.name + ' ' + s.l + '×' + s.w + '×' + s.h, material: material };
  }).filter(function(o) { return o !== null; });

  options.sort(function(a, b) {
    if (a.boxesNeeded !== b.boxesNeeded) return a.boxesNeeded - b.boxesNeeded;
    return b.orderFillRate - a.orderFillRate;
  });
  return options.length > 0 ? options[0] : null;
}

function runBatchPrediction() {
  if (!state._batchCsvData || state._batchCsvData.rows.length === 0) { alert('请先选择一个有效的 CSV 文件'); return; }
  if (!state.model) { alert('模型未加载，请先完成训练'); return; }
  var header = state._batchCsvData.header;
  var rows = state._batchCsvData.rows;
  var totalAll = rows.length;

  var batchSize = parseInt(document.getElementById('batchSizeInput').value, 10);
  if (!batchSize || batchSize < 1) batchSize = 1;

  document.getElementById('batchProgressArea').style.display = 'block';
  document.getElementById('batchResultsArea').style.display = 'none';
  document.getElementById('batchPredictBtn').disabled = true;

  var results = [];
  var i = 0;
  var CHUNK = 10;

  function processChunk() {
    var chunkEnd = Math.min(i + CHUNK, totalAll);
    for (var j = i; j < chunkEnd; j++) {
      var row = rows[j];
      var rowObj = {};
      header.forEach(function(h, idx) { rowObj[h] = row[idx] !== undefined ? row[idx] : ''; });
      var features = _buildFeaturesFromRow(rowObj);
      var usage = parseFloat(rowObj['单车用量']);
      var totalParts = !isNaN(usage) ? batchSize * usage : null;
      try {
        var pred = predictLGB(state.model, features);
        var confPct = pred.confidence !== undefined && pred.confidence !== null ? (pred.confidence * 100).toFixed(2) + '%' : '—';
        var confVal = pred.confidence !== undefined && pred.confidence !== null ? parseFloat((pred.confidence * 100).toFixed(2)) : 0;
        var actualClass = rowObj['CKD包装类型'] || rowObj['ckd包装类型'] || '';
        var pkgMethod = computeBestPkgMethodForRow(rowObj, pred.predicted_class, totalParts);
        results.push({ rowIndex: j + 1, rawFeatures: rowObj, actualClass: actualClass, predictedClass: pred.predicted_class, confidence: confPct, confidenceVal: confVal, allProbabilities: pred.all_probabilities || [], totalParts: totalParts, pkgMethod: pkgMethod, error: null });
      } catch (err) {
        results.push({ rowIndex: j + 1, rawFeatures: rowObj, actualClass: rowObj['CKD包装类型'] || '', predictedClass: '错误', confidence: '—', confidenceVal: 0, allProbabilities: [], totalParts: totalParts, pkgMethod: null, error: err.message });
      }
    }
    i = chunkEnd;
    var pct = Math.round(i / totalAll * 100);
    document.getElementById('batchProgressBar').style.width = pct + '%';
    document.getElementById('batchProgressText').textContent = '已预测 ' + i + ' / ' + totalAll + ' 条（' + pct + '%）';
    if (i < totalAll) {
      requestAnimationFrame(processChunk);
    } else {
      state._batchResults = results;
      document.getElementById('batchProgressBar').style.width = '100%';
      document.getElementById('batchProgressText').textContent = '预测完成！';
      renderBatchTable(results, header);
      document.getElementById('batchResultsArea').style.display = 'block';
      document.getElementById('batchPredictBtn').disabled = false;
      document.getElementById('batchExportBtn').disabled = false;
      persistState();
    }
  }
  requestAnimationFrame(processChunk);
}

function renderBatchTable(results, csvHeader) {
  var thead = document.getElementById('batchResultsHead');
  var tbody = document.getElementById('batchResultsBody');
  var summary = document.getElementById('batchResultSummary');
  var ths = ['#'].concat(csvHeader, ['实际类型', '预测类型', '结果', '置信度', '零件总量', '一级包装推荐']);
  thead.innerHTML = '<tr>' + ths.map(function(h, idx) {
    var stickyCls = idx <= 2 ? ' sticky-col sticky-col-' + (idx + 1) : '';
    return '<th' + (stickyCls ? ' class="' + stickyCls + '"' : '') + '>' + escapeHtml(h) + '</th>';
  }).join('') + '</tr>';

  var correctCount = 0, errorCount = 0;
  var allClasses = state.model && state.model.class_names ? state.model.class_names : [];
  tbody.innerHTML = results.map(function(r, resultIdx) {
    if (r.error) errorCount++;
    var isCorrect = !r.error && r.actualClass && String(r.actualClass) === String(r.predictedClass);
    if (isCorrect) correctCount++;
    var predSelect = '<select style="border:1px solid #ddd;border-radius:4px;padding:2px 4px;font-size:12px;background:#fff;cursor:pointer;" onchange="updatePredClass(' + resultIdx + ', this.value)">';
    allClasses.forEach(function(cls) { predSelect += '<option value="' + escapeHtml(cls) + '"' + (cls === r.predictedClass ? ' selected' : '') + '>' + escapeHtml(cls) + '</option>'; });
    predSelect += '</select>';
    var pkgMethod = r.pkgMethod || null;
    var pkgBtnText = pkgMethod ? pkgMethod.name : '—';
    var pkgBtnStyle = pkgMethod ? 'background:#0052d9;color:#fff;border-color:#0052d9;' : '';
    var pkgBtn = '<button class="btn" style="height:26px;padding:0 10px;font-size:12px;cursor:pointer;' + pkgBtnStyle + '" onclick="openPkgMethodModal(' + resultIdx + ')">' + escapeHtml(pkgBtnText) + '</button>';
    var cells = [r.rowIndex].concat(csvHeader.map(function(col) { return r.rawFeatures[col] !== undefined ? escapeHtml(r.rawFeatures[col]) : ''; }),
      [escapeHtml(r.actualClass), predSelect, r.error ? '<span class="err-col">错误</span>' : (isCorrect ? '<span class="correct-col">✓</span>' : '<span class="wrong-col">✗</span>'), r.error ? '<span class="err-col">' + escapeHtml(r.error) + '</span>' : r.confidence, r.totalParts !== null && r.totalParts !== undefined ? r.totalParts.toFixed(0) : '—', pkgBtn]);
    return '<tr>' + cells.map(function(v, idx) {
      var cls = '';
      if (idx <= 2) cls = 'sticky-col sticky-col-' + (idx + 1);
      else if (idx === cells.length - 5) cls = 'pred-col';
      return '<td' + (cls ? ' class="' + cls + '"' : '') + '>' + v + '</td>';
    }).join('') + '</tr>';
  }).join('');

  var total = results.length;
  var valid = Math.max(0, total - errorCount);
  var accuracy = valid > 0 ? ((correctCount / valid) * 100).toFixed(1) : 0;
  summary.innerHTML = '共 <strong>' + total + '</strong> 条 | <strong style="color:var(--success)">正确 ' + correctCount + '</strong> | <strong style="color:var(--danger)">错误 ' + (valid - correctCount) + '</strong>' + (errorCount > 0 ? ' | <strong style="color:var(--danger)">解析错误 ' + errorCount + '</strong>' : '') + ' | 准确率 <strong>' + accuracy + '%</strong>';
}

function updatePredClass(resultIdx, newClass) {
  if (state._batchResults && state._batchResults[resultIdx]) {
    state._batchResults[resultIdx].predictedClass = newClass;
    persistState();
  }
}

function clearBatchResults() {
  state._batchCsvData = null;
  state._batchResults = null;
  document.getElementById('batchResultsArea').style.display = 'none';
  document.getElementById('batchProgressArea').style.display = 'none';
  document.getElementById('batchFileName').textContent = '';
  document.getElementById('batchCsvInput').value = '';
  document.getElementById('batchSizeInput').value = '';
  document.getElementById('batchPredictBtn').disabled = true;
  document.getElementById('batchExportBtn').disabled = true;
  document.getElementById('batchProgressBar').style.width = '0%';
  document.getElementById('batchResultSummary').innerHTML = '';
  persistState();
}

function exportBatchResults() {
  if (!state._batchResults || !state._batchCsvData) return;
  var header = state._batchCsvData.header;
  var lines = [['序号'].concat(header, ['实际包装类型', '预测包装类型', '是否正确', '置信度', '零件总量', '包装方式', '错误信息'])];
  state._batchResults.forEach(function(r) {
    var rawVals = header.map(function(col) { return r.rawFeatures[col] !== undefined ? r.rawFeatures[col] : ''; });
    var isCorrect = !r.error && r.actualClass && String(r.actualClass) === String(r.predictedClass);
    lines.push([r.rowIndex, ...rawVals, r.actualClass, r.predictedClass, r.error ? '错误' : (isCorrect ? '✓' : '✗'), r.confidence, r.totalParts !== null && r.totalParts !== undefined ? r.totalParts.toFixed(0) : '', r.pkgMethod ? r.pkgMethod.name : '', r.error || '']);
  });
  var csv = lines.map(function(row) { return row.map(function(v) { return '"' + String(v).replace(/"/g, '""') + '"'; }).join(','); }).join('\n');
  var BOM = '\uFEFF';
  var blob = new Blob([BOM + csv], { type: 'text/csv;charset=utf-8;' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'batch_predictions_' + new Date().toISOString().slice(0, 10) + '.csv';
  a.click();
  URL.revokeObjectURL(url);
}
