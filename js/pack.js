function sortPkgMethodsByConfidence(result) {
  const predClass = String(result.predicted_class || '');
  const probMap = {};
  (result.all_probabilities || []).forEach(p => { probMap[String(p.class)] = p.probability || 0; });
  state.pkgMethods.forEach(m => {
    const materialName = MATERIAL_PRICES[m.material].name;
    let matchScore = 0;
    if (predClass.includes(materialName)) matchScore = 2;
    else if (predClass.includes('STD')) matchScore = 1;
    m.matchScore = matchScore;
    m.predConfidence = probMap[predClass] || 0;
  });
  state.pkgMethods.sort((a, b) => {
    if (b.matchScore !== a.matchScore) return b.matchScore - a.matchScore;
    if (b.predConfidence !== a.predConfidence) return b.predConfidence - a.predConfidence;
    return b.containerFillRate - a.containerFillRate;
  });
}

function checkPkgConstraints(pack) {
  const pL = pack.l + 2 * pack.buffer + 2 * (pack.tolerance || 0);
  const pW = pack.w + 2 * pack.buffer + 2 * (pack.tolerance || 0);
  const pH = pack.h + 2 * pack.buffer + 2 * (pack.tolerance || 0);
  const ctnL = pack.ctnSpec.l;
  const ctnW = pack.ctnSpec.w;
  const ctnH = pack.ctnSpec.h;
  const ctnMaxWeight = pack.ctnSpec.maxWeight;
  const maxLoad = pack.maxLoad || 0;

  var smallestBoxL = 600, smallestBoxW = 400, smallestBoxH = 300;
  if (pL > smallestBoxL || pW > smallestBoxW || pH > smallestBoxH) {
    return '零件尺寸（' + pack.l + '×' + pack.w + '×' + pack.h + ' mm，含缓冲' + pack.buffer + 'mm+公差' + (pack.tolerance||0) + 'mm）超出最小包箱规格（' + smallestBoxL + '×' + smallestBoxW + '×' + smallestBoxH + ' mm）。\n请减小零件尺寸或增加缓冲厚度。';
  }
  if (pL > ctnL || pW > ctnW || pH > ctnH) {
    return '零件尺寸（' + pL + '×' + pW + '×' + pH + ' mm，含缓冲+公差）超出了集装箱内径（' + ctnL + '×' + ctnW + '×' + ctnH + ' mm）。\n请更换更大集装箱或减小零件尺寸。';
  }
  var approxPkgWeight = pack.weight * 1;
  var pkgMatWeightEst = 5;
  var singleBoxWeight = approxPkgWeight + pkgMatWeightEst;
  if (singleBoxWeight > ctnMaxWeight) {
    return '零件重量（' + pack.weight + ' kg）过大，超过了集装箱最大载重（' + ctnMaxWeight + ' kg）。\n请减小单件重量或拆分零件。';
  }
  if (maxLoad > 0 && singleBoxWeight > maxLoad) {
    return '零件单箱重量（' + singleBoxWeight.toFixed(1) + ' kg）超过了零件最大承重限制（' + maxLoad + ' kg）。\n请减小零件重量或调整最大承重参数。';
  }
  return null;
}

function computePkgMethods(part) {
  const methods = [];
  const ctnL = part.ctnSpec.l;
  const ctnW = part.ctnSpec.w;
  const ctnH = part.ctnSpec.h;
  const ctnMaxWeight = part.ctnSpec.maxWeight;
  const maxLoad = part.maxLoad || 0;
  const loadChars = part.loadChars || [];
  const isFragile = loadChars.includes('易碎');

  const pL = part.l + 2 * part.buffer + 2 * (part.tolerance || 0);
  const pW = part.w + 2 * part.buffer + 2 * (part.tolerance || 0);
  const pH = part.h + 2 * part.buffer + 2 * (part.tolerance || 0);

  function calcSNP(boxL, boxW, boxH) {
    const hasNoFlip = loadChars.includes('不可倒置');
    let orientations;
    if (hasNoFlip) {
      orientations = [
        { d: [pL, pW, pH], label: '平放' },
        { d: [pW, pL, pH], label: '侧放（Z轴）' },
        { d: [pH, pW, pL], label: 'X翻转' },
        { d: [pH, pL, pW], label: 'X翻转+旋转' },
        { d: [pL, pH, pW], label: 'Y翻转' },
        { d: [pW, pH, pL], label: 'Y翻转+旋转' },
      ];
    } else {
      const dims = [pL, pW, pH];
      const oLabels = [
        '平放', 'XY换向', 'XZ换向', 'XYZ换向',
        'XZ换向', 'YXZ换向', 'YZ换向', 'Z轴180',
        'XY换向', 'YX换向', 'YZ换向', 'Z轴180',
        'YZ换向', 'ZYX换向', 'ZX换向', 'Z轴180',
        'ZX换向', 'YZX换向', 'YX换向', 'Z轴180',
        'XYZ换向', 'ZXY换向', 'YZW换向', 'Z轴180',
      ];
      orientations = [];
      const perms = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]];
      for (let pi = 0; pi < perms.length; pi++) {
        const [ai, bi, ci] = perms[pi];
        orientations.push({ d: [dims[ai], dims[bi], dims[ci]], label: oLabels[pi * 4] });
        orientations.push({ d: [dims[bi], dims[ai], dims[ci]], label: oLabels[pi * 4 + 1] });
        orientations.push({ d: [dims[ci], dims[bi], dims[ai]], label: oLabels[pi * 4 + 2] });
        orientations.push({ d: [dims[bi], dims[ci], dims[ai]], label: oLabels[pi * 4 + 3] });
      }
    }

    const viable = orientations.filter(o => o.d[0] <= boxL && o.d[1] <= boxW && o.d[2] <= boxH);
    if (viable.length === 0) return null;

    let bestSnp = 0, bestOrient = viable[0], bestNx = 1, bestNy = 1, bestNz = 1;
    for (const o of viable) {
      const [dX, dY, dZ] = o.d;
      const nx = Math.max(1, Math.floor(boxL / dX));
      const ny = Math.max(1, Math.floor(boxW / dY));
      const nz = Math.max(1, Math.floor(boxH / dZ));
      const gsnp = nx * ny * nz;
      if (gsnp > bestSnp) {
        bestSnp = gsnp;
        bestOrient = o;
        bestNx = nx; bestNy = ny; bestNz = nz;
      }
    }
    return { snp: bestSnp, orientation: bestOrient.label, nx: bestNx, ny: bestNy, nz: bestNz };
  }

  function evalMethod(cfg) {
    const { boxId, boxName, boxL, boxW, boxH, material, snp, orientation, nx, ny, nz } = cfg;
    const pkgL = boxL, pkgW = boxW, pkgH = boxH;
    if (pkgL > ctnL || pkgW > ctnW || pkgH > ctnH) return null;
    const pkgMatWeight = estimatePkgWeight(pkgL, pkgW, pkgH, part);
    const perBoxWeight = part.weight * snp + pkgMatWeight;
    if (maxLoad > 0 && perBoxWeight > maxLoad) return null;
    if (perBoxWeight > ctnMaxWeight) return null;
    if (isFragile && snp > 1 && orientation !== '平放' && orientation !== '侧放（Z轴）') return null;

    const ctnNX = Math.floor(ctnL / pkgL);
    const ctnNY = Math.floor(ctnW / pkgW);
    const ctnNZ = Math.floor(ctnH / pkgH);
    const byVolume = ctnNX * ctnNY * ctnNZ;
    const byWeight = perBoxWeight > 0 ? Math.floor(ctnMaxWeight / perBoxWeight) : byVolume;
    const boxesPerCtn = Math.max(1, Math.min(byVolume, byWeight));
    const totalParts = boxesPerCtn * snp;
    const usedVol = boxesPerCtn * pkgL * pkgW * pkgH;
    const containerVol = ctnL * ctnW * ctnH;
    const containerFillRate = usedVol / containerVol;
    const partVol = pL * pW * pH;
    const boxVol = pkgL * pkgW * pkgH;
    const fillRate = (snp * partVol) / boxVol;
    const compositeScore = fillRate * totalParts;

    const matInfo = MATERIAL_PRICES[material];
    const surfaceArea = (pkgL * pkgW + pkgL * pkgH + pkgW * pkgH) * 2 / 1e6;
    const materialCost = surfaceArea * matInfo.surfacePricePerM2;
    const laborCost = matInfo.laborCost;
    const antiRustCost = part.antiRust ? 25 : 0;
    const consumableCost = part.antiRust ? (15 + snp * 3) : snp * 2;
    const singlePkgCost = materialCost + laborCost + antiRustCost + consumableCost;

    const orientationLabels = {
      '平放': '平放', '侧放（Z轴）': '侧放（Z轴）', 'X翻转': 'X翻转', 'X翻转+旋转': 'X翻转+旋转',
      'Y翻转': 'Y翻转', 'Y翻转+旋转': 'Y翻转+旋转', 'XY换向': 'XY换向', 'XZ换向': 'XZ换向',
      'XYZ换向': 'XYZ换向', 'YXZ换向': 'YXZ换向', 'YZ换向': 'YZ换向', 'Z轴180': 'Z轴180',
      'ZYX换向': 'ZYX换向', 'ZX换向': 'ZX换向', 'YZX换向': 'YZX换向', 'ZXY换向': 'ZXY换向',
      'flat': '平放', 'rot-x': '立放（绕X轴）', 'rot-y': '旋转（绕Y轴）', 'rot-z': '侧放（绕Z轴）',
      'flip-x': '翻转（绕X轴）', 'flip-y': '翻转（绕Y轴）', 'side': '侧放', 'stand': '立放', 'stack': '叠放'
    };

    return {
      id: `${material}-${boxId}`,
      name: `${matInfo.name} ${boxName}`,
      desc: `${orientationLabels[orientation] || orientation} | 每箱 ${snp} 件 | 包箱 ${pkgL}×${pkgW}×${pkgH}mm`,
      snp, orientation, nx, ny, nz, pkgL, pkgW, pkgH, material, boxId,
      perBoxWeight, boxesPerCtn, totalParts, fillRate, containerFillRate, compositeScore, singlePkgCost,
    };
  }

  for (const box of BOX_SPECS) {
    for (const mat of ['wood', 'carton', 'iron']) {
      const snpResult = calcSNP(box.l, box.w, box.h);
      if (!snpResult || snpResult.snp < 1) continue;
      const m = evalMethod({
        boxId: box.id, boxName: box.name, boxL: box.l, boxW: box.w, boxH: box.h,
        material: mat, snp: snpResult.snp, orientation: snpResult.orientation,
        nx: snpResult.nx || 1, ny: snpResult.ny || 1, nz: snpResult.nz || 1,
      });
      if (m) methods.push(m);
    }
  }
  methods.sort((a, b) => b.compositeScore - a.compositeScore);
  return methods;
}
