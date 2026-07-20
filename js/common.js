const AppState = (function () {
  const KEY = 'pkg_state_v1';
  const defaults = {
    model: null,
    featureMeta: [],
    featureNodes: [],
    part: null,
    pkgMethods: [],
    selectedPkgMethod: '',
    scheme: null,
    predicted: null,
    selectedContainer: '20gp',
    loadChars: [],
    imageData: null,
    manualPartsList: [],
    _batchCsvData: null,
    _batchResults: null,
  };

  function load() {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        return { ...defaults, ...parsed };
      }
    } catch (e) {
      console.error("load state error", e);
    }
    return { ...defaults };
  }

  function save(state) {
    try {
      localStorage.setItem(KEY, JSON.stringify(state));
    } catch (e) {
      console.error("save state error", e);
    }
  }

  function clear() {
    localStorage.removeItem(KEY);
  }

  return { load, save, clear, KEY };
})();

const state = AppState.load();

const CONTAINER_SPECS = {
  '20gp': { l: 5900, w: 2350, h: 2390, maxWeight: 22100, label: "20'GP" },
  '40gp': { l: 12030, w: 2350, h: 2390, maxWeight: 27600, label: "40'GP" },
  '40hc': { l: 12030, w: 2350, h: 2690, maxWeight: 27600, label: "40'HC" },
};

const BOX_SPECS = [
  { id: 'A', name: '固定包箱A', l: 1000, w: 800, h: 500 },
  { id: 'B', name: '固定包箱B', l: 800, w: 600, h: 400 },
  { id: 'C', name: '固定包箱C', l: 600, w: 400, h: 300 },
  { id: 'D', name: '固定包箱D', l: 1200, w: 1000, h: 600 },
];

const MATERIAL_PRICES = {
  STD:  { name: 'STD',  surfacePricePerM2: 40,  laborCost: 10 },
  carton: { name: '纸箱', surfacePricePerM2: 45,  laborCost: 12 },
  tdg:   { name: '天地盖纸箱', surfacePricePerM2: 55, laborCost: 15 },
  wood:  { name: '木箱', surfacePricePerM2: 120, laborCost: 28 },
  iron:  { name: '铁架/轻钢', surfacePricePerM2: 180, laborCost: 38 },
};

const MODAL_BOX_SIZES = {
  STD: (function() {
    var sizes = [];
    [360].forEach(function(l) {
      [280,560].forEach(function(w) {
        [200,280].forEach(function(h) { sizes.push({l:l,w:w,h:h}); });
      });
    });
    return sizes;
  })(),
  carton: (function() {
    var sizes = [];
    [360,560,720,1120,1480].forEach(function(l) {
      [280,560].forEach(function(w) {
        [140,180,200,280,300,400,500].forEach(function(h) { sizes.push({l:l,w:w,h:h}); });
      });
    });
    return sizes;
  })(),
  tdg: (function() {
    var sizes = [];
    [980,1180,1320,1480,1700,1960].forEach(function(l) {
      [1140,2280].forEach(function(w) {
        [460,625,700,760,835,900,955,1000,1050,1100,1120,1150,1200].forEach(function(h) { sizes.push({l:l,w:w,h:h}); });
      });
    });
    return sizes;
  })(),
  wood: (function() {
    var sizes = [];
    [980,1620,1700].forEach(function(l) {
      [1140,2280].forEach(function(w) {
        [400,550,1200].forEach(function(h) { sizes.push({l:l,w:w,h:h}); });
      });
    });
    return sizes;
  })(),
  iron: (function() {
    var sizes = [];
    [980,1180,1320,1480,1700,2360,2960,3660].forEach(function(l) {
      [1140,2280].forEach(function(w) {
        [300,615,625,700,800,900,1000,1100,1200,1250,1400,1480,1600,1700,1800].forEach(function(h) { sizes.push({l:l,w:w,h:h}); });
      });
    });
    return sizes;
  })(),
};

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function roundNum(num) {
  const n = Number(num);
  if (!Number.isFinite(n)) return '—';
  return Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(2);
}

function formatRange(minVal, maxVal) {
  const minNum = Number(minVal);
  const maxNum = Number(maxVal);
  if (!Number.isFinite(minNum) || !Number.isFinite(maxNum)) return '—';
  return `${roundNum(minNum)} ~ ${roundNum(maxNum)}`;
}

function estimatePkgWeight(pkgL, pkgW, pkgH, part) {
  const surfaceArea = (pkgL * pkgW + pkgL * pkgH + pkgW * pkgH) * 2 / 1e6;
  return surfaceArea * 5;
}

function persistState() {
  AppState.save(state);
}

window.addEventListener('beforeunload', persistState);
