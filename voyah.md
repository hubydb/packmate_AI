# voyah.md

This file provides guidance to Voyah Code (claude.ai/code) when working with code in this repository.

## Project Overview

**零部件包装测算系统** — A single-page web application for calculating optimal packaging solutions for parts/components. The system guides users through a 4-step workflow: input part parameters → recommend packaging method → confirm scheme → calculate costs.

- **Technology**: Pure HTML/CSS/JavaScript (no build step, no framework)
- **Entry point**: `index.html` — open directly in a browser
- **State**: Global JavaScript `state` object, persisted to `localStorage` under key `pkg_projects`
- **main.py**: Non-functional placeholder from JetBrains template — ignore it

## Running

- **train_server.py** (Flask 后端): Start with `python train_server.py`, serves LightGBM_damo.html with training API and model export
- **LightGBM_damo.html**: Works in two modes:
  - 联网模式：训练完成后可下载独立版 HTML（含内嵌模型 JSON）
  - 离线模式：独立版打开即用，纯 JS 预测引擎不依赖网络

### LightGBM 离线预测

train_server.py exports model via `/api/model` (JSON with tree structure) and `/api/standalone-html` (full self-contained HTML). The JS predictor `LGBPredictor` (index.html ~line 200) traverses LightGBM tree JSON using the same split/leaf logic as the Python booster. Binary classification uses sigmoid, multiclass uses softmax over GBDT leaf values.

## Code Architecture

### State Management
All application state lives in a global `state` object (line ~1044):
```js
const state = {
  step, imageData, selectedContainer, selectedPkgMethod,
  pkgMethods, scheme, cost, loadedParts, layoutPart, layoutMethod,
  layoutAlpha, layoutBeta, previewAlpha, previewBeta,
  currentPart
};
```

### Core Computation Functions
- `computePkgMethods(part)` (line ~1132): Iterates over all 6 combinations (2 box specs × 3 materials), calculates SNP for each by fitting part+buffer into fixed box with 3 orientation modes, filters by weight/fit constraints, returns sorted by fill rate.
- `calcSNP(boxL, boxW, boxH)` (line ~1156): Calculates how many parts fit in a fixed box using flat/stand/side orientations.
- `estimatePkgWeight(pkgL, pkgW, pkgH, part)` (line ~1416): Estimates packaging box weight from surface area.
- `estimateSinglePkgCost(method, part)` (line ~1433): Calculates per-box cost (materials + labor + anti-rust + consumables).

### 3D Rendering
Canvas-based 3D wireframe rendering with custom perspective projection:
- `drawLayout(...)` (line ~1444): Container loading visualization with interactive rotation via mouse/touch drag
- `drawPkgPreview(...)` (line ~1810): Package box 3D preview showing outer box, inner part, and cushion gap
- Both use Y-axis then X-axis rotation matrices; depth-sorted painter's algorithm for face ordering

### Constants
- `CONTAINER_SPECS` (line ~1032): 20GP/40GP/40HC dimensions and max weights
- `BOX_SPECS` (line ~1037): Fixed box internal dimensions (规格A: 1000×800×500, 规格B: 800×600×400)
- `MATERIAL_PRICES` (line ~1044): Pricing for wood/carton/iron materials

### Key Business Logic
- Package box specs are **fixed** (not calculated from part dimensions):
  - 规格A: 1000mm × 800mm × 500mm (internal dimensions)
  - 规格B: 800mm × 600mm × 400mm (internal dimensions)
- Materials: 木箱 (wood), 纸箱 (carton), 铁箱 (iron)
- SNP (Standard Packing Quantity) is calculated by fitting part+buffer into the fixed box using 3 orientation modes: flat (平放), stand (立放), side (侧放)
- Parts with `不可倒置` are excluded from stand orientation
- Parts with `易碎` are excluded from orientations that stack parts
- Buffer thickness adds to part dimensions when calculating fit inside the fixed box
- Fill rate = (boxesPerContainer × boxVolume) / containerVolume, capped by weight constraints

## Workflow Steps

1. **Step 1** — Part info entry: dimensions, weight, load characteristics, container type, material
2. **Step 2** — Package method recommendation: AI-ranked options displayed as cards with fill rate
3. **Step 3** — Scheme confirmation: table of all calculated values + interactive 3D container loading visualization
4. **Step 4** — Cost calculation: material cost breakdown, per-part cost, shipping estimate, batch total

## Behavioral Guidelines

- Think before coding: state assumptions, surface tradeoffs, ask if unclear
- Simplicity first: minimum code that solves the problem, no speculative abstractions
- Surgical changes: touch only what must change, don't refactor adjacent code
- Match existing style: the codebase uses Chinese UI text, camelCase JS variables, and 2-space indentation in HTML attributes