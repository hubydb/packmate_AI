async function loadModelFromApi() {
  try {
    const resp = await fetch('/api/model');
    const data = await resp.json();
    if (data && data.success && data.data) {
      state.model = data.data;
      return true;
    }
  } catch (e) {
    console.error('加载模型失败:', e);
  }
  return false;
}

function predictLGB(model, features) {
  const numClasses = model.num_classes || 2;
  const trees = model.trees && model.trees.tree_info ? model.trees.tree_info : [];
  const isMulti = numClasses > 2;
  const contribs = new Array(numClasses).fill(0.0);
  for (const treeData of trees) {
    if (!treeData.tree_structure) continue;
    const targetClass = isMulti ? (treeData.tree_index % numClasses) : 0;
    if (targetClass < 0 || targetClass >= numClasses) continue;
    contribs[targetClass] += getLeafValue(treeData.tree_structure, features);
  }
  let probs;
  if (isMulti) {
    const maxC = Math.max(...contribs);
    const exps = contribs.map(c => Math.exp(c - maxC));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    probs = exps.map(e => e / sumExp);
  } else {
    const sig = 1 / (1 + Math.exp(-contribs[0]));
    probs = [1 - sig, sig];
  }
  const predIdx = probs.indexOf(Math.max(...probs));
  const className = model.class_names && model.class_names[predIdx] !== undefined ? model.class_names[predIdx] : String(predIdx);
  return {
    predicted_class: className,
    confidence: probs[predIdx],
    all_probabilities: (model.class_names || []).map((c, i) => ({ class: c, probability: probs[i] })),
  };
}

function getLeafValue(node, features) {
  if (!node) return 0.0;
  if (node.leaf_value !== undefined) return node.leaf_value;
  const featIdx = node.split_feature;
  const value = (featIdx !== undefined && featIdx !== null) ? features[featIdx] : undefined;
  const missing = value === undefined || Number.isNaN(value);
  if (missing) {
    return getLeafValue(node.default_left ? node.left_child : node.right_child, features);
  }
  const goLeft = (value <= node.threshold);
  return getLeafValue(goLeft ? node.left_child : node.right_child, features);
}
