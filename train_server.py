"""
LightGBM 训练测试服务
启动方式: python train_server.py
然后在浏览器打开 http://127.0.0.1:5000/LightGBM_damo.html
"""

import os
import io
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, Response
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

app = Flask(__name__)
DATA_FILE = 'train_data.csv'

# ── 全局模型训练结果 ──
train_result = None
_trained_model = None
_trained_le = None
_trained_params = None
_feature_names = None
_cached_standalone_html = None  # 训练后缓存生成的独立预测 HTML


def train_model(df):
    """训练 LightGBM 模型，df 由调用方传入"""
    global train_result, _trained_model, _trained_le, _trained_params, _feature_names
    feature_names = df.drop('label', axis=1).columns.tolist()
    X = df.drop('label', axis=1).values
    y_raw = df['label'].values

    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # LightGBM 参数（分类）
    params = {
        'objective': 'multiclass' if num_classes > 2 else 'binary',
        'num_class': num_classes,
        'metric': 'multi_logloss' if num_classes > 2 else 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
    )

    # 保存模型供预测使用
    _trained_model = model
    _trained_le = le
    _trained_params = params
    _feature_names = feature_names

    # 预测
    y_pred_proba = model.predict(X_test)
    if num_classes > 2:
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # 准确率
    accuracy = float(np.mean(y_pred == y_test))

    # 特征重要性
    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

    # 特征统计（用于随机填充）
    feat_stats = []
    for i, name in enumerate(feature_names):
        col = X[:, i].astype(float)
        feat_stats.append({
            'name': name,
            'min': float(np.min(col)),
            'max': float(np.max(col)),
            'mean': float(np.mean(col)),
            'std': float(np.std(col)) if np.std(col) > 1e-9 else float((np.max(col) - np.min(col)) / 4),
        })

    train_result = {
        'num_classes': int(num_classes),
        'class_names': [str(c) for c in le.classes_],
        'num_features': int(X.shape[1]),
        'feature_names': feature_names,
        'num_train': int(X_train.shape[0]),
        'num_test': int(X_test.shape[0]),
        'accuracy': round(accuracy, 4),
        'accuracy_pct': f'{accuracy * 100:.2f}%',
        'num_trees': model.num_trees(),
        'model_ready': True,
    }
    # 将 numpy float 转换为 Python float
    train_result['feature_importance'] = [(name, float(score)) for name, score in feat_imp[:15]]
    train_result['per_class_accuracy'] = get_per_class_accuracy(y_test, y_pred, le.classes_, num_classes)
    train_result['feature_stats'] = feat_stats

    return train_result


def get_per_class_accuracy(y_true, y_pred, classes, num_classes):
    """计算每个类别的准确率"""
    results = []
    for i, c in enumerate(classes):
        mask = y_true == i
        if mask.sum() > 0:
            cls_acc = np.mean((y_pred == i)[mask])
            results.append({'class': str(c), 'accuracy': round(float(cls_acc), 4), 'count': int(mask.sum())})
    return results


# ── API 路由（优先注册，避免被其他路由拦截）─────────────────────────────

@app.route('/api/train', methods=['POST'])
def api_train():
    """触发训练，可上传 CSV 文件（multipart/form-data，字段名为 file），无文件则使用默认数据集"""
    try:
        # 支持文件上传
        file_obj = request.files.get('file')
        if file_obj:
            import io as _io
            df = pd.read_csv(_io.StringIO(file_obj.read().decode('utf-8', errors='replace')))
        else:
            df = pd.read_csv(DATA_FILE)

        if 'label' not in df.columns:
            return jsonify({'success': False, 'error': 'CSV 必须包含 label 列'}), 400

        result = train_model(df)
        # 缓存独立预测版 HTML，供 /predict 路由使用
        global _cached_standalone_html
        _cached_standalone_html = _build_standalone_predict_html()
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/result', methods=['GET'])
def api_result():
    """获取上次训练结果"""
    if train_result is None:
        return jsonify({'success': False, 'error': '尚未训练，请先调用 /api/train'}), 400
    return jsonify({'success': True, 'data': train_result})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """使用训练好的模型进行预测（服务端模式）"""
    if _trained_model is None:
        return jsonify({'success': False, 'error': '模型尚未训练，请先调用 /api/train'}), 400

    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'success': False, 'error': '请提供 features 数组'}), 400

    features = data['features']
    if len(features) != len(_feature_names):
        return jsonify({'success': False, 'error': f'特征数量不匹配，需要 {len(_feature_names)} 个特征'}), 400

    try:
        X = np.array(features, dtype=float).reshape(1, -1)
        proba = _trained_model.predict(X)[0]

        if train_result['num_classes'] > 2:
            pred_class_idx = int(np.argmax(proba))
            pred_proba = [float(p) for p in proba]
        else:
            pred_class_idx = int(proba[0] > 0.5)
            pred_proba = [float(1 - proba[0]), float(proba[0])]

        pred_label = str(_trained_le.classes_[pred_class_idx])
        confidence = float(proba[pred_class_idx])

        return jsonify({
            'success': True,
            'data': {
                'predicted_class': pred_label,
                'predicted_class_idx': pred_class_idx,
                'confidence': round(confidence * 100, 2),
                'probability': round(confidence, 4),
                'all_probabilities': [
                    {'class': str(c), 'probability': round(float(p), 4)}
                    for c, p in zip(_trained_le.classes_, pred_proba)
                ],
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/model', methods=['GET'])
def api_model():
    """导出模型 JSON（含树结构），供前端离线预测"""
    if _trained_model is None:
        return jsonify({'success': False, 'error': '模型尚未训练'}), 400

    try:
        model_json = _trained_model.dump_model()
        export = {
            'num_classes': int(train_result['num_classes']),
            'class_names': [str(c) for c in _trained_le.classes_],
            'feature_names': _feature_names,
            'num_features': len(_feature_names),
            'objective': _trained_params['objective'],
            'trees': model_json,
            'feature_stats': train_result.get('feature_stats', []),
        }
        return jsonify({'success': True, 'data': export})
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/standalone-html', methods=['GET'])
def api_standalone_html():
    """生成可独立运行的完整 HTML（内嵌模型）"""
    import time
    if _trained_model is None:
        return jsonify({'success': False, 'error': '模型尚未训练'}), 400

    try:
        model_json = _trained_model.dump_model()
        export = {
            'num_classes': int(train_result['num_classes']),
            'class_names': [str(c) for c in _trained_le.classes_],
            'feature_names': _feature_names,
            'num_features': len(_feature_names),
            'objective': _trained_params['objective'],
            'trees': model_json,
            'feature_stats': train_result.get('feature_stats', []),
        }
        model_js = 'const LGBMODEL = ' + json.dumps(export, ensure_ascii=False) + ';'

        # 读取 HTML 模板，注入模型
        import pathlib
        html_path = pathlib.Path(__file__).parent / 'LightGBM_damo.html'
        template = html_path.read_text(encoding='utf-8')

        # 在 </body> 前注入模型
        standalone = template.replace(
            '</body>',
            f'<script>\n{model_js}\nwindow.LGB_STANDALONE=true;\n</script>\n</body>'
        )
        standalone = standalone.replace('<title>LightGBM 训练测试平台</title>',
            '<title>LightGBM 独立预测版</title>')

        resp = Response(standalone, mimetype='text/html; charset=utf-8')
        resp.headers['Content-Disposition'] = 'attachment; filename=LightGBM_standalone.html'
        return resp
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


def _build_standalone_predict_html():
    """构建并返回独立预测版 HTML 字符串（供 /predict 和下载共用）"""
    raw = _trained_model.dump_model()

    def _strip_tree(node):
        if node is None:
            return None
        if isinstance(node, dict):
            if 'leaf_value' in node:
                return {'leaf_value': node['leaf_value']}
            result = {
                'split_feature': node.get('split_feature'),
                'threshold': node.get('threshold'),
                'decision_type': node.get('decision_type', '<='),
                'default_left': node.get('default_left', True),
            }
            lc = node.get('left_child')
            rc = node.get('right_child')
            if lc is not None:
                result['left_child'] = _strip_tree(lc)
            if rc is not None:
                result['right_child'] = _strip_tree(rc)
            return result
        return node

    tree_info = []
    for t in (raw.get('tree_info') or []):
        ts = _strip_tree(t.get('tree_structure'))
        if ts:
            tree_info.append({
                'tree_index': t.get('tree_index', 0),
                'num_leaves': t.get('num_leaves', 1),
                'tree_structure': ts,
            })

    export = {
        'num_classes': int(train_result['num_classes']),
        'class_names': [str(c) for c in _trained_le.classes_],
        'feature_names': _feature_names,
        'num_features': len(_feature_names),
        'trees': {'tree_info': tree_info},
        'accuracy_pct': train_result['accuracy_pct'],
        'num_trees': train_result['num_trees'],
        'feature_stats': train_result.get('feature_stats', []),
    }
    model_js = json.dumps(export, ensure_ascii=False)
    model_js = model_js.replace('</script', '<\\/script')
    return _build_predict_only_html(model_js, _feature_names, train_result)


@app.route('/api/standalone-predict-only', methods=['GET'])
def api_standalone_predict_only():
    """生成只含预测模块的独立 HTML（最小化体积），供下载"""
    if _trained_model is None:
        return jsonify({'success': False, 'error': '模型尚未训练'}), 400
    try:
        html_content = _build_standalone_predict_html()
        resp = Response(html_content, mimetype='text/html; charset=utf-8')
        resp.headers['Content-Disposition'] = 'attachment; filename=LightGBM_predict.html'
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/predict')
def page_predict():
    """独立预测页面 URL（训练后可用，模型已嵌入，可完全离线运行）"""
    if _cached_standalone_html is None:
        return '<h2>请先在 <a href="/">训练页面</a> 完成模型训练</h2>', 404
    resp = Response(_cached_standalone_html, mimetype='text/html; charset=utf-8')
    resp.headers['Cache-Control'] = 'no-cache'
    return resp



def _build_predict_only_html(model_js, feature_names, train_result):
    """构建纯预测版 HTML"""
    # 生成特征输入行
    feat_rows = []
    for i, name in enumerate(feature_names):
        feat_rows.append(
            '      <div class="feat-input-group">'
            '<label for="feat{}">{}</label>'
            '<input type="number" id="feat{}" placeholder="选填" step="any" />'
            '</div>'.format(i, name, i)
        )
    feat_grid_html = '\n'.join(feat_rows)

    # 类别准确率标签
    class_cards = []
    for c in (train_result.get('per_class_accuracy') or []):
        class_cards.append(
            '<div class="class-chip">'
            '<span class="class-chip-name">\u7c7b\u522b {}</span>'
            '<span class="class-chip-acc">{:.1f}%</span>'
            '</div>'.format(c['class'], c['accuracy'] * 100)
        )
    class_chips_html = ('<div class="class-chips">' + ''.join(class_cards) + '</div>') if class_cards else ''

    # 直接替换占位符，避免 string.Template 处理大文本时破坏转义序列
    html = (_STANDALONE_HTML_TEMPLATE
        .replace('$MODEL', model_js)
        .replace('$FEAT', feat_grid_html)
        .replace('$CHIPS', class_chips_html))
    return html


_STANDALONE_HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>LightGBM 预测工具</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:#1c2435;background:#f3f5f7;line-height:1.6;min-height:100vh}
.header{background:#fff;border-bottom:1px solid #e8ecf0;padding:14px 28px;display:flex;align-items:center;gap:12px}
.header h1{font-size:17px;font-weight:700;color:#0052d9}
.header .subtitle{font-size:12px;color:#8a94a6}
.header .model-info{margin-left:auto;font-size:12px;color:#8a94a6}
.main{max-width:860px;margin:20px auto;padding:0 20px}
.card{background:#fff;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.08);padding:20px 24px;margin-bottom:16px}
.card-title{font-size:15px;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:6px}
.alert{padding:10px 14px;border-radius:6px;font-size:13px;margin-bottom:14px;background:#e8f0fe;color:#0052d9}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:5px;padding:0 18px;height:36px;border:1px solid #c0c6d1;border-radius:4px;font-size:14px;cursor:pointer;background:#fff;color:#1c2435}
.btn:hover{border-color:#8a94a6}
.btn-primary{background:#0052d9;color:#fff;border-color:#0052d9}
.btn-primary:hover{background:#0041b3}
.btn-lg{height:42px;padding:0 28px;font-size:15px}
.btn:disabled{opacity:0.5;cursor:not-allowed}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid #c0c6d1;border-top-color:#0052d9;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle}
@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
.feat-inputs-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin:14px 0}
.feat-input-group{display:flex;flex-direction:column;gap:3px}
.feat-input-group label{font-size:12px;color:#4f5d73;font-weight:500}
.feat-input-group input{height:31px;padding:0 9px;border:1px solid #c0c6d1;border-radius:4px;font-size:13px;outline:none;color:#1c2435}
.feat-input-group input:focus{border-color:#0052d9;box-shadow:0 0 0 2px rgba(0,82,217,.12)}
.feat-input-group input::placeholder{color:#c0c6d1}
.pred-result{background:#e8f0fe;border-radius:8px;padding:24px;text-align:center;margin:16px 0}
.pred-result .pred-label{font-size:13px;color:#0052d9;margin-bottom:6px;font-weight:500}
.pred-result .pred-class{font-size:44px;font-weight:800;color:#0052d9;line-height:1.2}
.pred-result .pred-conf{font-size:15px;color:#4f5d73;margin-top:6px}
.proba-bars{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin-top:12px}
.proba-item{display:flex;flex-direction:column;gap:3px}
.proba-item .proba-label{font-size:12px;color:#4f5d73;display:flex;justify-content:space-between}
.proba-item .proba-track{height:7px;background:#e8ecf0;border-radius:4px;overflow:hidden}
.proba-item .proba-fill{height:100%;border-radius:4px;transition:width .5s}
.proba-item .proba-fill.highlight{background:#0052d9}
.proba-item .proba-fill.normal{background:#8a94a6}
.model-stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.model-stat{background:#f3f5f7;border-radius:6px;padding:8px 14px;font-size:12px;color:#4f5d73}
.model-stat strong{color:#0052d9;font-size:15px;font-weight:700}
.class-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.class-chip{display:inline-flex;align-items:center;gap:8px;padding:5px 12px;border:1px solid #c0c6d1;border-radius:20px;font-size:12px;color:#4f5d73}
.class-chip .class-chip-acc{font-weight:700;color:#2ba471;font-size:13px}
.alert-error{background:#fef0ef;color:#d54941;padding:10px 14px;border-radius:6px;font-size:13px}
.btn-row{display:flex;gap:10px;margin-top:8px}
.batch-table{width:100%;border-collapse:collapse;font-size:13px}
.batch-table th,.batch-table td{padding:7px 12px;border-bottom:1px solid #e8ecf0;text-align:left}
.batch-table th{background:#f3f5f7;font-weight:600;color:#4f5d73;font-size:12px}
.batch-table tr:hover td{background:#f8f9fb}
.batch-table .pred-col{font-weight:700;color:#0052d9}
.batch-table .conf-col{color:#4f5d73;font-size:12px}
</style>
</head>
<body>
<header class="header">
  <h1>LightGBM 预测工具</h1>
  <span class="subtitle">离线版 · 无需网络</span>
  <span class="model-info" id="modelInfo"></span>
<span id="initStatus" style="margin-left:8px;font-size:11px;color:#d54941"></span>
</header>
<div class="main">
  <div class="card">
    <div class="card-title">&#128736; 输入特征值</div>
    <div class="alert">&#127811; 特征可留空（NaN 自动走缺失值逻辑）。点击"随机填充"生成符合模型特征的随机示例值。</div>
    <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
      <button class="btn" id="autoFillBtn" onclick="autoFillRandom()">&#127919; 随机填充</button>
      <button class="btn" onclick="clearAll()">&#128465; 清空</button>
      <button class="btn" onclick="document.getElementById('batchFileInput').click()">&#128194; 批量导入CSV</button>
      <input type="file" id="batchFileInput" accept=".csv" style="display:none" onchange="handleBatchFile(this.files[0])" />
    </div>
    <div class="feat-inputs-grid" id="featGrid">$FEAT</div>
    <div class="btn-row">
      <button class="btn btn-primary btn-lg" id="predBtn" onclick="doPredict()">&#128372; 开始预测</button>
    </div>
  </div>

  <div class="card" id="batchCard" style="display:none;">
    <div class="card-title">&#128202; 批量预测结果 <span id="batchCount" style="font-weight:400;font-size:13px;color:#8a94a6"></span></div>
    <div style="overflow-x:auto;">
      <table id="batchTable" style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead id="batchHead"></thead>
        <tbody id="batchBody"></tbody>
      </table>
    </div>
    <div style="margin-top:12px;">
      <button class="btn" onclick="exportBatchCsv()">&#128190; 导出CSV</button>
    </div>
  </div>

  <div class="card" id="resultCard" style="display:none;">
    <div class="card-title">&#128279; 预测结果</div>
    <div class="pred-result">
      <div class="pred-label">预测类别</div>
      <div class="pred-class" id="predClass">&#8212;</div>
      <div class="pred-conf">置信度：<strong id="predConf">&#8212;</strong></div>
    </div>
    <div style="font-size:12px;color:#8a94a6;margin:12px 0 4px;">&#21516;&#31867;&#21035;&#27010;&#29575;</div>
    <div class="proba-bars" id="probaBars"></div>
  </div>

  <div class="card" id="errorCard" style="display:none;">
    <div class="alert-error" id="errorMsg"></div>
  </div>

  <div class="card">
    <div class="card-title">&#128202; 模型信息</div>
    <div class="model-stats" id="modelStats"></div>
    $CHIPS
  </div>
</div>
<script>
var _modelErr = [];
console.log('[INIT] Script starting');
try{ const MODEL_RAW = $MODEL; window.MODEL = MODEL_RAW; console.log('[INIT] MODEL_RAW OK, trees:', (MODEL_RAW.trees&&MODEL_RAW.trees.tree_info)?MODEL_RAW.trees.tree_info.length:0); }
catch(e){ _modelErr.push('MODEL parse: ' + e.message); window.MODEL = null; console.error('[INIT] MODEL_RAW FAILED:', e.message); }

(function(){
  try{
    if (!window.MODEL) { alert('\u6a21\u578b\u6570\u636e\u52a0\u8f7d\u5931\u8d25\uff1a' + _modelErr.join(', ')); return; }
    var names = window.MODEL.feature_names;
    var classes = window.MODEL.class_names;
    if (!names || !names.length) { alert('\u7279\u5f81\u540d\u4e0d\u5b58\u5728\uff1anames=' + names); return; }
    if (!classes || !classes.length) { alert('\u7c7b\u522b\u540d\u4e0d\u5b58\u5728\uff1aclasses=' + classes); return; }
    document.getElementById('modelInfo').textContent = classes.length + ' \u4e2a\u7c7b\u522b \u00b7 ' + names.length + ' \u4e2a\u7279\u5f81';
    document.getElementById('modelStats').innerHTML =
      '<div class="model-stat">\u6d4b\u8bd5\u51c6\u786e\u7387 <strong>' + (window.MODEL.accuracy_pct||'\u2014') + '</strong></div>' +
      '<div class="model-stat">\u6811\u6570\u91cf <strong>' + (window.MODEL.num_trees||'\u2014') + '</strong></div>' +
      '<div class="model-stat">\u7279\u5f81\u6570 <strong>' + names.length + '</strong></div>' +
      '<div class="model-stat">\u7c7b\u522b\u6570 <strong>' + classes.length + '</strong></div>';
    window._names = names;
    window._classes = classes;
    console.log('[INIT] OK: _names=' + names.length + ', _classes=' + classes.length);
    document.getElementById('initStatus').textContent = '\u2713 \u6a21\u578b\u5df2\u52a0\u8f7d';
    document.getElementById('initStatus').style.color = '#2ba471';
  } catch(e) {
    var errMsg = '[INIT] FAILED: ' + e.message;
    console.error(errMsg);
    document.getElementById('initStatus').textContent = '\u2717 ' + errMsg;
    alert('\u521d\u59cb\u5316\u9519\u8bef: ' + e.message);
  }
})();

// 随机填充：根据训练时统计的 feature_stats 随机生成值
function autoFillRandom() {
  try {
    if (!window._names) { alert('\u6a21\u578b\u672a\u5c31\u7eea\uff0c\u8bf7\u5148\u8fdb\u884c\u9884\u6d4b'); return; }
    var stats = window.MODEL && window.MODEL.feature_stats;
    window._names.forEach(function(name, i){
      var el = document.getElementById('feat'+i);
      if (!el) return;
      var minVal = 0, maxVal = 100;
      if (stats && stats[i]) {
        minVal = stats[i].min;
        maxVal = stats[i].max;
      }
      var range = maxVal - minVal;
      var v = range > 0 ? minVal + Math.random() * range : minVal;
      el.value = parseFloat(v.toFixed(6));
    });
  } catch(e) { alert('\u968f\u673a\u586b\u5145\u5931\u8d25: ' + e.message); }
}

// 批量导入 CSV：读取文件，对每行数据执行预测，输出结果表格
var _batchResults = [];
function handleBatchFile(file) {
  if (!file) return;
  if (!window._names || !window.MODEL) { alert('\u6a21\u578b\u672a\u5c31\u7eea\uff0c\u8bf7\u5148\u8fdb\u884c\u9884\u6d4b'); return; }
  var reader = new FileReader();
  reader.onload = function(e) {
    try {
      var text = e.target.result;
      var lines = text.trim().split('\\n');
      if (lines.length < 2) { alert('\u6570\u636e\u6587\u4ef6\u884c\u6570\u4e0d\u8db3\uff0c\u9700\u8981\u7b7e\u540d\u884c+\u6570\u636e\u884c'); return; }
      var header = lines[0].split(',').map(function(h){ return h.trim(); });
      var results = [];
      for (var li = 1; li < lines.length; li++) {
        var cols = lines[li].split(',');
        if (cols.length < window._names.length) continue;
        var features = window._names.map(function(_,i){ var v = cols[i]; return v===''||v===undefined ? NaN : parseFloat(v); });
        var r = predictLGB(window.MODEL, features);
        results.push({row: li, features: cols.slice(0, window._names.length), prediction: r.predicted_class, confidence: r.confidence, probs: r.all_probabilities});
      }
      if (!results.length) { alert('\u672a\u627e\u5230\u6709\u6548\u6570\u636e\u884c'); return; }
      _batchResults = results;
      renderBatchResults(results, header);
    } catch(err) { alert('\u6279\u91cf\u5bfc\u5165\u5931\u8d25: ' + err.message); console.error(err); }
  };
  reader.onerror = function() { alert('\u6587\u4ef6\u8bfb\u53d6\u5931\u8d25'); };
  reader.readAsText(file);
}

function renderBatchResults(results, header) {
  document.getElementById('batchCount').textContent = results.length + ' \u6761\u8bb0\u5f55';
  var headHtml = '<tr><th>#</th>';
  header.slice(0, window._names.length).forEach(function(h){ headHtml += '<th>' + h + '</th>'; });
  headHtml += '<th>\u9884\u6d4b\u7ed3\u679c</th><th>\u7f6e\u4fe1\u5ea6</th></tr>';
  document.getElementById('batchHead').innerHTML = headHtml;
  var maxP = Math.max.apply(null, results.map(function(r){ return r.confidence; }));
  var bodyHtml = '';
  results.forEach(function(r, i){
    bodyHtml += '<tr' + (i % 2 === 1 ? ' style="background:#f8f9fb"' : '') + '>' +
      '<td>' + r.row + '</td>';
    r.features.forEach(function(f){ bodyHtml += '<td>' + f + '</td>'; });
    bodyHtml += '<td class="pred-col">' + r.prediction + '</td>' +
      '<td class="conf-col">' + r.confidence + '%</td></tr>';
  });
  document.getElementById('batchBody').innerHTML = bodyHtml;
  document.getElementById('batchCard').style.display = 'block';
  document.getElementById('batchCard').scrollIntoView({behavior:'smooth'});
}

function exportBatchCsv() {
  if (!_batchResults.length) return;
  var lines2 = ['\u884c\u53f7'];
  if (window._names) window._names.forEach(function(n){ lines2.push(n); });
  lines2.push('\u9884\u6d4b\u7ed3\u679c', '\u7f6e\u4fe1\u5ea6');
  var _nl = String.fromCharCode(10);
  var csv = lines2.join(',') + _nl;
  _batchResults.forEach(function(r, i){
    var row = [(i+1)].concat(r.features).concat([r.prediction, r.confidence+'%']);
    csv += row.join(',') + _nl;
  });
  var blob = new Blob(['\ufeff' + csv], {type:'text/csv;charset=utf-8'});
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url; a.download = 'batch_predictions.csv'; a.click();
  URL.revokeObjectURL(url);
}

function clearAll() {
  try {
    console.log('[clearAll] _names:', window._names ? window._names.length : 'NULL');
    if (!window._names) { console.log('[clearAll] early return - no _names'); return; }
    window._names.forEach(function(_,i){
      var el = document.getElementById('feat'+i);
      if(el) el.value = '';
    });
    document.getElementById('resultCard').style.display='none';
    document.getElementById('errorCard').style.display='none';
  } catch(e) { console.error('clearAll:', e.message); }
}

function getFeatures() {
  if (!window._names) { throw new Error('\u6a21\u578b\u672a\u5c31\u7eea'); }
  return window._names.map(function(_,i){
    var v = document.getElementById('feat'+i).value;
    return v==='' ? NaN : parseFloat(v);
  });
}

function doPredict() {
  try {
    console.log('[doPredict] MODEL:', window.MODEL ? 'OK' : 'NULL');
    console.log('[doPredict] _names:', window._names ? 'OK' : 'NULL');
    console.log('[doPredict] _classes:', window._classes ? 'OK' : 'NULL');
    if (!window._names || !window._classes) { alert('\u6a21\u578b\u672a\u5c31\u7eea\uff0c\u8bf7\u5148\u8fdb\u884c\u9884\u6d4b'); return; }
    var features = getFeatures();
    var result = predictLGB(window.MODEL, features);
    renderResult(result);
  } catch(e) { alert('\u9884\u6d4b\u9519\u8bef: ' + e.message + '\\n\\n' + (e.stack||'').split('\\n').slice(0,3).join('\\n')); console.error(e); }
}

function renderResult(r) {
  document.getElementById('predClass').textContent = r.predicted_class;
  document.getElementById('predConf').textContent = r.confidence + '%';
  var maxP = Math.max.apply(null, r.all_probabilities.map(function(p){return p.probability;}));
  document.getElementById('probaBars').innerHTML = r.all_probabilities.map(function(p){
    var pct = (p.probability/maxP*100).toFixed(1);
    var isPred = String(p['class']) === String(r.predicted_class);
    return '<div class="proba-item">' +
      '<div class="proba-label"><span>\u7c7b\u522b ' + p['class'] + '</span><span>' + (p.probability*100).toFixed(1) + '%</span></div>' +
      '<div class="proba-track"><div class="proba-fill ' + (isPred?'highlight':'normal') + '" style="width:' + pct + '%"></div></div>' +
      '</div>';
  }).join('');
  document.getElementById('resultCard').style.display='block';
  document.getElementById('errorCard').style.display='none';
}

function predictLGB(model, features) {
  try {
  var numClasses = model.num_classes || 2;
  var trees = (model.trees && model.trees.tree_info) ? model.trees.tree_info : [];
  var isMulti = numClasses > 2;
  var contribs = [];
  for (var ci = 0; ci < numClasses; ci++) contribs[ci] = 0.0;

  for (var ti = 0; ti < trees.length; ti++) {
    var treeEntry = trees[ti];
    var treeIdx = treeEntry.tree_index;
    var targetClass = isMulti ? (treeIdx % numClasses) : 0;
    if (targetClass < 0 || targetClass >= numClasses) continue;
    if (!treeEntry.tree_structure) continue;
    contribs[targetClass] += getLeafValue(treeEntry.tree_structure, features);
  }

  var probs;
  if (isMulti) {
    var maxC = Math.max.apply(null, contribs);
    var exps = contribs.map(function(c){return Math.exp(c - maxC);});
    var sumExp = exps.reduce(function(a, b){return a + b;}, 0);
    probs = exps.map(function(e){return e / sumExp;});
  } else {
    var sig = 1.0 / (1.0 + Math.exp(-contribs[0]));
    probs = [1.0 - sig, sig];
  }

  var predIdx = probs.indexOf(Math.max.apply(null, probs));
  var className = window._classes[predIdx] !== undefined ? window._classes[predIdx] : String(predIdx);
  return {
    predicted_class: className,
    confidence: Math.round(probs[predIdx] * 100, 2),
    all_probabilities: window._classes.map(function(c, i){
      return {class: c, probability: probs[i]};
    })
  };
  } catch(e) { throw new Error('predictLGB: ' + e.message + ' trees=' + (model.trees ? model.trees.tree_info.length : 'N/A')); }
}

// LightGBM 嵌套对象树遍历
// node: { split_feature, threshold, decision_type, default_left, left_child, right_child, leaf_value }
// 叶子节点有 leaf_value，内部节点有 left_child/right_child
function getLeafValue(node, features) {
  if (!node) return 0.0;
  if (node.leaf_value !== undefined) return node.leaf_value;
  var featIdx = node.split_feature;
  var v = (featIdx !== undefined && featIdx !== null) ? features[featIdx] : undefined;
  var isMissing = (v === undefined || isNaN(v));
  if (isMissing) return getLeafValue(node.default_left ? node.left_child : node.right_child, features);
  var goLeft = (node.decision_type === '<=') ? (v <= node.threshold) : (v <= node.threshold);
  return getLeafValue(goLeft ? node.left_child : node.right_child, features);
}
</script>
</body>
</html>'''


# ── 测试路由（不依赖训练，可直接访问）──────────────────────────────────

# ── 静态 HTML 路由 ────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/LightGBM_damo.html')
def demo_page():
    return send_from_directory('.', 'LightGBM_damo.html')


@app.route('/train_data.csv')
def serve_csv():
    return send_from_directory('.', 'train_data.csv')


if __name__ == '__main__':
    print(f'当前目录: {os.getcwd()}')
    print(f'数据文件: {os.path.exists(DATA_FILE)}')
    print('启动服务: http://127.0.0.1:5000/LightGBM_damo.html')
    app.run(host='0.0.0.0', port=5000, debug=True)