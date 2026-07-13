"""
LightGBM 包装类型预测训练服务（适配 LightGBM_damo_copy.html）
启动方式: python train_server_copy.py
然后在浏览器打开 http://127.0.0.1:5000/LightGBM_damo_copy.html
预测目标：CKD包装类型（由前端预处理后发来 label 列）
"""

import os
import io
import json
import math
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, Response
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

app = Flask(__name__)


class NaNSafeJSONProvider(app.json_provider_class):
    """Flask JSON 编码器：把 nan/inf 转为 None（序列化 为 null），避免无效 JSON"""
    def default(self, o):
        if isinstance(o, float):
            try:
                if math.isnan(o) or math.isinf(o):
                    return None
            except (TypeError, ValueError):
                pass
        elif isinstance(o, (np.floating, np.integer)):
            f = float(o)
            if math.isnan(f) or math.isinf(f):
                return None
        return super().default(o)
app.json = NaNSafeJSONProvider(app)


def _clean_nan(obj):
    """递归把 nan/inf 替换为 None，避免 JSON 序列化失败"""
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        try:
            f = float(obj)
            if math.isnan(f) or math.isinf(f):
                return None
        except (TypeError, ValueError):
            pass
    return obj
DATA_FILE = 'data/test_input_total.csv'  # 默认训练文件

# ── 全局模型训练结果 ──
train_result = None
_trained_model = None
_trained_le = None
_trained_params = None
_feature_names = None
_col_encoders = None  # col_name -> {enc: LabelEncoder, mapping: dict} or None
_cat_feature_names = None  # 需要下拉选择的分类特征名列表


def _is_numeric_dtype(series):
    """检查 pandas Series 是否可全部转为数值（允许部分 NaN）"""
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors='coerce')
    # 若转换后非 NaN 的值数量与原列非空数量相同，则是数值列
    non_null_before = series.notna().sum()
    non_null_after = converted.notna().sum()
    return non_null_before > 0 and non_null_after == non_null_before
_cached_standalone_html = None


def train_model(df):
    """训练 LightGBM 模型，df 由调用方传入（label 列已由前端添加）"""
    global train_result, _trained_model, _trained_le, _trained_params, _feature_names, _col_encoders
    feature_names = df.drop('label', axis=1).columns.tolist()
    df_features = df.drop('label', axis=1).copy()

    # 对非数值列做 Label Encoding（转为整数），以便 LightGBM 处理
    _col_encoders = {}   # col_name -> {enc: LabelEncoder, mapping: dict} or None
    for col in feature_names:
        orig = df[col].fillna('__MISSING__').astype(str)
        if _is_numeric_dtype(df[col]):
            df_features[col] = pd.to_numeric(df[col], errors='coerce')
            _col_encoders[col] = None
        else:
            enc = LabelEncoder()
            encoded = enc.fit_transform(orig)
            df_features[col] = encoded
            _col_encoders[col] = {'enc': enc, 'mapping': dict(zip(enc.classes_, range(len(enc.classes_))))}
    global _cat_feature_names
    _cat_feature_names = [c for c in feature_names if _col_encoders[c] is not None]

    X = df_features.values.astype(float)
    y_raw = df['label'].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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

    _trained_model = model
    _trained_le = le
    _trained_params = params
    _feature_names = feature_names

    y_pred_proba = model.predict(X_test)
    if num_classes > 2:
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    accuracy = float(np.mean(y_pred == y_test))

    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

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
    train_result['feature_importance'] = [(name, float(score)) for name, score in feat_imp[:15]]
    train_result['per_class_accuracy'] = get_per_class_accuracy(y_test, y_pred, le.classes_, num_classes)
    train_result['feature_stats'] = feat_stats

    return train_result


def get_per_class_accuracy(y_true, y_pred, classes, num_classes):
    results = []
    for i, c in enumerate(classes):
        mask = y_true == i
        if mask.sum() > 0:
            cls_acc = np.mean((y_pred == i)[mask])
            results.append({'class': str(c), 'accuracy': round(float(cls_acc), 4), 'count': int(mask.sum())})
    return results


# ── API 路由 ──────────────────────────────────────────────────────────

@app.route('/api/train', methods=['POST'])
def api_train():
    """接收前端预处理后的 CSV（已含 label 列），执行训练"""
    try:
        file_obj = request.files.get('file')
        if file_obj:
            raw_bytes = file_obj.read()
            print('[DEBUG] Uploaded file size:', len(raw_bytes))
            csv_text = raw_bytes.decode('utf-8', errors='replace')
            # Show first 200 chars of first line (header)
            first_line_end = csv_text.find('\n')
            print('[DEBUG] CSV header:', csv_text[:first_line_end][:200])
            df = pd.read_csv(io.StringIO(csv_text))
        else:
            df = pd.read_csv(DATA_FILE)
        print('[DEBUG] df.columns:', list(df.columns))
        print('[DEBUG] df.shape:', df.shape)
        if 'label' not in df.columns:
            print('[DEBUG] ERROR: CSV has no label column! Columns:', list(df.columns))
            return jsonify({'success': False, 'error': f'CSV没有 label 列（列名={list(df.columns)[:10]}...），请确认上传的是 test_input_total.csv 而非 test_input.csv，前端预处理未生效'}), 400

        if 'label' not in df.columns:
            return jsonify({'success': False, 'error': 'CSV 必须包含 label 列'}), 400

        result = train_model(df)
        print('[DEBUG] Trained feature_names count:', len(_feature_names))
        print('[DEBUG] Feature names:', _feature_names)
        global _cached_standalone_html
        _cached_standalone_html = _build_standalone_predict_html()
        return jsonify(_clean_nan({'success': True, 'data': result}))
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/result', methods=['GET'])
def api_result():
    if train_result is None:
        return jsonify({'success': False, 'error': '尚未训练，请先调用 /api/train'}), 400
    return jsonify(_clean_nan({'success': True, 'data': train_result}))


@app.route('/api/predict', methods=['POST'])
def api_predict():
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

        pred_data = {
            'predicted_class': pred_label,
            'predicted_class_idx': pred_class_idx,
            'confidence': round(float(confidence) * 100, 2) if not (isinstance(confidence, float) and (math.isnan(confidence) or math.isinf(confidence))) else 0,
            'probability': round(float(confidence), 4) if not (isinstance(confidence, float) and (math.isnan(confidence) or math.isinf(confidence))) else 0,
            'all_probabilities': [
                {'class': str(c), 'probability': round(float(p), 4)}
                for c, p in zip(_trained_le.classes_, pred_proba)
            ],
        }
        return jsonify(_clean_nan({'success': True, 'data': pred_data}))
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/model', methods=['GET'])
def api_model():
    if _trained_model is None:
        return jsonify({'success': False, 'error': '模型尚未训练'}), 400

    try:
        model_json = _trained_model.dump_model()
        print('[DEBUG] api_model feature_names count:', len(_feature_names))
        # 构建分类特征的类别映射 {特征名: [原始类别值列表]}
        feature_categories = {}
        if _col_encoders:
            for col, info in _col_encoders.items():
                if info is not None:
                    feature_categories[col] = list(info['enc'].classes_)
        # 构建分类特征的编码映射 {特征名: {类别值: 整数编码}}
        feature_cat_encoding = {}
        if _col_encoders:
            for col, info in _col_encoders.items():
                if info is not None:
                    feature_cat_encoding[col] = info['mapping']
        export = {
            'num_classes': int(train_result['num_classes']),
            'class_names': [str(c) for c in _trained_le.classes_],
            'feature_names': _feature_names,
            'num_features': len(_feature_names),
            'objective': _trained_params['objective'],
            'trees': model_json,
            'feature_stats': train_result.get('feature_stats', []),
            'feature_categories': feature_categories,      # {col: [val1, val2, ...]}
            'feature_cat_encoding': feature_cat_encoding,  # {col: {val: int, ...}}
        }
        return jsonify(_clean_nan({'success': True, 'data': export}))
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/standalone-predict-only', methods=['GET'])
def api_standalone_predict_only():
    if _trained_model is None:
        return jsonify({'success': False, 'error': '模型尚未训练'}), 400
    try:
        html_content = _build_standalone_predict_html()
        resp = Response(html_content, mimetype='text/html; charset=utf-8')
        resp.headers['Content-Disposition'] = 'attachment; filename=LightGBM_predict_copy.html'
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/predict_copy')
def page_predict_copy():
    if _cached_standalone_html is None:
        return '<h2>请先在 <a href="/">训练页面</a> 完成模型训练</h2>', 404
    resp = Response(_cached_standalone_html, mimetype='text/html; charset=utf-8')
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


def _build_standalone_predict_html():
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
    # 添加分类特征信息
    if _col_encoders:
        export['feature_categories'] = {
            col: list(info['enc'].classes_)
            for col, info in _col_encoders.items() if info is not None
        }
        export['feature_cat_encoding'] = {
            col: info['mapping']
            for col, info in _col_encoders.items() if info is not None
        }
    model_js = json.dumps(export, ensure_ascii=False)
    model_js = model_js.replace('</script', '<\\/script')
    return _build_predict_only_html(model_js, _feature_names, train_result)


def _build_predict_only_html(model_js, feature_names, train_result):
    QUALITY_ITEMS = [
        '防变形', '防潮湿', '防冲击', '防丢失', '防腐蚀', '防划伤', '防挤压', '防静电',
        '防老化', '防磨损', '防碰撞', '防破碎', '防破损', '防损坏', '防损伤', '防脱落',
        '防污损', '防锈蚀', '防油污', '防震动'
    ]

    # 获取分类特征信息（全局变量，在 train_model 中填充）
    feat_categories = {}
    if _col_encoders:
        for col, info in _col_encoders.items():
            if info is not None:
                feat_categories[col] = list(info['enc'].classes_)

    feat_rows = []
    for name in feature_names:
        if name == '零件质量要求':
            feat_rows.append(
                '      <div class="feat-input-group">'
                '<label for="qualitySelect">零件质量要求</label>'
                '<select id="qualitySelect" style="height:31px;padding:0 9px;border:1px solid #c0c6d1;border-radius:4px;font-size:13px;outline:none;width:100%;">'
                '<option value="">-- 请选择 --</option>'
                '</select>'
                '</div>'
            )
            feat_rows.append(
                '      <div style="grid-column:1/-1;margin-top:4px;">'
                '<div style="font-size:12px;color:#4f5d73;font-weight:500;margin-bottom:8px;">核心防护项目（质量特征）</div>'
                '<div id="qualityCheckboxes" style="display:flex;flex-wrap:wrap;gap:8px;">' +
                ''.join(
                    f'<label style="display:flex;align-items:center;gap:4px;font-size:13px;color:#4f5d73;cursor:pointer;">'
                    f'<input type="checkbox" id="qc_{item}" disabled /> {item}</label>'
                    for item in QUALITY_ITEMS
                ) +
                '</div>'
                '</div>'
            )
        elif name.startswith('质量_'):
            pass  # 由下拉框联动，不单独渲染
        elif name in feat_categories:
            # 分类特征：渲染为下拉框
            cats = feat_categories[name]
            opts = ''.join(f'<option value="{c}">{c}</option>' for c in cats)
            feat_rows.append(
                f'      <div class="feat-input-group">'
                f'<label for="feat_{name}">{name}</label>'
                f'<select id="feat_{name}" style="height:31px;padding:0 9px;border:1px solid #c0c6d1;border-radius:4px;font-size:13px;outline:none;width:100%;">'
                f'<option value="">-- 请选择 --</option>{opts}'
                f'</select>'
                f'</div>'
            )
        else:
            # 数值特征
            feat_rows.append(
                f'      <div class="feat-input-group">'
                f'<label for="feat_{name}">{name}</label>'
                f'<input type="number" id="feat_{name}" placeholder="选填" step="any" />'
                f'</div>'
            )

    feat_grid_html = '\n'.join(feat_rows)

    class_cards = []
    for c in (train_result.get('per_class_accuracy') or []):
        class_cards.append(
            f'<div class="class-chip">'
            f'<span class="class-chip-name">类别 {c["class"]}</span>'
            f'<span class="class-chip-acc">{c["accuracy"] * 100:.1f}%</span>'
            f'</div>'
        )
    class_chips_html = ('<div class="class-chips">' + ''.join(class_cards) + '</div>') if class_cards else ''

    html = (_STANDALONE_PREDICT_TEMPLATE
        .replace('$MODEL', model_js)
        .replace('$FEAT', feat_grid_html)
        .replace('$CHIPS', class_chips_html))
    return html


_STANDALONE_PREDICT_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>预测工具（包装类型）</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:#1c2435;background:#f3f5f7;line-height:1.6;min-height:100vh}
.header{background:#fff;border-bottom:1px solid #e8ecf0;padding:14px 28px;display:flex;align-items:center;gap:12px}
.header h1{font-size:17px;font-weight:700;color:#0052d9}
.header .subtitle{font-size:12px;color:#8a94a6}
.header .model-info{margin-left:auto;font-size:12px;color:#8a94a6}
.main{max-width:900px;margin:20px auto;padding:0 20px}
.card{background:#fff;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.08);padding:20px 24px;margin-bottom:16px}
.card-title{font-size:15px;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:6px}
.alert{padding:10px 14px;border-radius:6px;font-size:13px;margin-bottom:14px;background:#e8f0fe;color:#0052d9}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:5px;padding:0 18px;height:36px;border:1px solid #c0c6d1;border-radius:4px;font-size:14px;cursor:pointer;background:#fff;color:#1c2435}
.btn:hover{border-color:#8a94a6}
.btn-primary{background:#0052d9;color:#fff;border-color:#0052d9}
.btn-primary:hover{background:#0041b3}
.btn-success{background:#2ba471;color:#fff;border-color:#2ba471}
.btn-success:hover{background:#1f8b5e}
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
.feat-input-group select{height:31px;padding:0 9px;border:1px solid #c0c6d1;border-radius:4px;font-size:13px;outline:none;color:#1c2435;width:100%}
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
.batch-table{width:100%;border-collapse:collapse;font-size:13px}
.batch-table th,.batch-table td{padding:7px 12px;border-bottom:1px solid #e8ecf0;text-align:left}
.batch-table th{background:#f3f5f7;font-weight:600;color:#4f5d73;font-size:12px}
.batch-table tr:hover td{background:#f8f9fb}
.batch-table .pred-col{font-weight:700;color:#0052d9}
.batch-table .conf-col{color:#4f5d73;font-size:12px}
.progress-bar-wrap{height:6px;background:#e8ecf0;border-radius:3px;overflow:hidden;margin:6px 0}
.progress-bar-fill{height:100%;background:#0052d9;width:0%;transition:width .3s}
</style>
</head>
<body>
<header class="header">
  <h1>预测工具（包装类型）</h1>
  <span class="subtitle">离线版 · 无需网络</span>
  <span class="model-info" id="modelInfo"></span>
  <span id="initStatus" style="margin-left:8px;font-size:11px;color:#d54941"></span>
</header>
<div class="main">
  <!-- 单条预测 -->
  <div class="card">
    <div class="card-title">特征输入</div>
    <div class="alert">零件质量要求请从下拉框选择，其他特征可留空。点击"随机填充"生成符合模型特征的随机示例值。</div>
    <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
      <button class="btn" id="autoFillBtn" onclick="autoFillRandom()">随机填充</button>
      <button class="btn" onclick="clearAll()">清空</button>
    </div>
    <div class="feat-inputs-grid" id="featGrid">$FEAT</div>
    <div class="btn-row">
      <button class="btn btn-primary btn-lg" id="predBtn" onclick="doPredict()">开始预测</button>
    </div>
  </div>

  <div class="card" id="resultCard" style="display:none;">
    <div class="card-title">预测结果</div>
    <div class="pred-result">
      <div class="pred-label">预测类别</div>
      <div class="pred-class" id="predClass">—</div>
      <div class="pred-conf">置信度：<strong id="predConf">—</strong></div>
    </div>
    <div style="font-size:12px;color:#8a94a6;margin:12px 0 4px;">各类别概率</div>
    <div class="proba-bars" id="probaBars"></div>
  </div>

  <div class="card" id="errorCard" style="display:none;">
    <div class="alert-error" id="errorMsg"></div>
  </div>

  <!-- 批量预测卡片 -->
  <div class="card" id="batchCard">
    <div class="card-title">📦 批量预测</div>
    <div class="alert">
      选择包含特征数据的 CSV 文件（表头需与特征名一致），将对每一行进行预测并输出结果。
    </div>
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px;">
      <button class="btn btn-primary" id="batchSelectBtn" onclick="document.getElementById('batchCsvInput').click()">
        📂 选择 CSV 文件
      </button>
      <span id="batchFileName" style="font-size:13px;color:#8a94a6;"></span>
      <input type="file" id="batchCsvInput" accept=".csv" style="display:none" onchange="handleBatchCsv(this.files[0])" />
    </div>
    <div style="display:flex;gap:10px;margin-bottom:12px;">
      <button class="btn btn-success btn-lg" id="batchPredictBtn" onclick="runBatchPrediction()" disabled>
        🚀 开始批量预测
      </button>
      <button class="btn btn-lg" id="batchExportBtn" onclick="exportBatchResults()" disabled>
        💾 导出 CSV
      </button>
      <button class="btn btn-lg" onclick="clearBatchResults()">
        🗑 清空
      </button>
    </div>
    <div id="batchProgressArea" style="display:none;margin-bottom:12px;">
      <div class="progress-bar-wrap"><div class="progress-bar-fill" id="batchProgressBar"></div></div>
      <div style="font-size:13px;color:#4f5d73;margin-top:4px;" id="batchProgressText"></div>
    </div>
    <div id="batchResultsArea" style="display:none;">
      <div style="margin-bottom:8px;font-size:13px;color:#4f5d73;">
        共 <strong id="batchResultCount">0</strong> 条预测结果
      </div>
      <div style="overflow-x:auto;max-height:400px;overflow-y:auto;border:1px solid #e8ecf0;border-radius:4px;">
        <table class="batch-table" id="batchResultsTable">
          <thead id="batchResultsHead" style="position:sticky;top:0;background:#f3f5f7;z-index:1;"></thead>
          <tbody id="batchResultsBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">模型信息</div>
    <div class="model-stats" id="modelStats"></div>
    $CHIPS
  </div>
</div>

<script>
var _modelErr = [];
try{ var MODEL_RAW = $MODEL; window.MODEL = MODEL_RAW; }
catch(e){ _modelErr.push('MODEL parse: ' + e.message); window.MODEL = null; }

// 通用预测函数 (与单条共用)
function predictLGB(model, features) {
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
}

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

// 解析零件质量要求，返回质量标志对象
function parseQualityText(text) {
  var flags = {};
  QUALITY_ITEMS.forEach(function(item){ flags['质量_' + item] = 0; });
  if (!text) return flags;
  var m = text.match(/核心防护：([^；]+)/);
  if (m) {
    m[1].split('、').forEach(function(item){
      if (flags.hasOwnProperty('质量_' + item)) flags['质量_' + item] = 1;
    });
  }
  return flags;
}

// 获取特征数组（用于单条预测）
function getFeatures() {
  if (!window._names) throw new Error('模型未就绪');
  var catEnc = window.MODEL && window.MODEL.feature_cat_encoding;
  return window._names.map(function(name){
    if (name === '零件质量要求') return NaN;
    if (name.startsWith('质量_')) {
      var item = name.replace('质量_', '');
      var cb = document.getElementById('qc_' + item);
      return cb && cb.checked ? 1 : 0;
    }
    var el = document.getElementById('feat_' + name);
    var v = el ? el.value.trim() : '';
    if (v === '') return NaN;
    if (catEnc && catEnc.hasOwnProperty(name)) {
      var mapping = catEnc[name];
      return mapping.hasOwnProperty(v) ? mapping[v] : NaN;
    }
    var f = parseFloat(v);
    return isNaN(f) ? NaN : f;
  });
}

// 单条预测入口
function doPredict() {
  try {
    var features = getFeatures();
    var result = predictLGB(window.MODEL, features);
    renderResult(result);
  } catch(e) { alert('预测错误: ' + e.message); console.error(e); }
}

function renderResult(r) {
  document.getElementById('predClass').textContent = r.predicted_class;
  document.getElementById('predConf').textContent = r.confidence + '%';
  var maxP = Math.max.apply(null, r.all_probabilities.map(function(p){return p.probability;}));
  document.getElementById('probaBars').innerHTML = r.all_probabilities.map(function(p){
    var pct = (p.probability/maxP*100).toFixed(1);
    var isPred = String(p['class']) === String(r.predicted_class);
    return '<div class="proba-item">' +
      '<div class="proba-label"><span>类别 ' + p['class'] + '</span><span>' + (p.probability*100).toFixed(1) + '%</span></div>' +
      '<div class="proba-track"><div class="proba-fill ' + (isPred?'highlight':'normal') + '" style="width:' + pct + '%"></div></div>' +
      '</div>';
  }).join('');
  document.getElementById('resultCard').style.display='block';
  document.getElementById('errorCard').style.display='none';
}

// ---- 批量预测相关 ----
var _batchCsvData = null;      // { header: [...], rows: [[...], ...] }
var _batchResults = null;

function handleBatchCsv(file) {
  if (!file) return;
  var nameEl = document.getElementById('batchFileName');
  var btn = document.getElementById('batchPredictBtn');
  if (file) {
    nameEl.textContent = '已选: ' + file.name;
    btn.disabled = false;
  } else {
    nameEl.textContent = '';
    btn.disabled = true;
  }
  _batchCsvData = null;
  _batchResults = null;
  document.getElementById('batchResultsArea').style.display = 'none';
  document.getElementById('batchExportBtn').disabled = true;
  var reader = new FileReader();
  reader.onload = function(e) {
    try {
      var text = e.target.result;
      var lines = text.trim().split('\\n');
      if (lines.length < 2) { alert('CSV 文件内容不足'); return; }
      var header = lines[0].split(',').map(function(h){ return h.trim(); });
      var rows = lines.slice(1).map(function(line){ return line.split(',').map(function(v){ return v.trim(); }); });
      _batchCsvData = { header: header, rows: rows };
      nameEl.textContent = '已选: ' + file.name + ' (' + rows.length + ' 条)';
    } catch(err) { alert('CSV 解析失败: ' + err.message); }
  };
  reader.onerror = function() { alert('文件读取失败'); };
  reader.readAsText(file);
}

function runBatchPrediction() {
  if (!_batchCsvData || _batchCsvData.rows.length === 0) {
    alert('请先选择一个有效的 CSV 文件');
    return;
  }
  if (!window.MODEL) { alert('模型未加载'); return; }
  var header = _batchCsvData.header;
  var rows = _batchCsvData.rows;
  var total = rows.length;
  document.getElementById('batchProgressArea').style.display = 'block';
  document.getElementById('batchResultsArea').style.display = 'none';
  document.getElementById('batchPredictBtn').disabled = true;
  var results = [];
  var catEnc = window.MODEL.feature_cat_encoding || {};
  var names = window._names;
  var classes = window._classes;

  for (var i = 0; i < total; i++) {
    var row = rows[i];
    var rowObj = {};
    header.forEach(function(h, idx){ rowObj[h] = (row[idx] !== undefined ? row[idx] : ''); });

    // 构建特征数组（与 getFeatures 逻辑一致，但基于 CSV 行）
    var features = names.map(function(name){
      if (name === '零件质量要求') return NaN;
      if (name.startsWith('质量_')) {
        var qtext = rowObj['零件质量要求'] || '';
        var flags = parseQualityText(qtext);
        return flags[name] !== undefined ? flags[name] : 0;
      }
      var v = rowObj[name];
      if (v === undefined || v === '' || v === null) return NaN;
      if (catEnc && catEnc.hasOwnProperty(name)) {
        var mapping = catEnc[name];
        return mapping.hasOwnProperty(v) ? mapping[v] : NaN;
      }
      var n = parseFloat(v);
      return isNaN(n) ? NaN : n;
    });

    try {
      var predResult = predictLGB(window.MODEL, features);
      var predClass = predResult.predicted_class;
      var confidence = predResult.confidence;
      results.push({
        rowIndex: i + 1,
        actual_class: rowObj['CKD包装类型'] || '',
        predicted_class: predClass,
        confidence: confidence,
        probability: predResult.all_probabilities[classes.indexOf(predClass)] ? predResult.all_probabilities[classes.indexOf(predClass)].probability : 0,
        rawFeatures: rowObj,
        error: null
      });
    } catch (err) {
      results.push({
        rowIndex: i + 1,
        actual_class: rowObj['CKD包装类型'] || '',
        predicted_class: '错误',
        confidence: 0,
        probability: 0,
        rawFeatures: rowObj,
        error: err.message
      });
    }

    var pct = Math.round((i + 1) / total * 100);
    document.getElementById('batchProgressBar').style.width = pct + '%';
    document.getElementById('batchProgressText').textContent = '已预测 ' + (i + 1) + ' / ' + total + ' 条（' + pct + '%）';
    if ((i + 1) % 50 === 0) {
      // 让 UI 更新
      setTimeout(function(){}, 0);
    }
  }

  _batchResults = results;
  document.getElementById('batchProgressBar').style.width = '100%';
  document.getElementById('batchProgressText').textContent = '预测完成！';
  renderBatchTable(results, header);
  document.getElementById('batchResultsArea').style.display = 'block';
  document.getElementById('batchPredictBtn').disabled = false;
  document.getElementById('batchExportBtn').disabled = false;
}

function renderBatchTable(results, csvHeader) {
  var thead = document.getElementById('batchResultsHead');
  var tbody = document.getElementById('batchResultsBody');
  var ths = ['序号'].concat(csvHeader, ['真实包装类型', '预测包装类型', '是否正确', '置信度(%)']);
  thead.innerHTML = '<tr>' + ths.map(function(h){ return '<th style="padding:8px 12px;border-bottom:1px solid #e8ecf0;white-space:nowrap;">' + h + '</th>'; }).join('') + '</tr>';
  tbody.innerHTML = results.map(function(r){
    var isCorrect = r.actual_class && String(r.actual_class) === String(r.predicted_class);
    var cells = [
      r.rowIndex,
      ...csvHeader.map(function(col){ return r.rawFeatures[col] !== undefined ? r.rawFeatures[col] : ''; }),
      r.actual_class,
      r.predicted_class,
      r.error ? '错误' : (isCorrect ? '✓' : '✗'),
      r.error ? r.error : r.confidence
    ];
    return '<tr>' + cells.map(function(v, idx){
      var style = 'padding:6px 12px;border-bottom:1px solid #e8ecf0;';
      if (idx === cells.length - 3) style += 'font-weight:600;color:#2ba471;';
      if (idx === cells.length - 2) style += 'font-weight:600;color:#0052d9;';
      return '<td style="' + style + '">' + v + '</td>';
    }).join('') + '</tr>';
  }).join('');
  document.getElementById('batchResultCount').textContent = results.length;
}

function exportBatchResults() {
  if (!_batchResults || !_batchCsvData) return;
  var header = _batchCsvData.header;
  var lines = [['序号', ...header, '真实包装类型', '预测包装类型', '是否正确', '置信度(%)']];
  _batchResults.forEach(function(r){
    var rawVals = header.map(function(col){ return r.rawFeatures[col] !== undefined ? r.rawFeatures[col] : ''; });
    var isCorrect = r.actual_class && String(r.actual_class) === String(r.predicted_class);
    lines.push([
      r.rowIndex,
      ...rawVals,
      r.actual_class,
      r.predicted_class,
      r.error ? '错误' : (isCorrect ? '✓' : '✗'),
      r.error ? r.error : r.confidence
    ]);
  });
  var csv = lines.map(function(row){ return row.map(function(v){ return '"' + String(v).replace(/"/g,'""') + '"'; }).join(','); }).join('\\n');
  var BOM = '\\uFEFF';
  var blob = new Blob([BOM + csv], { type: 'text/csv;charset=utf-8;' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'batch_predictions.csv';
  a.click();
  URL.revokeObjectURL(url);
}

function clearBatchResults() {
  _batchCsvData = null;
  _batchResults = null;
  document.getElementById('batchResultsArea').style.display = 'none';
  document.getElementById('batchProgressArea').style.display = 'none';
  document.getElementById('batchFileName').textContent = '';
  document.getElementById('batchCsvInput').value = '';
  document.getElementById('batchPredictBtn').disabled = true;
  document.getElementById('batchExportBtn').disabled = true;
  document.getElementById('batchProgressBar').style.width = '0%';
}

// ---- 初始化及辅助 ----
(function(){
  try{
    if (!window.MODEL) { alert('模型数据加载失败：' + _modelErr.join(', ')); return; }
    var names = window.MODEL.feature_names;
    var classes = window.MODEL.class_names;
    if (!names || !names.length) { alert('特征名不存在'); return; }
    if (!classes || !classes.length) { alert('类别名不存在'); return; }
    document.getElementById('modelInfo').textContent = classes.length + ' 个类别 · ' + names.length + ' 个特征';
    document.getElementById('modelStats').innerHTML =
      '<div class="model-stat">测试准确率 <strong>' + (window.MODEL.accuracy_pct||'—') + '</strong></div>' +
      '<div class="model-stat">树数量 <strong>' + (window.MODEL.num_trees||'—') + '</strong></div>' +
      '<div class="model-stat">特征数 <strong>' + names.length + '</strong></div>' +
      '<div class="model-stat">类别数 <strong>' + classes.length + '</strong></div>';
    window._names = names;
    window._classes = classes;
    document.getElementById('initStatus').textContent = '✓ 模型已加载';
    document.getElementById('initStatus').style.color = '#2ba471';

    // 零件质量要求下拉框联动勾选框
    var selectEl = document.getElementById('qualitySelect');
    var checkboxesEl = document.getElementById('qualityCheckboxes');
    if (selectEl && checkboxesEl) {
      selectEl.addEventListener('change', function() {
        var text = this.value;
        var flags = parseQualityText(text);
        QUALITY_ITEMS.forEach(function(item){
          var cb = document.getElementById('qc_' + item);
          if (cb) cb.checked = flags['质量_' + item] === 1;
        });
      });
    }
  } catch(e) {
    document.getElementById('initStatus').textContent = '✗ ' + e.message;
    alert('初始化错误: ' + e.message);
  }
})();

function autoFillRandom() {
  try {
    var stats = window.MODEL && window.MODEL.feature_stats;
    window._names.forEach(function(name, i){
      if (name === '零件质量要求' || name.startsWith('质量_')) return;
      var el = document.getElementById('feat_' + name);
      if (!el) return;
      var minVal = 0, maxVal = 100;
      if (stats && stats[i]) { minVal = stats[i].min; maxVal = stats[i].max; }
      var range = maxVal - minVal;
      var v = range > 0 ? minVal + Math.random() * range : minVal;
      el.value = parseFloat(v.toFixed(6));
    });
  } catch(e) { alert('随机填充失败: ' + e.message); }
}

function clearAll() {
  try {
    if (!window._names) return;
    window._names.forEach(function(name){
      if (name === '零件质量要求') {
        var sel = document.getElementById('qualitySelect');
        if (sel) sel.selectedIndex = 0;
        QUALITY_ITEMS.forEach(function(item){
          var cb = document.getElementById('qc_' + item);
          if (cb) cb.checked = false;
        });
      } else if (!name.startsWith('质量_')) {
        var el = document.getElementById('feat_' + name);
        if (el) el.value = '';
      }
    });
    document.getElementById('resultCard').style.display='none';
    document.getElementById('errorCard').style.display='none';
  } catch(e) { console.error('clearAll:', e.message); }
}

var QUALITY_ITEMS = [
  '防变形','防潮湿','防冲击','防丢失','防腐蚀','防划伤','防挤压','防静电',
  '防老化','防磨损','防碰撞','防破碎','防破损','防损坏','防损伤','防脱落',
  '防污损','防锈蚀','防油污','防震动'
];
</script>
</body>
</html>'''


# ── 静态路由 ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/LightGBM_damo_copy.html')
def demo_page_copy():
    return send_from_directory('.', 'LightGBM_damo_copy.html')


@app.route('/LightGBM_damo.html')
def demo_page():
    return send_from_directory('.', 'LightGBM_damo.html')


if __name__ == '__main__':
    print(f'当前目录: {os.getcwd()}')
    print(f'数据文件: {DATA_FILE} (存在: {os.path.exists(DATA_FILE)})')
    print('启动服务: http://127.0.0.1:5000/LightGBM_damo_copy.html')
    app.run(host='0.0.0.0', port=5000, debug=True)