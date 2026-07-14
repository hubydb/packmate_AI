"""
LightGBM 零件包装方案训练服务（plus）
启动方式: python train_server_plus.py
默认训练文件: data/train_data.csv
特征列：零件号、零件名称、包装袋情况、干燥剂情况、零件质量要求、
       包装等级分类、单车用量、零件分类、零件重量（KG）、零件种类
训练目标：CKD包装类型
训练后可通过 /predict_plus 打开离线预测页，或通过 /api/standalone-predict-only 下载独立 HTML。
"""

import io
import json
import math
import os
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

DATA_FILE = "data/train_data.csv"
TEMPLATE_FILE = "LightGBM_damo_plus.html"
LABEL_COL = "CKD包装类型"

# 期望的特征列（与 data/train_data.csv 一致）
FEATURE_ORDER = [
    "零件号",
    "零件名称",
    "包装袋情况",
    "干燥剂情况",
    "零件质量要求",
    "包装等级分类",
    "单车用量",
    "零件分类",
    "零件重量（KG）",
    "零件种类",
]

train_result = None
_trained_model = None
_trained_le = None
_trained_params = None
_feature_names = None
_feature_meta = None
_col_encoders = None
_cached_standalone_html = None


class NaNSafeJSONProvider(app.json_provider_class):
    def default(self, o):
        if isinstance(o, float):
            try:
                if math.isnan(o) or math.isinf(o):
                    return None
            except (TypeError, ValueError):
                pass
        if isinstance(o, (np.floating, np.integer)):
            value = float(o)
            if math.isnan(value) or math.isinf(value):
                return None
        return super().default(o)


app.json = NaNSafeJSONProvider(app)


def _clean_nan(obj):
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        try:
            value = float(obj)
            if math.isnan(value) or math.isinf(value):
                return None
        except (TypeError, ValueError):
            pass
    return obj


def _is_numeric_dtype(series):
    """检查 pandas Series 是否可全部转为数值（允许部分 NaN）。"""
    if pd.api.types.is_numeric_dtype(series):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    non_null_before = series.notna().sum()
    non_null_after = converted.notna().sum()
    return non_null_before > 0 and non_null_after == non_null_before


def _normalize_dataframe(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace(r"^\s*$", np.nan, regex=True)

    if LABEL_COL not in df.columns:
        raise ValueError(f"CSV 必须包含 {LABEL_COL} 列")

    missing = [col for col in FEATURE_ORDER if col not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少特征列: {missing}")

    # 过滤标签为空/缺失的行
    df = df[df[LABEL_COL].notna()].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[df[LABEL_COL] != ""]

    # 仅保留需要的列 + label
    df = df[FEATURE_ORDER + [LABEL_COL]].copy()

    # 数值列转数值；非数值列填充缺失标记
    for col in FEATURE_ORDER:
        if _is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].fillna("__MISSING__").astype(str).str.strip()
            df.loc[df[col] == "", col] = "__MISSING__"

    return df


def train_model(df):
    global train_result, _trained_model, _trained_le, _trained_params, _feature_names, _feature_meta, _col_encoders

    source_df = _normalize_dataframe(df)
    df_features = source_df[FEATURE_ORDER].copy()

    # 对非数值列做 Label Encoding，数值列保持原样
    _col_encoders = {}
    feature_meta = []
    for col in FEATURE_ORDER:
        if _is_numeric_dtype(df_features[col]):
            df_features[col] = pd.to_numeric(df_features[col], errors="coerce")
            _col_encoders[col] = None
            values = pd.to_numeric(df_features[col], errors="coerce").dropna()
            if values.empty:
                stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
            else:
                std = float(values.std(ddof=0))
                if std <= 1e-9:
                    std = float((float(values.max()) - float(values.min())) / 4.0)
                stats = {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                    "std": float(std),
                }
            feature_meta.append({
                "name": col,
                "kind": "numeric",
                "ui_type": "number",
                "stats": stats,
            })
        else:
            enc = LabelEncoder()
            orig = df_features[col].fillna("__MISSING__").astype(str)
            encoded = enc.fit_transform(orig)
            df_features[col] = encoded
            _col_encoders[col] = {"enc": enc, "mapping": dict(zip(enc.classes_, range(len(enc.classes_))))}
            # 文本较长的列用 textarea，其它分类列用 select
            is_long_text = col in {"零件号", "零件名称", "零件质量要求"}
            feature_meta.append({
                "name": col,
                "kind": "categorical",
                "ui_type": "textarea" if is_long_text else "select",
                "categories": [str(v) for v in enc.classes_],
                "mode": "text" if is_long_text else "choice",
            })

    X = df_features.values.astype(float)
    y_raw = source_df[LABEL_COL].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    # 当类别样本不足时分层会报错，仅在安全时启用
    class_counts = pd.Series(y).value_counts()
    stratify = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    params = {
        "objective": "multiclass" if num_classes > 2 else "binary",
        "num_class": num_classes,
        "metric": "multi_logloss" if num_classes > 2 else "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
    )

    _trained_model = model
    _trained_le = le
    _trained_params = params
    _feature_names = FEATURE_ORDER[:]
    _feature_meta = feature_meta

    y_pred_proba = model.predict(X_test)
    if num_classes > 2:
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    accuracy = float(np.mean(y_pred == y_test))
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(_feature_names, importance), key=lambda x: x[1], reverse=True)

    feat_stats = []
    for i, name in enumerate(_feature_names):
        col = X[:, i].astype(float)
        feat_stats.append({
            "name": name,
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "mean": float(np.mean(col)),
            "std": float(np.std(col)) if np.std(col) > 1e-9 else float((np.max(col) - np.min(col)) / 4),
        })

    train_result = {
        "num_classes": int(num_classes),
        "class_names": [str(c) for c in le.classes_],
        "num_features": int(X.shape[1]),
        "feature_names": _feature_names,
        "num_train": int(X_train.shape[0]),
        "num_test": int(X_test.shape[0]),
        "accuracy": round(accuracy, 4),
        "accuracy_pct": f"{accuracy * 100:.2f}%",
        "num_trees": model.num_trees(),
        "model_ready": True,
        "feature_importance": [(name, float(score)) for name, score in feat_imp[:15]],
        "per_class_accuracy": _get_per_class_accuracy(y_test, y_pred, le.classes_),
        "feature_stats": feat_stats,
        "feature_meta": feature_meta,
    }

    return train_result


def _get_per_class_accuracy(y_true, y_pred, classes):
    results = []
    for i, c in enumerate(classes):
        mask = y_true == i
        if mask.sum() > 0:
            cls_acc = np.mean((y_pred == i)[mask])
            results.append({"class": str(c), "accuracy": round(float(cls_acc), 4), "count": int(mask.sum())})
    return results


def _make_export_payload():
    raw = _trained_model.dump_model()

    def _strip_tree(node):
        if node is None:
            return None
        if isinstance(node, dict):
            if "leaf_value" in node:
                return {"leaf_value": node["leaf_value"]}
            result = {
                "split_feature": node.get("split_feature"),
                "threshold": node.get("threshold"),
                "decision_type": node.get("decision_type", "<="),
                "default_left": node.get("default_left", True),
            }
            lc = node.get("left_child")
            rc = node.get("right_child")
            if lc is not None:
                result["left_child"] = _strip_tree(lc)
            if rc is not None:
                result["right_child"] = _strip_tree(rc)
            return result
        return node

    tree_info = []
    for t in (raw.get("tree_info") or []):
        ts = _strip_tree(t.get("tree_structure"))
        if ts:
            tree_info.append({
                "tree_index": t.get("tree_index", 0),
                "num_leaves": t.get("num_leaves", 1),
                "tree_structure": ts,
            })

    feature_categories = {}
    feature_cat_encoding = {}
    if _col_encoders:
        for col, info in _col_encoders.items():
            if info is not None:
                feature_categories[col] = list(info["enc"].classes_)
                feature_cat_encoding[col] = info["mapping"]

    return {
        "num_classes": int(train_result["num_classes"]),
        "class_names": [str(c) for c in _trained_le.classes_],
        "feature_names": _feature_names,
        "feature_meta": _feature_meta,
        "num_features": len(_feature_names),
        "objective": _trained_params["objective"],
        "trees": {"tree_info": tree_info},
        "accuracy_pct": train_result["accuracy_pct"],
        "num_trees": train_result["num_trees"],
        "feature_stats": train_result.get("feature_stats", []),
        "feature_categories": feature_categories,
        "feature_cat_encoding": feature_cat_encoding,
    }


def _render_standalone_html():
    template_path = pathlib.Path(__file__).with_name(TEMPLATE_FILE)
    template = template_path.read_text(encoding="utf-8")
    payload = _make_export_payload()
    model_script = "<script>window.LGB_MODEL = " + json.dumps(payload, ensure_ascii=False) + ";</script>"
    return template.replace("<!--MODEL_SCRIPT-->", model_script)


# ── API 路由 ──────────────────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def api_train():
    try:
        file_obj = request.files.get("file")
        if file_obj:
            raw = file_obj.read()
            df = pd.read_csv(io.StringIO(raw.decode("utf-8", errors="replace")))
        else:
            df = pd.read_csv(DATA_FILE)

        result = train_model(df)
        global _cached_standalone_html
        _cached_standalone_html = _render_standalone_html()
        return jsonify(_clean_nan({"success": True, "data": result}))
    except Exception as exc:
        import traceback
        return jsonify({"success": False, "error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/api/result", methods=["GET"])
def api_result():
    if train_result is None:
        return jsonify({"success": False, "error": "尚未训练，请先调用 /api/train"}), 400
    return jsonify(_clean_nan({"success": True, "data": train_result}))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if _trained_model is None:
        return jsonify({"success": False, "error": "模型尚未训练，请先调用 /api/train"}), 400

    data = request.get_json(silent=True) or {}
    features = data.get("features")
    if not isinstance(features, list):
        return jsonify({"success": False, "error": "请提供 features 数组"}), 400
    if len(features) != len(_feature_names):
        return jsonify({"success": False, "error": f"特征数量不匹配，需要 {len(_feature_names)} 个特征"}), 400

    try:
        x = np.array(features, dtype=float).reshape(1, -1)
        proba = _trained_model.predict(x)[0]

        if train_result["num_classes"] > 2:
            pred_class_idx = int(np.argmax(proba))
            pred_proba = [float(p) for p in proba]
        else:
            pred_class_idx = int(proba[0] > 0.5)
            pred_proba = [float(1 - proba[0]), float(proba[0])]

        pred_label = str(_trained_le.classes_[pred_class_idx])
        confidence = float(proba[pred_class_idx])

        pred_data = {
            "predicted_class": pred_label,
            "predicted_class_idx": pred_class_idx,
            "confidence": round(float(confidence) * 100, 2),
            "probability": round(float(confidence), 4),
            "all_probabilities": [
                {"class": str(c), "probability": round(float(p), 4)}
                for c, p in zip(_trained_le.classes_, pred_proba)
            ],
        }
        return jsonify(_clean_nan({"success": True, "data": pred_data}))
    except Exception as exc:
        import traceback
        return jsonify({"success": False, "error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/api/model", methods=["GET"])
def api_model():
    if _trained_model is None:
        return jsonify({"success": False, "error": "模型尚未训练"}), 400
    try:
        return jsonify(_clean_nan({"success": True, "data": _make_export_payload()}))
    except Exception as exc:
        import traceback
        return jsonify({"success": False, "error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/api/standalone-predict-only", methods=["GET"])
def api_standalone_predict_only():
    if _trained_model is None:
        return jsonify({"success": False, "error": "模型尚未训练"}), 400
    try:
        html_content = _render_standalone_html()
        resp = Response(html_content, mimetype="text/html; charset=utf-8")
        resp.headers["Content-Disposition"] = "attachment; filename=LightGBM_damo_plus.html"
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp
    except Exception as exc:
        import traceback
        return jsonify({"success": False, "error": str(exc), "trace": traceback.format_exc()}), 500


_TRAIN_PAGE_HTML = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>模型训练（plus）</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:#1c2435;background:#f3f5f7;line-height:1.6;min-height:100vh}
.header{background:#fff;border-bottom:1px solid #e8ecf0;padding:16px 32px}
.header h1{font-size:18px;font-weight:600;color:#0052d9}
.main{max-width:800px;margin:24px auto;padding:0 24px}
.card{background:#fff;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.08);padding:24px 28px;margin-bottom:20px}
.card-title{font-size:16px;font-weight:600;margin-bottom:16px}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;padding:0 20px;height:36px;border:1px solid #c0c6d1;border-radius:4px;font-size:14px;cursor:pointer;background:#fff;color:#1c2435;text-decoration:none}
.btn-primary{background:#0052d9;color:#fff;border-color:#0052d9}
.btn:disabled{opacity:.5;cursor:not-allowed}
.progress-bar-wrap{height:8px;background:#e8ecf0;border-radius:4px;overflow:hidden;margin:12px 0}
.progress-bar-fill{height:100%;background:#0052d9;width:0%;transition:width .3s}
.progress-log{background:#f3f5f7;border-radius:4px;padding:12px 16px;font-family:monospace;font-size:12px;max-height:240px;overflow-y:auto;line-height:1.8}
.alert{padding:12px 16px;border-radius:6px;font-size:13px;margin-bottom:16px;background:#e8f0fe;color:#0052d9}
.alert-error{background:#fef0ef;color:#d54941}
.alert-success{background:#e8f8ef;color:#2ba471}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid #c0c6d1;border-top-color:#0052d9;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<header class="header"><h1>LightGBM 模型训练（plus）</h1></header>
<div class="main">
  <div class="card">
    <div class="card-title">选择训练文件</div>
    <div class="alert">
      默认使用 <b>data/train_data.csv</b>；也可上传自定义 CSV。<br/>
      CSV 必须包含特征列：零件号、零件名称、包装袋情况、干燥剂情况、零件质量要求、包装等级分类、单车用量、零件分类、零件重量（KG）、零件种类；以及标签列 <b>CKD包装类型</b>。
    </div>
    <input type="file" id="fileInput" accept=".csv" style="margin-bottom:16px;" />
    <div>
      <button class="btn btn-primary" id="trainBtn" onclick="startTrain()">
        <span class="spinner" id="btnSpinner" style="display:none;"></span><span id="btnText">开始训练</span>
      </button>
      <a class="btn" id="predictLink" href="/predict_plus" style="display:none;margin-left:10px;">打开预测工具</a>
      <a class="btn" id="downloadLink" href="/api/standalone-predict-only" style="display:none;margin-left:10px;">下载离线版</a>
    </div>
    <div class="progress-bar-wrap" id="progressWrap" style="display:none;"><div class="progress-bar-fill" id="progressBar"></div></div>
    <div class="progress-log" id="log"></div>
    <div id="resultArea"></div>
  </div>
</div>
<script>
function log(msg, cls) {
  const el = document.getElementById('log');
  el.innerHTML += '<div class="' + (cls||'') + '">' + new Date().toLocaleTimeString() + ' ' + msg + '</div>';
  el.scrollTop = el.scrollHeight;
}
async function startTrain() {
  const file = document.getElementById('fileInput').files[0];
  const form = new FormData();
  if (file) form.append('file', file);
  document.getElementById('trainBtn').disabled = true;
  document.getElementById('btnSpinner').style.display = 'inline-block';
  document.getElementById('progressWrap').style.display = 'block';
  document.getElementById('progressBar').style.width = '40%';
  log('开始上传并训练...');
  try {
    const resp = await fetch('/api/train', {method:'POST', body:form});
    const data = await resp.json();
    document.getElementById('progressBar').style.width = '100%';
    document.getElementById('btnSpinner').style.display = 'none';
    if (!data.success) {
      log('训练失败: ' + data.error, 'alert-error');
      document.getElementById('trainBtn').disabled = false;
      return;
    }
    const r = data.data;
    log('训练完成！测试集准确率: ' + r.accuracy_pct, 'alert-success');
    document.getElementById('resultArea').innerHTML =
      '<div class="alert alert-success">' +
      '特征数: ' + r.num_features + ' | 类别数: ' + r.num_classes +
      ' | 训练样本: ' + r.num_train + ' | 测试样本: ' + r.num_test +
      ' | 树数量: ' + r.num_trees + '</div>';
    document.getElementById('predictLink').style.display = 'inline-flex';
    document.getElementById('downloadLink').style.display = 'inline-flex';
  } catch(e) {
    document.getElementById('btnSpinner').style.display = 'none';
    log('请求失败: ' + e.message, 'alert-error');
    document.getElementById('trainBtn').disabled = false;
  }
}
</script>
</body>
</html>'''


@app.route("/train_plus")
def page_train_plus():
    return Response(_TRAIN_PAGE_HTML, mimetype="text/html; charset=utf-8")


@app.route("/predict_plus")
def page_predict_plus():
    if _cached_standalone_html is None:
        return '<h2>请先在 <a href="/train_plus">训练页面</a> 完成模型训练</h2>', 404
    resp = Response(_cached_standalone_html, mimetype="text/html; charset=utf-8")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/LightGBM_damo_plus.html")
def demo_page_plus():
    if _cached_standalone_html is not None:
        resp = Response(_cached_standalone_html, mimetype="text/html; charset=utf-8")
        resp.headers["Cache-Control"] = "no-cache"
        return resp
    return send_from_directory(".", TEMPLATE_FILE)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/LightGBM_damo.html")
def legacy_demo_page():
    return send_from_directory(".", "LightGBM_damo.html")


@app.route("/LightGBM_damo_copy.html")
def legacy_demo_page_copy():
    return send_from_directory(".", "LightGBM_damo_copy.html")


if __name__ == "__main__":
    print(f"当前目录: {os.getcwd()}")
    print(f"数据文件: {DATA_FILE} (存在: {os.path.exists(DATA_FILE)})")
    print("训练页面: http://127.0.0.1:5000/train_plus")
    print("预测页面: http://127.0.0.1:5000/predict_plus")
    app.run(host="0.0.0.0", port=5000, debug=True)