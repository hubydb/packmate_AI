"""
零部件包装测算系统 —— 后端 API 服务（前后端解耦版）
启动方式: python api_server.py

功能特性：
1. 完整复用 train_server_llm.py 的核心逻辑与功能函数，仅做前后端解耦。
2. 后端仅暴露 REST API 与训练上传页面（/train_llm），不再渲染业务前端页面。
3. 前端页面（app.html）通过 /api/model 异步加载模型 JSON，独立部署。

配置方式（环境变量）：
- LLM_PROVIDER: siliconflow | dashscope | openai (默认: siliconflow)
- LLM_API_KEY: 你的API密钥
- LLM_BASE_URL: 自定义API地址（可选，用于SiliconFlow/OpenAI兼容接口）
- LLM_MODEL: 模型名称（默认: Qwen/Qwen2-7B-Instruct）

使用示例：
  # 1. 训练模型
  curl -X POST http://127.0.0.1:5001/api/train -F "file=@data/train_data.csv"

  # 2. 获取模型 JSON（供前端加载）
  curl http://127.0.0.1:5001/api/model

  # 3. 直接用 LLM 预测（无需训练）
  curl -X POST http://127.0.0.1:5001/api/llm_predict \
    -H "Content-Type: application/json" \
    -d '{"零件号":"test","零件名称":"螺钉","包装等级分类":"C","零件重量（KG）":0.5,"零件分类":"采购件"}'
"""

import io
import json
import math
import os
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from flask import Flask, Response, jsonify, request, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# ===== 配置 =====
DATA_FILE = None  # 不再设置默认训练文件，每次训练必须上传
LABEL_COL = "CKD包装类型"

# LLM 配置（支持多provider）
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "siliconflow").lower()
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")

# 根据provider设置默认模型
_DEFAULT_MODELS = {
    "siliconflow": "Qwen/Qwen2-7B-Instruct",
    "dashscope": "qwen-turbo",
    "openai": "gpt-3.5-turbo",
}
if not LLM_MODEL:
    LLM_MODEL = _DEFAULT_MODELS.get(LLM_PROVIDER, "Qwen/Qwen2-7B-Instruct")

# 根据provider设置默认URL
if not LLM_BASE_URL:
    if LLM_PROVIDER == "siliconflow":
        LLM_BASE_URL = "https://api.siliconflow.cn/v1"
    elif LLM_PROVIDER == "dashscope":
        LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    elif LLM_PROVIDER == "openai":
        LLM_BASE_URL = "https://api.openai.com/v1"

# 期望的特征列
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

# 包装类型选项（用于LLM输出解析）
PACKAGE_TYPES = [
    "纸箱", "木箱", "铁架/轻钢", "STD", "天地盖纸箱"
]

# 包装尺寸选项（用于LLM输出解析）
PACKAGE_SIZES = {
    "STD": [
        "360×280×200", "360×280×280", "360×560×200", "360×560×280",
    ],
    "纸箱": [
        "360×280×140", "360×280×180", "360×280×200", "360×280×280", "360×280×300", "360×280×400", "360×280×500",
        "560×280×140", "560×280×180", "560×280×200", "560×280×280", "560×280×300", "560×280×400", "560×280×500",
        "720×280×140", "720×280×180", "720×280×200", "720×280×280", "720×280×300", "720×280×400", "720×280×500",
        "1120×280×140", "1120×280×180", "1120×280×200", "1120×280×280", "1120×280×300", "1120×280×400", "1120×280×500",
        "1480×280×140", "1480×280×180", "1480×280×200", "1480×280×280", "1480×280×300", "1480×280×400", "1480×280×500",
        "360×560×140", "360×560×180", "360×560×200", "360×560×280", "360×560×300", "360×560×400", "360×560×500",
        "560×560×140", "560×560×180", "560×560×200", "560×560×280", "560×560×300", "560×560×400", "560×560×500",
        "720×560×140", "720×560×180", "720×560×200", "720×560×280", "720×560×300", "720×560×400", "720×560×500",
        "1120×560×140", "1120×560×180", "1120×560×200", "1120×560×280", "1120×560×300", "1120×560×400", "1120×560×500",
        "1480×560×140", "1480×560×180", "1480×560×200", "1480×560×280", "1480×560×300", "1480×560×400", "1480×560×500",
    ],
    "天地盖纸箱": [
        "980×1140×460", "980×1140×625", "980×1140×835", "980×1140×955", "980×1140×1120",
        "1180×1140×460", "1180×1140×625", "1180×1140×835", "1180×1140×955", "1180×1140×1120",
        "1320×1140×460", "1320×1140×625", "1320×1140×835", "1320×1140×955", "1320×1140×1120",
        "1480×1140×460", "1480×1140×625", "1480×1140×835", "1480×1140×955", "1480×1140×1120",
        "1700×1140×460", "1700×1140×625", "1700×1140×835", "1700×1140×955", "1700×1140×1120",
        "1960×1140×460", "1960×1140×625", "1960×1140×835", "1960×1140×955", "1960×1140×1120",
        "980×2280×460", "980×2280×625", "980×2280×835", "980×2280×955", "980×2280×1120",
        "1180×2280×460", "1180×2280×625", "1180×2280×835", "1180×2280×955", "1180×2280×1120",
        "1320×2280×460", "1320×2280×625", "1320×2280×835", "1320×2280×955", "1320×2280×1120",
        "1480×2280×460", "1480×2280×625", "1480×2280×835", "1480×2280×955", "1480×2280×1120",
        "1700×2280×460", "1700×2280×625", "1700×2280×835", "1700×2280×955", "1700×2280×1120",
        "1960×2280×460", "1960×2280×625", "1960×2280×835", "1960×2280×955", "1960×2280×1120",
    ],
    "木箱": [
        "980×1140×400", "980×1140×550", "980×1140×1200",
        "980×2280×400", "980×2280×550", "980×2280×1200",
        "1620×1140×400", "1620×1140×550", "1620×1140×1200",
        "1620×2280×400", "1620×2280×550", "1620×2280×1200",
        "1700×1140×400", "1700×1140×550", "1700×1140×1200",
        "1700×2280×400", "1700×2280×550", "1700×2280×1200",
    ],
    "铁架/轻钢": [
        "980×1140×300", "980×1140×625", "980×1140×835", "980×1140×1250", "980×1140×1480",
        "1180×1140×300", "1180×1140×625", "1180×1140×835", "1180×1140×1250", "1180×1140×1480",
        "1320×1140×300", "1320×1140×625", "1320×1140×835", "1320×1140×1250", "1320×1140×1480",
        "1480×1140×300", "1480×1140×625", "1480×1140×835", "1480×1140×1250", "1480×1140×1480",
        "1700×1140×300", "1700×1140×625", "1700×1140×835", "1700×1140×1250", "1700×1140×1480",
        "2360×1140×300", "2360×1140×625", "2360×1140×835", "2360×1140×1250", "2360×1140×1480",
        "2960×1140×300", "2960×1140×625", "2960×1140×835", "2960×1140×1250", "2960×1140×1480",
        "3660×1140×300", "3660×1140×625", "3660×1140×835", "3660×1140×1250", "3660×1140×1480",
        "980×2280×300", "980×2280×625", "980×2280×835", "980×2280×1250", "980×2280×1480",
        "1180×2280×300", "1180×2280×625", "1180×2280×835", "1180×2280×1250", "1180×2280×1480",
        "1320×2280×300", "1320×2280×625", "1320×2280×835", "1320×2280×1250", "1320×2280×1480",
        "1480×2280×300", "1480×2280×625", "1480×2280×835", "1480×2280×1250", "1480×2280×1480",
        "1700×2280×300", "1700×2280×625", "1700×2280×835", "1700×2280×1250", "1700×2280×1480",
        "2360×2280×300", "2360×2280×625", "2360×2280×835", "2360×2280×1250", "2360×2280×1480",
        "2960×2280×300", "2960×2280×625", "2960×2280×835", "2960×2280×1250", "2960×2280×1480",
        "3660×2280×300", "3660×2280×625", "3660×2280×835", "3660×2280×1250", "3660×2280×1480",
    ],
}

# 全局变量
train_result = None
_trained_model = None
_trained_le = None
_trained_params = None
_feature_names = None
_feature_meta = None
_col_encoders = None
_history_df = None  # 缓存历史训练数据用于软约束查找
_quality_keywords = []  # 从训练数据中提取的"防XX"关键词列表，排序后固定


def _parse_quality_text(text):
    """从零件质量要求文本中提取所有"防XX"关键词，返回集合"""
    if not text or not isinstance(text, str):
        return set()
    return set(re.findall(r'防[\u4e00-\u9fa5a-zA-Z0-9]+', text))


def _extract_quality_keywords(df):
    """从训练数据中提取所有不重复的"防XX"关键词，排序返回"""
    col = df["零件质量要求"] if "零件质量要求" in df.columns else pd.Series(dtype=str)
    all_kw = set()
    for val in col.dropna():
        all_kw.update(_parse_quality_text(str(val)))
    return sorted(all_kw)


def _expand_quality_features(df, keywords):
    """将零件质量要求列展开为多个二元特征列：质量_防XX"""
    if "零件质量要求" not in df.columns or not keywords:
        return df
    result = df.copy()
    for kw in keywords:
        col_name = f"质量_{kw}"
        result[col_name] = result["零件质量要求"].apply(
            lambda v: 1 if kw in _parse_quality_text(str(v)) else 0
        )
    return result



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


def _load_history_df():
    """惰性加载历史数据（用于软约束查找）"""
    global _history_df
    if _history_df is not None:
        return _history_df
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "data/train_data_AI.csv"),
        os.path.join(base_dir, "data/train_data.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
                df.columns = [str(c).strip() for c in df.columns]
                # 筛选已有实际包装尺寸的行
                size_cols = ["CKD 包装尺寸L", "CKD 包装尺寸L.1", "CKD 包装尺寸L.2"]
                has_size = df[size_cols].notna().any(axis=1) if all(c in df.columns for c in size_cols) else pd.Series([True] * len(df))
                _history_df = df[has_size].copy()
                print(f"历史数据已加载: {path}, 有效记录 {len(_history_df)} 条")
                return _history_df
            except Exception as e:
                print(f"加载历史数据失败 {path}: {e}")
    return None


def _find_similar_parts_history(part_data):
    """根据零件名称和分类，在历史数据中查找相似零件的包装经验"""
    df = _load_history_df()
    if df is None or len(df) == 0:
        return ""

    part_name = str(part_data.get("零件名称", "")).lower()
    category = str(part_data.get("零件分类", "")).lower()

    # 提取关键词（取零件名称中 ≥2 个字符的词）
    keywords = [w for w in re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]{2,}', part_name) if len(w) >= 2]

    candidates = []
    for _, row in df.iterrows():
        score = 0
        hist_name = str(row.get("零件名称", "")).lower()
        hist_cat = str(row.get("零件分类", "")).lower()

        # 名称关键词命中
        for kw in keywords:
            if kw in hist_name or kw in part_name:
                score += 1

        # 分类匹配
        if category and category != "" and category != "nan":
            if category in hist_cat or hist_cat in category:
                score += 2

        if score > 0:
            try:
                l = row.get("CKD 包装尺寸L", "")
                w = row.get("CKD 包装尺寸L.1", "")
                h = row.get("CKD 包装尺寸L.2", "")
                l_part = row.get("零件尺寸L", "")
                w_part = row.get("零件尺寸W", "")
                h_part = row.get("零件尺寸H", "")
                pkg_type = row.get("CKD包装类型", "")
                snp = row.get("CKD SNP", "")

                dim_str = f"{l}×{w}×{h}" if l and w and h else "未知"
                part_dim_str = f"{l_part}×{w_part}×{h_part}" if l_part and w_part and h_part and str(l_part) not in ("", "/", "nan") else "未知"
                candidates.append({
                    "score": score,
                    "name": hist_name,
                    "part_dim": part_dim_str,
                    "pkg_dim": dim_str,
                    "pkg_type": pkg_type,
                    "snp": snp,
                })
            except Exception:
                pass

    if not candidates:
        return ""

    # 按分数排序，取前5条
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:5]

    lines = []
    for c in top:
        lines.append(
            f"- 零件「{c['name']}」原始尺寸≈{c['part_dim']} → 实际使用{c['pkg_type']}包装尺寸{c['pkg_dim']}，SNP={c['snp']}"
        )
    return "\n".join(lines)


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

    df = df[df[LABEL_COL].notna()].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[df[LABEL_COL] != ""]
    df = df[FEATURE_ORDER + [LABEL_COL]].copy()

    for col in FEATURE_ORDER:
        if _is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].fillna("__MISSING__").astype(str).str.strip()
            df.loc[df[col] == "", col] = "__MISSING__"

    return df


def train_model(df):
    global train_result, _trained_model, _trained_le, _trained_params, _feature_names, _feature_meta, _col_encoders, _quality_keywords

    # 1. 提取质量要求关键词，展开为二元特征列
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    _quality_keywords = _extract_quality_keywords(df)
    print(f"[训练] 零件质量要求关键词数量: {len(_quality_keywords)}, 关键词: {_quality_keywords}")
    quality_binary_cols = [f"质量_{kw}" for kw in _quality_keywords]
    df = _expand_quality_features(df, _quality_keywords)

    # 2. 构建本轮训练使用的特征顺序（不含"零件质量要求"，替换为质量二元列）
    base_features = [c for c in FEATURE_ORDER if c != "零件质量要求"]
    run_feature_order = base_features + quality_binary_cols
    # 验证基础特征列存在
    missing = [col for col in base_features if col not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少特征列: {missing}")

    # 3. 标签列过滤
    if LABEL_COL not in df.columns:
        raise ValueError(f"CSV 必须包含 {LABEL_COL} 列")
    df = df[df[LABEL_COL].notna()].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[df[LABEL_COL] != ""]

    # 4. 特征标准化（数值→数值，分类→LabelEncoder）
    df_features = df[run_feature_order].copy()
    for col in run_feature_order:
        if _is_numeric_dtype(df_features[col]):
            df_features[col] = pd.to_numeric(df_features[col], errors="coerce")
        else:
            df_features[col] = df_features[col].fillna("__MISSING__").astype(str).str.strip()
            df_features.loc[df_features[col] == "", col] = "__MISSING__"

    # 5. 保存原始 df（用于后续步骤）
    source_df = df.copy()
    source_df[LABEL_COL] = df[LABEL_COL]

    _col_encoders = {}
    feature_meta = []
    for col in run_feature_order:
        if _is_numeric_dtype(df_features[col]):
            _col_encoders[col] = None
            values = pd.to_numeric(df_features[col], errors="coerce").dropna()
            if values.empty:
                stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
            else:
                std = float(values.std(ddof=0))
                if std <= 1e-9:
                    std = float((float(values.max()) - float(values.min())) / 4.0)
                stats = {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "std": std}
            feature_meta.append({"name": col, "kind": "numeric", "ui_type": "number", "stats": stats})
        else:
            enc = LabelEncoder()
            orig = df_features[col].fillna("__MISSING__").astype(str)
            encoded = enc.fit_transform(orig)
            df_features[col] = encoded
            _col_encoders[col] = {"enc": enc, "mapping": dict(zip(enc.classes_, range(len(enc.classes_))))}
            is_long_text = col in {"零件号", "零件名称"}
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

    class_counts = pd.Series(y).value_counts()
    stratify = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

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
        params, train_data, num_boost_round=200,
        valid_sets=[train_data, valid_data], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
    )

    _trained_model = model
    _trained_le = le
    _trained_params = params
    _feature_names = run_feature_order[:]
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
        "quality_keywords": _quality_keywords,
    }


# ===== LLM API 相关函数 =====

def _call_llm_api(messages, temperature=0.7, max_tokens=1000):
    """调用LLM API的统一接口"""
    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY 未设置，请设置环境变量 LLM_API_KEY")

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    url = f"{LLM_BASE_URL}/chat/completions"

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        raise ValueError(f"LLM返回格式异常: {result}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"LLM API调用失败: {str(e)}")


def _build_llm_prompt(part_data, history_context=""):
    """构建发送给LLM的prompt（支持软约束等效尺寸推断）"""
    # 提取关键特征
    part_no = part_data.get("零件号", "")
    part_name = part_data.get("零件名称", "")
    pkg_level = part_data.get("包装等级分类", "")
    weight = part_data.get("零件重量（KG）", "")
    category = part_data.get("零件分类", "")
    quality_req = part_data.get("零件质量要求", "")
    # 尺寸字段（train_data_AI.csv 中的扩展字段，可能为空）
    part_l = part_data.get("零件尺寸L", "")
    part_w = part_data.get("零件尺寸W", "")
    part_h = part_data.get("零件尺寸H", "")

    dim_info = ""
    if part_l and part_w and part_h and str(part_l) not in ("", "/", "nan"):
        dim_info = f"- 零件原始尺寸：{part_l} × {part_w} × {part_h} mm"

    hist_info = ""
    if history_context:
        hist_info = f"\n\n同类零件的历史包装经验：\n{history_context}\n\n请参考上述经验，判断该零件是否可以折叠、盘绕或压缩，并给出等效包装尺寸。"

    prompt = f"""你是一个汽车零部件包装方案专家。请根据以下零件信息，推荐最合适的包装类型和尺寸。

零件信息：
- 零件号：{part_no}
- 零件名称：{part_name}
- 包装等级分类：{pkg_level}
- 零件重量：{weight} kg
- 零件分类：{category}
- 质量要求：{quality_req}
{dim_info}{hist_info}

可选包装类型：{', '.join(PACKAGE_TYPES)}

请按照以下JSON格式返回推荐结果（只返回JSON，不要其他内容）：
{{
    "包装类型": "纸箱/木箱/铁箱/STD/天地盖/胶盒/托盘",
    "推荐尺寸": "长×宽×高（单位mm）",
    "等效零件尺寸": "长×宽×高（单位mm）- 考虑折叠/盘绕/压缩后的等效尺寸，如无折叠必要则与原始尺寸相同",
    "理由": "简短说明推荐原因，重点说明是否可折叠、折叠方式及等效尺寸计算依据"
}}

重要规则：
1. 零件重量判断包装类型：轻量件(<10kg)优先纸箱，中量件(10-50kg)优先木箱，重型件(>50kg)用铁箱
2. 可折叠零件（如电线/线束/软管/地毯/座椅垫等）：等效尺寸可显著小于原始尺寸，折叠后最大边不超过800mm的零件通常可用360×280×400纸箱
3. 刚性零件（如金属件/支架/壳体）：等效尺寸等于原始实际尺寸，必须能完整放入包装箱
4. 优先选择空间利用率高且满足零件放置的最小尺寸
5. 等效零件尺寸应考虑零件特性：
   - 电线/线束/软管：可盘绕，等效尺寸按盘绕后最大边计算（5m电线盘绕后约300-400mm）
   - 软质/柔性件：可折叠压扁，等效高度可大幅缩减
   - 刚性异形件：考虑插空放置，等效尺寸可略大于理论值（乘1.1-1.2系数）
   - 刚性标准件：严格按实际尺寸计算"""

    return prompt


def llm_predict_single(part_data):
    """使用LLM预测单个零件的包装方案（支持软约束等效尺寸）"""
    # 从训练历史中查找同类零件的经验
    history_context = _find_similar_parts_history(part_data)

    prompt = _build_llm_prompt(part_data, history_context)

    messages = [
        {"role": "system", "content": "你是一个专业的汽车零部件包装方案推荐专家，擅长根据零件特性和历史经验推荐最合适的包装。"},
        {"role": "user", "content": prompt}
    ]

    response = _call_llm_api(messages)

    # 解析JSON响应
    try:
        json_match = re.search(r'\{[\s\S]*?\}', response)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response)
    except json.JSONDecodeError:
        result = {"包装类型": "STD", "推荐尺寸": "", "等效零件尺寸": "", "理由": response}

    # 验证和标准化结果
    pkg_type = result.get("包装类型", "STD")
    if pkg_type not in PACKAGE_TYPES:
        pkg_type = "STD"

    return {
        "predicted_type": pkg_type,
        "recommended_size": result.get("推荐尺寸", ""),
        "effective_dimensions": result.get("等效零件尺寸", ""),
        "reason": result.get("理由", ""),
        "raw_response": response,
    }


# ===== API 路由 =====

@app.route("/api/llm_status", methods=["GET"])
def api_llm_status():
    """检查LLM API状态"""
    status = {
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "base_url": LLM_BASE_URL,
        "api_key_set": bool(LLM_API_KEY),
    }

    # 尝试一个简单的API调用测试
    if LLM_API_KEY:
        try:
            messages = [{"role": "user", "content": "你好"}]
            _call_llm_api(messages, max_tokens=10)
            status["connection"] = "ok"
        except Exception as e:
            status["connection"] = f"error: {str(e)}"
    else:
        status["connection"] = "api_key_not_set"

    # 历史数据状态
    hist_df = _load_history_df()
    if hist_df is not None:
        status["history_loaded"] = True
        status["history_count"] = len(hist_df)
        status["history_sources"] = list(hist_df["CKD包装类型"].value_counts().to_dict())
    else:
        status["history_loaded"] = False
        status["history_count"] = 0

    return jsonify({"success": True, "data": status})


@app.route("/api/llm_load_history", methods=["POST"])
def api_llm_load_history():
    """上传自定义历史数据文件，用于软约束尺寸推断"""
    global _history_df
    file_obj = request.files.get("file")
    if not file_obj:
        return jsonify({"success": False, "error": "请上传CSV文件"}), 400

    try:
        raw = file_obj.read()
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        df.columns = [str(c).strip() for c in df.columns]

        size_cols = ["CKD 包装尺寸L", "CKD 包装尺寸L.1", "CKD 包装尺寸L.2"]
        has_size = df[size_cols].notna().any(axis=1) if all(c in df.columns for c in size_cols) else pd.Series([True] * len(df))
        _history_df = df[has_size].copy()

        return jsonify(_clean_nan({
            "success": True,
            "data": {
                "message": f"已加载 {len(_history_df)} 条历史记录",
                "columns": list(_history_df.columns),
                "pkg_types": list(_history_df["CKD包装类型"].value_counts().to_dict()) if "CKD包装类型" in _history_df.columns else {},
            }
        }))
    except Exception as exc:
        import traceback
        return jsonify({"success": False, "error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/api/llm_predict", methods=["POST"])
def api_llm_predict():
    """直接使用LLM预测包装方案（无需训练模型）"""
    data = request.get_json(silent=True) or {}

    # 验证必填字段
    missing = []
    for col in ["零件号", "零件名称", "包装等级分类", "零件重量（KG）"]:
        if not data.get(col):
            missing.append(col)

    if missing:
        return jsonify({"success": False, "error": f"缺少必填字段: {', '.join(missing)}"}), 400

    try:
        result = llm_predict_single(data)
        return jsonify({"success": True, "data": result})
    except Exception as exc:
        import traceback
        return jsonify({"success": False, "error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/api/llm_enhance", methods=["POST"])
def api_llm_enhance():
    """使用LLM增强训练数据（补全缺失的标签）"""
    try:
        file_obj = request.files.get("file")
        if not file_obj:
            return jsonify({"success": False, "error": "请上传 CSV 训练文件"}), 400
        raw = file_obj.read()
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")

        # 找出缺失标签的行
        if LABEL_COL not in df.columns:
            return jsonify({"success": False, "error": f"CSV必须包含 {LABEL_COL} 列"}), 400

        missing_mask = df[LABEL_COL].isna() | (df[LABEL_COL] == "")
        missing_count = missing_mask.sum()

        if missing_count == 0:
            return jsonify({"success": True, "data": {"message": "没有需要增强的数据", "enhanced": 0}})

        # 只处理前N条缺失数据（避免API调用过多）
        max_enhance = min(missing_count, request.json.get("max_enhance", 10) if request.is_json else 10)
        missing_indices = df[missing_mask].index[:max_enhance].tolist()

        enhanced = 0
        results = []

        for idx in missing_indices:
            row = df.loc[idx].to_dict()
            try:
                result = llm_predict_single(row)
                df.loc[idx, LABEL_COL] = result["predicted_type"]
                results.append({"index": int(idx), "零件号": row.get("零件号", ""), "result": result})
                enhanced += 1
            except Exception as e:
                results.append({"index": int(idx), "error": str(e)})

            # 避免API限流
            if enhanced > 0 and enhanced % 5 == 0:
                time.sleep(1)

        # 返回增强后的数据（CSV格式）
        enhanced_csv = df.to_csv(index=False, encoding="utf-8-sig")

        return jsonify(_clean_nan({
            "success": True,
            "data": {
                "total_missing": missing_count,
                "enhanced": enhanced,
                "results": results,
                "enhanced_csv": enhanced_csv,
            }
        }))
    except Exception as exc:
        import traceback
        return jsonify({"success": False, "error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/api/llm_batch_predict", methods=["POST"])
def api_llm_batch_predict():
    """批量使用LLM预测（无需训练模型）"""
    try:
        file_obj = request.files.get("file")
        if file_obj:
            raw = file_obj.read()
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        else:
            return jsonify({"success": False, "error": "请上传CSV文件"}), 400

        # 验证必要字段
        required = ["零件号", "零件名称", "包装等级分类", "零件重量（KG）"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            return jsonify({"success": False, "error": f"CSV缺少必要列: {', '.join(missing_cols)}"}), 400

        limit = request.form.get("limit", 10, type=int)
        df_subset = df.head(limit)

        results = []
        for idx, row in df_subset.iterrows():
            try:
                result = llm_predict_single(row.to_dict())
                results.append({
                    "row": int(idx),
                    "零件号": row.get("零件号", ""),
                    "零件名称": row.get("零件名称", ""),
                    "predicted_type": result["predicted_type"],
                    "recommended_size": result["recommended_size"],
                    "effective_dimensions": result.get("effective_dimensions", ""),
                    "reason": result["reason"],
                })
            except Exception as e:
                results.append({"row": int(idx), "error": str(e)})

            # 避免API限流
            if len(results) > 0 and len(results) % 5 == 0:
                time.sleep(1)

        return jsonify(_clean_nan({"success": True, "data": {"total": len(df_subset), "results": results}}))
    except Exception as exc:
        import traceback
        return jsonify({"success": False, "error": str(exc), "trace": traceback.format_exc()}), 500


# ===== 原有API（来自train_server_plus.py）=====

@app.route("/api/train", methods=["POST"])
def api_train():
    try:
        file_obj = request.files.get("file")
        if not file_obj:
            return jsonify({"success": False, "error": "请上传 CSV 训练文件"}), 400
        raw = file_obj.read()
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")

        result = train_model(df)
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


@app.route("/train_llm")
def page_train_llm():
    return send_from_directory(".", "train_llm.html")


@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory("js", filename)


@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory("css", filename)


@app.route("/")
def index():
    return send_from_directory(".", "step1.html")


@app.route("/app.html")
def app_html():
    return send_from_directory(".", "step1.html")


@app.route("/step1.html")
def step1_html():
    return send_from_directory(".", "step1.html")

@app.route("/step2.html")
def step2_html():
    return send_from_directory(".", "step2.html")


@app.route("/step3.html")
def step3_html():
    return send_from_directory(".", "step3.html")


if __name__ == "__main__":
    print(f"当前目录: {os.getcwd()}")
    print(f"LLM配置: {LLM_PROVIDER} / {LLM_MODEL}")
    print(f"API地址: {LLM_BASE_URL}")
    print("训练页面: http://127.0.0.1:5001/train_llm")
    print("前端页面: http://127.0.0.1:5001/")
    print("LLM预测: POST /api/llm_predict")
    print("LLM增强: POST /api/llm_enhance")
    print("LLM批量预测: POST /api/llm_batch_predict")
    app.run(host="0.0.0.0", port=5001, debug=True)
