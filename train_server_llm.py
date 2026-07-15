"""
LightGBM 零件包装方案训练服务（LLM增强版）
启动方式: python train_server_llm.py

功能特性：
1. 保留 train_server_plus.py 所有功能
2. 新增 LLM API 增强训练数据/直接预测
3. 支持多模型：DashScope(通义千问)、SiliconFlow(OpenAI兼容)、OpenAI

配置方式（环境变量）：
- LLM_PROVIDER: siliconflow | dashscope | openai (默认: siliconflow)
- LLM_API_KEY: 你的API密钥
- LLM_BASE_URL: 自定义API地址（可选，用于SiliconFlow/OpenAI兼容接口）
- LLM_MODEL: 模型名称（默认: Qwen/Qwen2-7B-Instruct）

使用示例：
  # 1. 直接用LLM预测（无需训练）
  curl -X POST http://127.0.0.1:5001/api/llm_predict \
    -H "Content-Type: application/json" \
    -d '{"零件号":"test","零件名称":"螺钉","包装等级分类":"C","零件重量（KG）":0.5,"零件分类":"采购件"}'

  # 2. 增强训练数据后训练
  curl -X POST http://127.0.0.1:5001/api/llm_enhance \
    -F "file=@data/train_data.csv"
"""

import io
import json
import math
import os
import pathlib
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
TEMPLATE_FILE = "LightGBM_damo_plus.html"
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
_cached_standalone_html = None
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
    print(f"[导出] payload quality_keywords: {_quality_keywords}")


def _render_standalone_html():
    template_path = pathlib.Path(__file__).with_name(TEMPLATE_FILE)
    template = template_path.read_text(encoding="utf-8")
    payload = _make_export_payload()
    model_script = "<script>window.LGB_MODEL = " + json.dumps(payload, ensure_ascii=False) + ";</script>"

    # 材质选择器 HTML（注入到模态框）
    material_selector_html = '''<div style="margin-bottom:10px;display:flex;align-items:center;gap:8px;font-size:13px;">
  <span style="color:var(--gray-2);white-space:nowrap;">包装材料：</span>
  <select id="modalMaterialSelect" onchange="onModalMaterialChange(this.value)" style="border:1px solid #c0c6d1;border-radius:4px;padding:3px 8px;font-size:13px;cursor:pointer;">
    <option value="carton">纸箱</option>
    <option value="wood">木箱</option>
    <option value="iron">铁架/轻钢</option>
    <option value="tdg">天地盖纸箱</option>
    <option value="STD">STD</option>
  </select>
</div>'''

    # 托盘码放信息区域 HTML
    pallet_section_html = '''<div id="palletStackingSection" style="display:none;margin-top:16px;">
  <div class="card-title" style="font-size:14px;"><span class="icon">📦</span> 纸箱托盘码放方案（148A 托盘）</div>
  <div id="palletStackingList" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;margin-top:10px;"></div>
</div>'''

    # Step 2 批量预测汇总表 HTML
    step2_batch_summary_html = '''<div id="step2BatchSummary" style="display:none;margin-top:16px;">
  <div class="card-title" style="font-size:14px;"><span class="icon">📋</span> 预测包装方案汇总</div>
  <div style="margin-bottom:10px;font-size:12px;color:var(--gray-2);">
  </div>
  <div class="batch-table-wrap" style="max-height:420px;overflow-y:auto;">
    <table class="batch-table">
      <thead id="step2BatchSummaryHead"></thead>
      <tbody id="step2BatchSummaryBody"></tbody>
    </table>
  </div>
  <div id="palletDiagramsContainer"></div>
</div>'''

    # 托盘规格（1400×1060×1100 mm），简化为 2/4/8 等分底面积格子
    # 托盘底面积 1480×1060 mm（行业标准 148A），高度 1100 mm
    # 每条记录：[L等分数, W等分数, 格子长, 格子宽]
    # 底面积只有两个朝向： cartonL×cartonW 或 cartonW×cartonL
    PACK_PALLET_DIVS_JS = json.dumps([
        [2, 2, 700, 530],
        [2, 4, 700, 265],
        [4, 2, 350, 530],
        [4, 4, 350, 265],
        [8, 2, 175, 530],
        [8, 4, 175, 265],
    ], ensure_ascii=False)

    # 覆盖 openPkgMethodModal 的 JS（支持切换材质重新计算选项）
    injected_js = """var _currentModalMaterial = 'carton';
var PACK_PALLET_DIVS = """ + PACK_PALLET_DIVS_JS + """;
var PALLET_SIZE = [1400, 1060, 1100];  // 托盘长、宽、高

/**
 * 简化托盘码放计算：纸箱底面积是否满足2/4/8等分尺寸（长宽可旋转）
 * @param {number} cL 纸箱长 mm
 * @param {number} cW 纸箱宽 mm
 * @returns {{fits: boolean, spec: object|null, boxesPerPallet: number}}
 */
function computePalletStacking(cL, cW) {
  var best = null;
  for (var i = 0; i < PACK_PALLET_DIVS.length; i++) {
    var spec = PACK_PALLET_DIVS[i];
    var cellL = spec[2], cellW = spec[3];
    // 底面积两个朝向：L×W 或 W×L
    var fit1 = (cL <= cellL && cW <= cellW);
    var fit2 = (cW <= cellL && cL <= cellW);
    if (fit1 || fit2) {
      var alongL = Math.floor(cellL / cL);
      var alongW = Math.floor(cellW / cW);
      var total = alongL * alongW;
      if (!best || total > best.total) {
        best = { index: i, lDiv: spec[0], wDiv: spec[1], cellL: cellL, cellW: cellW, alongL: alongL, alongW: alongW, total: total };
      }
    }
  }
  if (!best) return { fits: false, spec: null, boxesPerPallet: 0 };
  return { fits: true, spec: best, boxesPerPallet: best.total };
}

/**
 * 更新托盘码放信息区域
 */
function updatePalletStackingSection() {
  var section = document.getElementById('palletStackingSection');
  var list = document.getElementById('palletStackingList');
  if (!section || !list) return;
  var method = state.pkgMethods.find(function(m) { return m.id === state.selectedPkgMethod; });
  if (!method || !method.name || method.name.indexOf('纸箱') === -1) {
    section.style.display = 'none';
    return;
  }
  var result = computePalletStacking(method.pkgL, method.pkgW);
  if (!result.fits) {
    section.style.display = 'block';
    list.innerHTML = '<div style="padding:12px;color:var(--gray-2);font-size:13px;">该纸箱尺寸无法使用 148A 托盘码放</div>';
    return;
  }
  var s = result.spec;
  list.innerHTML = '<div style="background:#f7f8fa;border:1px solid #e8ecf0;border-radius:6px;padding:10px 12px;font-size:12px;line-height:1.8;">' +
    '<b style="font-size:13px;">' + s.lDiv + '×' + s.wDiv + ' 等分</b>（格 ' + s.cellL + '×' + s.cellW + ' mm）<br/>' +
    '每层 <b>' + s.alongL + '×' + s.alongW + ' = ' + s.total + '</b> 箱/托盘' +
  '</div>';
  section.style.display = 'block';
}

// 重写 renderPkgOptions：追加托盘码放信息
var _origRenderPkgOptions = renderPkgOptions;
renderPkgOptions = function() {
  _origRenderPkgOptions.apply(this, arguments);
  updatePalletStackingSection();
};

// 重写 selectPkgMethod：切换选项后更新托盘区域
var _origSelectPkgMethod = selectPkgMethod;
selectPkgMethod = function(el, id) {
  _origSelectPkgMethod.apply(this, [el, id]);
  updatePalletStackingSection();
};

/**
 * 计算一级包装成本
 * 纸箱 4.6 元/m² | 天地盖 5.4 元/m² | 木箱按面积 36 元/m² | 铁箱 7.8 元/kg
 */
function computePkgCost(pkgL, pkgW, pkgH, boxesNeeded, material) {
  var surfaceM2 = 2 * (pkgL * pkgW + pkgL * pkgH + pkgW * pkgH) / 1e6;
  var costPerBox = 0;
  if (material === 'iron') {
    var weight = estimatePkgWeight(pkgL, pkgW, pkgH, null);
    costPerBox = weight * 7.8;
  } else if (material === 'tdg') {
    costPerBox = surfaceM2 * 5.4;
  } else if (material === 'wood') {
    costPerBox = surfaceM2 * 36;
  } else {
    // carton / STD：4.6 元/m²
    costPerBox = surfaceM2 * 4.6;
  }
  return Math.round(costPerBox * boxesNeeded * 100) / 100;
}

/**
 * 检查纸箱是否能放入托盘
 */
function canFitOnPallet(material, pkgL, pkgW, pkgH) {
  if (material !== 'STD' && material !== 'carton' && material !== 'tdg') return false;
  return computePalletStacking(pkgL, pkgW).fits;
}

/**
 * 获取托盘等分规格索引
 */
function getPalletSpecIndex(material, pkgL, pkgW, pkgH) {
  if (!canFitOnPallet(material, pkgL, pkgW, pkgH)) return -1;
  return computePalletStacking(pkgL, pkgW).spec ? computePalletStacking(pkgL, pkgW).spec.index : -1;
}

/**
 * 计算批量预测的托盘分配方案
 * @returns {{pallets: array, seqToPallet: object}}
 */
function computeBatchPalletAllocation(results) {
  var pallets = []; // [{seq:[], spec, cells:[], totalCells, totalBoxes}]
  results.forEach(function(r, idx) {
    if (r.error || !r.pkgMethod) return;
    var pkg = r.pkgMethod;
    var boxesNeeded = pkg.boxesNeeded || 1;
    var pIdx = getPalletSpecIndex(pkg.material, pkg.l, pkg.w, pkg.h);
    if (pIdx < 0) return;
    var spec = computePalletStacking(pkg.l, pkg.w).spec;

    // 把该零件的所有箱子逐个填入托盘（可跨多个托盘）
    var remainingBoxes = boxesNeeded;
    while (remainingBoxes > 0) {
      // 找还有空位的同规格托盘
      var foundPallet = null;
      for (var pi = 0; pi < pallets.length; pi++) {
        var p = pallets[pi];
        if (p.spec.index !== pIdx) continue;
        if (p.totalBoxes < p.totalCells) { foundPallet = p; break; }
      }
      // 没有合适的托盘就新建一个
      if (!foundPallet) {
        var totalCells = spec.lDiv * spec.wDiv;
        var cells = [];
        for (var c = 0; c < totalCells; c++) cells.push(-1);
        foundPallet = { seq: [], spec: spec, cells: cells, totalCells: totalCells, totalBoxes: 0 };
        pallets.push(foundPallet);
      }
      // 逐个填入空位
      for (var ci = 0; ci < foundPallet.totalCells && remainingBoxes > 0; ci++) {
        if (foundPallet.cells[ci] === -1) {
          foundPallet.cells[ci] = idx + 1;
          foundPallet.totalBoxes++;
          if (foundPallet.seq.indexOf(idx + 1) < 0) foundPallet.seq.push(idx + 1);
          remainingBoxes--;
        }
      }
    }
  });

  // 修正 totalBoxes 和去重 seq
  pallets.forEach(function(p) {
    var cnt = {};
    for (var c = 0; c < p.totalCells; c++) {
      var s = p.cells[c];
      if (s > 0) cnt[s] = (cnt[s] || 0) + 1;
    }
    p.totalBoxes = Object.keys(cnt).length > 0 ? Object.values(cnt).reduce(function(a, b) { return a + b; }, 0) : 0;
    var seen = {};
    p.seq = p.seq.filter(function(s) { if (seen[s]) return false; seen[s] = true; return true; });
  });

  var seqToPallet = {};
  pallets.forEach(function(p, pi) {
    p.seq.forEach(function(s) { seqToPallet[s] = pi + 1; });
  });

  return { pallets: pallets, seqToPallet: seqToPallet };
}

/**
 * 渲染 Step 2 批量预测包装方案汇总表（简化版）
 */
function renderStep2BatchSummary() {
  var section = document.getElementById('step2BatchSummary');
  var head = document.getElementById('step2BatchSummaryHead');
  var body = document.getElementById('step2BatchSummaryBody');
  if (!section || !head || !body) return;
  var results = (typeof _batchResults !== 'undefined' && _batchResults) ? _batchResults : [];
  if (results.length === 0) { section.style.display = 'none'; return; }

  var palletL = PALLET_SIZE[0], palletW = PALLET_SIZE[1];
  var alloc = computeBatchPalletAllocation(results);
  var pallets = alloc.pallets;
  var seqToPallet = alloc.seqToPallet;

  var ths = ['序号', '零件号', '零件名称', '单车用量', '总数量', '一级包装类型及尺寸', '批组箱数', 'CKD SNP', 'CKD重量（KG）', '一级包装成本（元）', '是否需要装托盘', '托盘号'];
  head.innerHTML = '<tr>' + ths.map(function(h) { return '<th>' + escapeHtml(h) + '</th>'; }).join('') + '</tr>';

  body.innerHTML = results.map(function(r, idx) {
    var seqNum = idx + 1;
    if (r.error) return '<tr><td colspan="12" style="color:var(--danger);font-size:12px;">行 ' + r.rowIndex + ' 预测错误：' + escapeHtml(r.error) + '</td></tr>';
    var pkg = r.pkgMethod;
    if (!pkg) return '<tr><td colspan="12" style="color:var(--gray-2);font-size:12px;">行 ' + r.rowIndex + ' 无推荐包装方案</td></tr>';
    var partNo = r.rawFeatures['零件号'] || '';
    var partName = r.rawFeatures['零件名称'] || '';
    var usage = parseFloat(r.rawFeatures['单车用量']);
    var totalQty = r.totalParts !== null && r.totalParts !== undefined ? r.totalParts : 1;
    var boxesNeeded = pkg.boxesNeeded;
    var snp = pkg.snp;
    // 零件总数少于箱容量时，SNP即为总数量（只装1箱且装不满）
    var displaySnp = totalQty < snp ? totalQty : snp;
    var partWeight = parseFloat(
      r.rawFeatures['零件重量（KG）'] !== undefined ? r.rawFeatures['零件重量（KG）'] :
      r.rawFeatures['零件重量(KG)'] !== undefined ? r.rawFeatures['零件重量(KG)'] :
      r.rawFeatures['零件重量KG'] !== undefined ? r.rawFeatures['零件重量KG'] : '0'
    ) || 0;
    // CKD重量 = CKD SNP × 单件重量（totalQty < snp时displaySnp即totalQty）
    var ckdWeight = Math.round(displaySnp * partWeight * 100) / 100;
    var cost = computePkgCost(pkg.l, pkg.w, pkg.h, boxesNeeded, pkg.material);
    var palletNeed = seqToPallet[seqNum] !== undefined;
    var palletText = palletNeed ? '<span style="color:var(--success);font-weight:600;">是</span>' : '否';
    var palletNum = palletNeed ? ('托盘' + seqToPallet[seqNum]) : '';
    return '<tr>' +
      '<td style="text-align:center;">' + seqNum + '</td>' +
      '<td>' + escapeHtml(String(partNo)) + '</td>' +
      '<td>' + escapeHtml(String(partName)) + '</td>' +
      '<td>' + (isNaN(usage) ? '—' : usage) + '</td>' +
      '<td>' + totalQty + '</td>' +
      '<td>' + escapeHtml(pkg.name) + '</td>' +
      '<td>' + boxesNeeded + '</td>' +
      '<td>' + displaySnp + '</td>' +
      '<td>' + ckdWeight.toFixed(2) + '</td>' +
      '<td>' + cost.toFixed(2) + '</td>' +
      '<td>' + palletText + '</td>' +
      '<td>' + (palletNum ? '<b style="color:var(--primary);">' + palletNum + '</b>' : '—') + '</td>' +
    '</tr>';
  }).join('');

  renderPalletDiagrams(pallets, results, palletL, palletW);
  section.style.display = 'block';
}

/**
 * 托盘码放示意图：彩色格子 + 每个托盘的零件分布和空间利用率
 * @param {Array} pallets - 托盘数组
 * @param {Array} results - 完整batch结果（含零件号、尺寸）
 * @param {number} palletL - 托盘长 mm
 * @param {number} palletW - 托盘宽 mm
 */
function renderPalletDiagrams(pallets, results, palletL, palletW) {
  var container = document.getElementById('palletDiagramsContainer');
  if (!container) return;
  if (pallets.length === 0) { container.innerHTML = ''; return; }

  var COLORS = ['#4a90e2','#e94b4b','#50c875','#f5a623','#9b59b6','#1abc9c','#e67e22','#34495e','#e91e63','#00bcd4','#8bc34a','#ff5722'];
  var svgW = 320, svgH = 220;
  var padX = 24, padY = 16;
  var innerW = svgW - padX * 2, innerH = svgH - padY * 2 - 20; // 留20px给底部标注

  // 建立 seq → 结果索引的映射
  var seqToResult = {};
  results.forEach(function(r, idx) { seqToResult[idx + 1] = r; });

  container.innerHTML = '<div style="margin-top:20px;"><div class="card-title" style="font-size:14px;margin-bottom:12px;"><span class="icon">🚢</span> 托盘码放示意图（俯视图）</div>' +
    '<div style="font-size:12px;color:#888;margin-bottom:12px;">托盘规格：' + palletL + ' mm × ' + palletW + ' mm × 1100 mm（高）</div>' +
    '<div style="display:flex;flex-wrap:wrap;gap:16px;">' +
    pallets.map(function(p, pi) {
      var spec = p.spec;
      var lDiv = spec.lDiv, wDiv = spec.wDiv;
      var cellL = spec.cellL, cellW = spec.cellW;

      // SVG 尺寸计算
      var scaleX = innerW / palletL;
      var scaleY = innerH / palletW;
      var scale = Math.min(scaleX, scaleY);
      var pw = palletL * scale;   // 实际SVG托盘宽
      var ph = palletW * scale;   // 实际SVG托盘高
      var ox = padX + (innerW - pw) / 2;
      var oy = padY + (innerH - ph) / 2;
      var cellPxW = pw / lDiv;
      var cellPxH = ph / wDiv;

      // 统计每个格子的颜色和箱子数量
      var seqCounts = {};
      p.seq.forEach(function(s) {
        var r = seqToResult[s];
        var boxes = r && r.pkgMethod ? r.pkgMethod.boxesNeeded : 1;
        seqCounts[s] = (seqCounts[s] || 0) + 1;
      });

      // 构建格子 rects
      var rects = '';
      for (var pos = 0; pos < p.totalCells; pos++) {
        var seqNo = p.cells[pos];
        var row = Math.floor(pos / lDiv);
        var col = pos % lDiv;
        var x = ox + col * cellPxW;
        var y = oy + row * cellPxH;

        if (seqNo > 0) {
          var color = COLORS[(seqNo - 1) % COLORS.length];
          var r2 = seqToResult[seqNo];
          var pkgL = r2 && r2.pkgMethod ? r2.pkgMethod.l : cellL;
          var pkgW2 = r2 && r2.pkgMethod ? r2.pkgMethod.w : cellW;
          var fillL = Math.min(1, pkgL / cellL);
          var fillW = Math.min(1, pkgW2 / cellW);
          var boxPxW = cellPxW * fillL;
          var boxPxH = cellPxH * fillW;
          var boxX = x + (cellPxW - boxPxW) / 2;
          var boxY = y + (cellPxH - boxPxH) / 2;
          rects += '<rect x="' + boxX + '" y="' + boxY + '" width="' + boxPxW + '" height="' + boxPxH + '" fill="' + color + '" stroke="#fff" stroke-width="1"/>';
          var fs = Math.min(12, boxPxW / 2, boxPxH / 2);
          if (fs >= 6) {
            rects += '<text x="' + (boxX + boxPxW / 2) + '" y="' + (boxY + boxPxH / 2) + '" text-anchor="middle" dominant-baseline="middle" fill="#fff" font-size="' + fs + '" font-weight="bold">' + seqNo + '</text>';
          }
        } else {
          // 空格子
          rects += '<rect x="' + x + '" y="' + y + '" width="' + cellPxW + '" height="' + cellPxH + '" fill="#f0f0f0" stroke="#ccc" stroke-width="0.5"/>';
        }
      }

      // 托盘底板
      rects = '<rect x="' + ox + '" y="' + oy + '" width="' + pw + '" height="' + ph + '" fill="#fff" stroke="#888" stroke-width="1"/>' + rects;
      // 托盘边框
      rects += '<rect x="' + ox + '" y="' + oy + '" width="' + pw + '" height="' + ph + '" fill="none" stroke="#333" stroke-width="2"/>';

      var svg = '<svg width="' + svgW + '" height="' + svgH + '" style="display:block;background:#fafafa;">' + rects + '</svg>';

      // 托盘利用率 = 货品底面积之和 / 托盘底面积
      var totalBoxes = p.seq.length;
      var totalBoxArea = 0;
      p.seq.forEach(function(s) {
        var r2 = seqToResult[s];
        if (r2 && r2.pkgMethod) {
          totalBoxArea += (r2.pkgMethod.l * r2.pkgMethod.w);
        } else {
          totalBoxArea += (cellL * cellW); // fallback: 用格子尺寸
        }
      });
      var palletArea = palletL * palletW;
      var utilPct = Math.round(totalBoxArea / palletArea * 100);

      // 零件分布文字
      var partInfo = p.seq.map(function(s) {
        var r = seqToResult[s];
        var partNo = r && r.rawFeatures ? (r.rawFeatures['零件号'] || ('零件' + s)) : ('零件' + s);
        var boxes = r && r.pkgMethod ? r.pkgMethod.boxesNeeded : 1;
        var color = COLORS[(s - 1) % COLORS.length];
        return '<span style="display:inline-flex;align-items:center;gap:3px;margin:2px 4px;">' +
          '<span style="width:10px;height:10px;border-radius:2px;background:' + color + ';display:inline-block;flex-shrink:0;"></span>' +
          '<span>' + escapeHtml(String(partNo).slice(-10)) + '×' + boxes + '箱</span>' +
        '</span>';
      }).join('');

      return '<div style="background:#fff;border:1px solid #e0e6f0;border-radius:8px;padding:12px;min-width:260px;">' +
        '<div style="font-size:13px;font-weight:600;margin-bottom:6px;">' +
          '托盘 ' + (pi + 1) + ' ' +
          '<span style="font-weight:normal;color:#666;font-size:11px;">' + lDiv + '×' + wDiv + ' 等分（格 ' + cellL + '×' + cellW + ' mm）</span>' +
          ' <span style="font-weight:normal;color:var(--primary);font-size:12px;">占用率 ' + utilPct + '%</span>' +
        '</div>' +
        svg +
        '<div style="font-size:11px;color:#555;margin-top:6px;line-height:1.8;">' + partInfo + '</div>' +
      '</div>';
    }).join('') + '</div></div>';
}

/**
 * 渲染批量方案汇总（Step 3 批量模式）
 */
function renderBatchSchemeSummary() {
  var panel = document.getElementById('batchSchemePanel');
  var header = document.getElementById('batchSchemeHeader');
  var body = document.getElementById('batchSchemeBody');
  var palletSummary = document.getElementById('batchPalletSummary');
  var costSummary = document.getElementById('batchCostSummary');
  if (!panel || !body || !header || !palletSummary || !costSummary) return;

  var results = (typeof _batchResults !== 'undefined' && _batchResults) ? _batchResults : [];
  if (results.length === 0) { panel.style.display = 'none'; return; }

  // 聚合包装材料规格
  var specs = {}; // key -> {material, name, l, w, h, count, snp}
  var totalPartCount = 0;
  results.forEach(function(r, idx) {
    if (r.error || !r.pkgMethod) return;
    var pkg = r.pkgMethod;
    totalPartCount += (r.totalParts || 1);
    var rowCost = computePkgCost(pkg.l, pkg.w, pkg.h, pkg.boxesNeeded, pkg.material);
    var key = pkg.material + '|' + pkg.l + '|' + pkg.w + '|' + pkg.h;
    if (!specs[key]) {
      specs[key] = {
        material: pkg.material,
        name: pkg.name,
        l: pkg.l,
        w: pkg.w,
        h: pkg.h,
        count: 0,
        rowCostSum: 0,
        snp: pkg.snp || 1,
        rows: []
      };
    }
    specs[key].count += (pkg.boxesNeeded || 1);
    specs[key].rowCostSum += rowCost;
    specs[key].rows.push({ seq: idx + 1, partNo: r.rawFeatures['零件号'] || '', rowCost: rowCost });
  });

  var totalMaterialCost = 0;
  var totalAuxCost = 0;

  // 排序：纸箱→STD→其他（按名称）
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

  if (!rowsHtml) {
    rowsHtml = '<tr><td colspan="6" style="text-align:center;color:#888;">无有效数据</td></tr>';
  }
  body.innerHTML = rowsHtml;

  // 托盘统计
  var alloc = computeBatchPalletAllocation(results);
  var palletCount = alloc.pallets.length;
  palletSummary.innerHTML = '<div class="alert" style="background:#e8f0fe;color:#0052d9;">' +
    '<b>托盘需求：</b>共需 ' + palletCount + ' 个 148A 托盘' +
    '</div>';

  // 费用汇总（辅料为0）
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

// 重写 goStep3：批量模式下展示批量方案汇总
var _origGoStep3 = goStep3;
goStep3 = function() {
  var hasBatch = (typeof _batchResults !== 'undefined' && _batchResults && _batchResults.length > 0);
  if (!hasBatch) {
    // 单零件模式：显示单零件面板，隐藏批量面板
    var singlePanel = document.getElementById('singleSchemePanel');
    var batchPanel = document.getElementById('batchSchemePanel');
    if (singlePanel) singlePanel.style.display = 'block';
    if (batchPanel) batchPanel.style.display = 'none';
    _origGoStep3.apply(this, arguments);
    return;
  }
  var singlePanel = document.getElementById('singleSchemePanel');
  var batchPanel = document.getElementById('batchSchemePanel');
  if (singlePanel) singlePanel.style.display = 'none';
  if (batchPanel) batchPanel.style.display = 'block';
  renderBatchSchemeSummary();
  goStep(3);
};

// 重写 goStep4：step4 已移除，直接跳转 step3
var _origGoStep4 = goStep4;
goStep4 = function() {
  goStep(3);
};

// 批量预测完成后自动渲染 Step 2 汇总表
var _origRunBatchPrediction = runBatchPrediction;
runBatchPrediction = function() {
  _origRunBatchPrediction.apply(this, arguments);
  var pkgOpts = document.getElementById('pkgOptions');
  if (pkgOpts) pkgOpts.style.display = 'none';
  var palletSec = document.getElementById('palletStackingSection');
  if (palletSec) palletSec.style.display = 'none';
  var partInfo = document.getElementById('partInfoSummary');
  if (partInfo) partInfo.style.display = 'none';
  var step2Card = document.querySelector('#step2 .card');
  if (step2Card) {
    var title = step2Card.querySelector('.card-title');
    if (title) title.style.display = 'none';
    var alert = step2Card.querySelector('.alert');
    if (alert) alert.style.display = 'none';
  }
  renderStep2BatchSummary();
};

// 进入 Step 2 / Step 3 时刷新汇总表、切换面板
var _origGoStep = goStep;
goStep = function(step) {
  _origGoStep.apply(this, arguments);
  var hasBatch = (typeof _batchResults !== 'undefined' && _batchResults && _batchResults.length > 0);
  if (step === 2) {
    if (hasBatch) {
      var pkgOpts = document.getElementById('pkgOptions'); if (pkgOpts) pkgOpts.style.display = 'none';
      var palletSec = document.getElementById('palletStackingSection'); if (palletSec) palletSec.style.display = 'none';
      var partInfo = document.getElementById('partInfoSummary'); if (partInfo) partInfo.style.display = 'none';
    }
    renderStep2BatchSummary();
  } else if (step === 3) {
    var singlePanel = document.getElementById('singleSchemePanel');
    var batchPanel = document.getElementById('batchSchemePanel');
    if (hasBatch) {
      if (singlePanel) singlePanel.style.display = 'none';
      if (batchPanel) { batchPanel.style.display = 'block'; renderBatchSchemeSummary(); }
    } else {
      if (singlePanel) singlePanel.style.display = 'block';
      if (batchPanel) batchPanel.style.display = 'none';
    }
  }
};
"""

    # 执行多处替换
    template = template.replace("<!--MODEL_SCRIPT-->", model_script)
    template = template.replace("<!--MODAL_MATERIAL_SELECTOR-->", material_selector_html)
    template = template.replace("/* INJECT_MODAL_JS */", injected_js)
    template = template.replace("<!--PACK_PALLET_SECTION-->", pallet_section_html)
    template = template.replace("<!--STEP2_BATCH_SUMMARY-->", step2_batch_summary_html)
    return template


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
<title>模型训练（LLM增强版）</title>
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
.alert{padding:12px 16px;border-radius:6px;font-size:13px;margin-bottom:16px;background:#e8f0fe;color:#0052d9}
.alert-error{background:#fef0ef;color:#d54941}
.alert-success{background:#e8f8ef;color:#2ba471}
</style>
</head>
<body>
<header class="header"><h1>LightGBM 模型训练（LLM增强版）</h1></header>
<div class="main">
  <div class="card">
    <div class="card-title">LLM 状态</div>
    <div id="llmStatus">检查中...</div>
  </div>
  <div class="card">
    <div class="card-title">选择训练文件</div>
    <div class="alert">
      请上传 CSV 文件，文件必须包含以下特征列和标签列 <b>CKD包装类型</b>。
    </div>
    <input type="file" id="fileInput" accept=".csv" style="margin-bottom:16px;" />
    <div>
      <button class="btn btn-primary" id="trainBtn" onclick="startTrain()">开始训练</button>
      <a class="btn" id="predictLink" href="/predict_llm" style="display:none;margin-left:10px;">打开预测工具</a>
      <a class="btn" id="downloadLink" href="/api/standalone-predict-only" style="display:none;margin-left:10px;">下载离线版</a>
    </div>
    <div id="resultArea"></div>
  </div>
</div>
<script>
async function checkLlMStatus() {
  try {
    const resp = await fetch('/api/llm_status');
    const data = await resp.json();
    if (data.success) {
      const s = data.data;
      document.getElementById('llmStatus').innerHTML = '<div class="alert alert-success">当前LLM: ' + s.provider + ' | 模型: ' + s.model + ' | 连接状态: ' + s.connection + '</div>';
    }
  } catch(e) {
    document.getElementById('llmStatus').innerHTML = '<div class="alert alert-error">LLM状态检查失败: ' + e.message + '</div>';
  }
}
async function startTrain() {
  const file = document.getElementById('fileInput').files[0];
  const form = new FormData();
  if (file) form.append('file', file);
  document.getElementById('trainBtn').disabled = true;
  try {
    const resp = await fetch('/api/train', {method:'POST', body:form});
    const data = await resp.json();
    if (!data.success) {
      document.getElementById('resultArea').innerHTML = '<div class="alert alert-error">训练失败: ' + data.error + '</div>';
      document.getElementById('trainBtn').disabled = false;
      return;
    }
    const r = data.data;
    document.getElementById('resultArea').innerHTML = '<div class="alert alert-success">训练完成！准确率: ' + r.accuracy_pct + '</div>';
    document.getElementById('predictLink').style.display = 'inline-flex';
    document.getElementById('downloadLink').style.display = 'inline-flex';
  } catch(e) {
    document.getElementById('resultArea').innerHTML = '<div class="alert alert-error">请求失败: ' + e.message + '</div>';
    document.getElementById('trainBtn').disabled = false;
  }
}
checkLlMStatus();
</script>
</body>
</html>'''


@app.route("/train_llm")
def page_train_llm():
    return Response(_TRAIN_PAGE_HTML, mimetype="text/html; charset=utf-8")


@app.route("/predict_llm")
def page_predict_llm():
    if _cached_standalone_html is None:
        return '<h2>请先在 <a href="/train_llm">训练页面</a> 完成模型训练</h2>', 404
    resp = Response(_cached_standalone_html, mimetype="text/html; charset=utf-8")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/LightGBM_damo.html")
def legacy_demo_page():
    return send_from_directory(".", "LightGBM_damo.html")


@app.route("/LightGBM_damo_copy.html")
def legacy_demo_page_copy():
    return send_from_directory(".", "LightGBM_damo_copy.html")


@app.route("/LightGBM_damo_plus.html")
def demo_page_plus():
    return send_from_directory(".", TEMPLATE_FILE)


if __name__ == "__main__":
    print(f"当前目录: {os.getcwd()}")
    print(f"LLM配置: {LLM_PROVIDER} / {LLM_MODEL}")
    print(f"API地址: {LLM_BASE_URL}")
    print("训练页面: http://127.0.0.1:5001/train_llm")
    print("预测页面: http://127.0.0.1:5001/predict_llm")
    print("LLM预测: POST /api/llm_predict")
    print("LLM增强: POST /api/llm_enhance")
    print("LLM批量预测: POST /api/llm_batch_predict")
    app.run(host="0.0.0.0", port=5001, debug=True)