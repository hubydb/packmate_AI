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

app = Flask(__name__, static_folder='.')
DATA_FILE = 'train_data.csv'

# ── 全局模型训练结果 ──
train_result = None

def train_model():
    """训练 LightGBM 模型"""
    global train_result

    df = pd.read_csv(DATA_FILE)
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
    feat_names = df.drop('label', axis=1).columns.tolist()
    feat_imp = sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)

    train_result = {
        'num_classes': int(num_classes),
        'class_names': [str(c) for c in le.classes_],
        'num_features': int(X.shape[1]),
        'num_train': int(X_train.shape[0]),
        'num_test': int(X_test.shape[0]),
        'accuracy': round(accuracy, 4),
        'accuracy_pct': f'{accuracy * 100:.2f}%',
        'num_trees': model.num_trees(),
        'feature_importance': [(name, round(float(score, 2)) for name, score in feat_imp[:15])],
    }
    # 将 numpy float 转换为 Python float
    train_result['feature_importance'] = [(name, float(score)) for name, score in feat_imp[:15]]
    train_result['per_class_accuracy'] = get_per_class_accuracy(y_test, y_pred, le.classes_, num_classes)

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


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/LightGBM_damo.html')
def demo_page():
    return send_from_directory('.', 'LightGBM_damo.html')


@app.route('/train_data.csv')
def serve_csv():
    return send_from_directory('.', 'train_data.csv')


@app.route('/api/train', methods=['POST'])
def api_train():
    """触发训练"""
    try:
        result = train_model()
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


if __name__ == '__main__':
    print(f'当前目录: {os.getcwd()}')
    print(f'数据文件: {os.path.exists(DATA_FILE)}')
    print('启动服务: http://127.0.0.1:5000/LightGBM_damo.html')
    app.run(host='0.0.0.0', port=5000, debug=True)