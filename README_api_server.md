# api_server.py 改造说明

## 一、改造目标

将 `train_server_llm.py` 由**前后端不分离**的服务改造为**前后端解耦**的纯 API 服务：

- 后端只负责模型训练、LLM 调用、模型导出等核心逻辑，不再渲染业务前端页面。
- 前端 `app.html` 独立部署，通过 `/api/model` 异步加载模型 JSON。
- 保留 `/train_llm` 训练上传页面（管理功能），但页面模板放到独立 HTML 文件中。

## 二、文件结构

```
packmate_AI/
├── api_server.py              # 后端 API 服务入口
├── step1.html                 # 步骤1：输入零件参数 / 批量预测
├── step2.html                 # 步骤2：推荐包装方式
├── step3.html                 # 步骤3：包装成本测算
├── js/                        # 前端 JS 文件
│   ├── common.js              # 常量、state、localStorage、工具函数
│   ├── api.js                 # /api/model 加载与 LightGBM JS 预测
│   ├── pack.js                # 包装方案计算（computePkgMethods 等）
│   ├── modal.js               # 包装方式选择弹窗
│   ├── step1.js               # 步骤1 页面逻辑
│   ├── step2.js               # 步骤2 页面逻辑
│   └── step3.js               # 步骤3 页面逻辑
├── train_llm.html             # 训练上传页面（由 /train_llm 返回）
├── train_server_llm.py        # 原始参考实现（功能函数来源）
├── data/train_data_AI.csv     # 训练数据示例
└── README_api_server.md       # 本说明文档
```

## 三、启动方式

```bash
cd /home/uos/PycharmProjects/PythonProject1/packmate_AI
python api_server.py
```

服务监听 `http://0.0.0.0:5001`。

前端已拆分为 3 个独立页面，状态通过 `localStorage`（key: `pkg_state_v1`）跨页面持久化。

## 四、后端 API 接口

| 方法 | 接口 | 说明 |
|------|------|------|
| POST | `/api/train` | 上传 CSV 训练模型 |
| GET  | `/api/result` | 获取训练结果摘要 |
| POST | `/api/predict` | 用训练好的模型预测 |
| GET  | `/api/model` | 导出模型 JSON（供前端加载） |
| GET  | `/api/llm_status` | 检查 LLM 配置与连接状态 |
| POST | `/api/llm_load_history` | 上传历史数据用于软约束推断 |
| POST | `/api/llm_predict` | 直接用 LLM 预测包装方案 |
| POST | `/api/llm_enhance` | 用 LLM 补全缺失标签 |
| POST | `/api/llm_batch_predict` | 批量 LLM 预测 |

## 五、页面路由

| 路由 | 说明 |
|------|------|
| `/` | 返回 `step1.html`（业务前端入口） |
| `/app.html` | 返回 `step1.html`（兼容旧链接） |
| `/step1.html` | 输入零件参数 / 批量预测 |
| `/step2.html` | 推荐包装方式 |
| `/step3.html` | 包装成本测算 |
| `/train_llm` | 返回 `train_llm.html`（训练上传页面） |

## 六、与 train_server_llm.py 的关系

`api_server.py` 完整复用了 `train_server_llm.py` 的核心功能函数，未改动算法逻辑：

- `train_model(df)` —— LightGBM 模型训练
- `_make_export_payload()` —— 模型 JSON 导出
- `_call_llm_api()` / `_build_llm_prompt()` / `llm_predict_single()` —— LLM 调用
- `_find_similar_parts_history()` —— 历史数据软约束查找

**仅移除的内容**：

1. `_render_standalone_html()` 函数（原本将前端 HTML/JS 注入模板）。
2. `/api/standalone-predict-only`、`/predict_llm` 等渲染相关路由。
3. `/LightGBM_damo.html`、`/LightGBM_damo_copy.html`、`/LightGBM_damo_plus.html` 等旧版页面路由。
4. `_TRAIN_PAGE_HTML` 字符串常量，迁移到 `train_llm.html`。

## 七、前端 app.html 的模型加载

`app.html` 在初始化时优先调用 `/api/model` 异步加载模型：

```javascript
async function loadModelFromApi() {
  var resp = await fetch('/api/model');
  var data = await resp.json();
  if (data && data.success && data.data) {
    state.model = data.data;
    return true;
  }
  return false;
}
```

如果后端尚未训练模型，`app.html` 会提示用户先前往 `/train_llm` 完成训练。

## 八、"是否需要装托盘"说明

"是否需要装托盘"完全由前端根据模型预测结果计算：

1. 只有包装材料为 `carton`（纸箱）、`STD`、`tdg`（天地盖）时才可能装托盘。
2. 托盘规格为 148A 标准托盘：底面积 1400 mm × 1060 mm。
3. 将托盘底面积划分为若干等分格子，检查纸箱底面（长×宽，可旋转）能否放入任一格子。
4. 批量预测时，把能装托盘的零件箱子分配进托盘，分配成功则显示“是”和托盘号，否则显示“否”。

## 九、环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | LLM 服务商：`siliconflow` / `dashscope` / `openai` | `siliconflow` |
| `LLM_API_KEY` | API 密钥 | 空 |
| `LLM_BASE_URL` | 自定义 API 地址 | 根据 provider 默认 |
| `LLM_MODEL` | 模型名称 | 根据 provider 默认 |

## 十、使用流程

1. 访问 `http://127.0.0.1:5001/train_llm` 上传训练 CSV。
2. 训练完成后点击“打开预测工具”进入 `http://127.0.0.1:5001/step1.html`。
3. `step1.html` 自动从 `/api/model` 加载模型；数据录入后进入 `step2.html`、`step3.html`，状态通过 localStorage 跨页面保持。

## 十一、注意事项

- `/api/model` 必须在训练完成后才能访问，否则会返回 `{"success": false, "error": "模型尚未训练"}`。
- 旧的 `app.html` 已拆分为 `step1.html` / `step2.html` / `step3.html` + `js/*.js`，通过 `localStorage` 持久化状态；原 `app.html` 文件已删除。
- `LightGBM_damo_plus.html` 是一个已内嵌旧模型的离线文件，不再作为模板使用。
- 如需与旧 HTML 内嵌模型的结果完全一致，需要使用同一份训练数据和同版本的特征工程重新训练。
