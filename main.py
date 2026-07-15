# 这是一个示例 Python 脚本。
import os

import pandas as pd


# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def connect_csv():
    # 读取两个 CSV
    df_main = pd.read_csv('data/test_input_total_new.csv')
    df_bom = pd.read_csv('data/test_input_bom.csv')

    # BOM 表按零件名称去重（保留第一条），避免笛卡尔积展开
    df_bom = df_bom.drop_duplicates(subset='零件名称', keep='first')

    # 要从 BOM 表匹配的字段
    # bom_cols = ['零件重量（KG）', '零件种类', 'CKD包装类型', 'CKD包装尺寸L', 'CKD包装尺寸W', 'CKD包装尺寸H', 'CKD_SNP',
    #             '批组箱数']
    # bom_cols = ['零件重量KG', '零件尺寸L', '零件尺寸W', '零件尺寸H', '来料SNP', '来料包装L', '来料包装W', '来料包装H']
    bom_cols = ['CKD 包装尺寸L', 'CKD 包装尺寸L', 'CKD 包装尺寸L', 'CKD SNP']
    # 以"零件名称"为 key，合并到主表
    df_merged = df_main.merge(
        df_bom[['零件名称'] + bom_cols],
        on='零件名称',
        how='left'
    )
    # 保存结果
    df_merged.to_csv('data/test_input_total_new1.csv', index=False, encoding='utf-8-sig')
    print(
        f"合并完成：主表 {len(df_main)} 行 + BOM {len(df_bom)} 行 → 共 {len(df_merged)} 行，已保存为 test_input_total_new.csv")


def separate_csv(file_name):
    # CSV文件路径
    csv_file = file_name  # 修改为你的文件名
    # 读取数据
    df = pd.read_csv(csv_file)

    # 随机抽取30%作为预测数据
    predict_data = df.sample(frac=0.1, random_state=42)

    # 剩余70%作为训练数据
    train_data = df.drop(predict_data.index)

    # 获取原文件所在目录
    file_dir = os.path.dirname(os.path.abspath(csv_file))

    # 保存到原目录
    predict_path = os.path.join(file_dir, "predict_data_AI.csv")
    train_path = os.path.join(file_dir, "train_data_AI.csv")

    predict_data.to_csv(predict_path, index=False)
    train_data.to_csv(train_path, index=False)

    print(f"预测数据已保存：{predict_path}")
    print(f"训练数据已保存：{train_path}")


def filter_valid_parts():
    """读取 train_data_AI.csv，去除零件尺寸/包装尺寸/零件重量为空的行，保存为实际测试.csv"""
    df = pd.read_csv('data/train_data_AI.csv', encoding='utf-8-sig', low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # 待检查的列
    size_cols = ['零件尺寸L', '零件尺寸W', '零件尺寸H']
    pkg_cols = ['CKD 包装尺寸L', 'CKD 包装尺寸L.1', 'CKD 包装尺寸L.2']
    weight_col = '零件重量KG'

    # 标记空值：空白、NaN、'/'、'nan' 字符串 均视为空
    def is_empty(series):
        s = series.astype(str).str.strip()
        return s.eq('') | s.eq('nan') | s.eq('/')

    # 只要任一关键列为空就去除
    mask = pd.Series([True] * len(df))
    for col in size_cols + pkg_cols + [weight_col]:
        if col in df.columns:
            mask &= ~is_empty(df[col])

    df_valid = df[mask].copy()
    df_valid.to_csv('data/实际测试.csv', index=False, encoding='utf-8-sig')

    removed = len(df) - len(df_valid)
    print(f"原始 {len(df)} 行 → 去除 {removed} 行 → 保留 {len(df_valid)} 行")
    print(f"已保存至 data/实际测试.csv")


if __name__ == '__main__':
    # filter_valid_parts()
    df = pd.read_csv('data/H97-LC-C项目包装PFEP.csv')
    df['零件质量要求'] = df['零件质量要求'].astype(str).str.replace('\n', '', regex=False)
    df.to_csv('data/H97-LC-C项目包装PFEP.csv', index=False, encoding='utf-8-sig')
