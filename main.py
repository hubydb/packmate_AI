# 这是一个示例 Python 脚本。
import pandas as pd


# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def connect_csv():
# 读取两个 CSV
    df_main = pd.read_csv('test_input.csv')
    df_bom = pd.read_csv('test_input_bom.csv')

    # BOM 表按零件名称去重（保留第一条），避免笛卡尔积展开
    df_bom = df_bom.drop_duplicates(subset='零件名称', keep='first')

    # 要从 BOM 表匹配的字段
    bom_cols = ['零件重量（KG）', '零件种类', 'CKD包装类型', 'CKD 包装尺寸L', 'CKD 包装尺寸W', 'CKD 包装尺寸H', 'CKD SNP',
                '批组箱数']
    # 以"零件名称"为 key，合并到主表
    df_merged = df_main.merge(
        df_bom[['零件名称'] + bom_cols],
        on='零件名称',
        how='left'
    )
    # 保存结果
    df_merged.to_csv('test_input_total.csv', index=False, encoding='utf-8-sig')
    print(
        f"合并完成：主表 {len(df_main)} 行 + BOM {len(df_bom)} 行 → 共 {len(df_merged)} 行，已保存为 test_input_total.csv")


if __name__ == '__main__':


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
