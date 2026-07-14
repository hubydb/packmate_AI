# -*- coding: utf-8 -*-
import pandas as pd

df1 = pd.read_csv('data/test_input_total_new.csv', encoding='utf-8-sig')
df2 = pd.read_csv('data/零件尺寸_97Y_C_局部.csv', encoding='utf-8-sig')

print('df1 rows:', len(df1))
print('df2 rows:', len(df2))
print()

target_cols = ['零件重量KG', '零件尺寸L', '零件尺寸L.1', '零件尺寸L.2',
               '来料包装L', '来料包装L.1', '来料包装L.2']
print('目标列在 df1 中的非空行数:')
for c in target_cols:
    if c in df1.columns:
        na_count = df1[c].isna().sum()
        empty_str = (df1[c] == '').sum() if df1[c].dtype == object else 0
        print(f'  {c}: 非空={df1[c].notna().sum()}, 空字符串={empty_str}')
    else:
        print(f'  {c}: 列不存在')

print()
src_cols = ['零件重量', '零件尺寸L', '零件尺寸W', '零件尺寸H',
            '来料包装L', '来料包装W', '来料包装H']
print('源列在 df2 中的非空行数:')
for c in src_cols:
    if c in df2.columns:
        print(f'  {c}: 非空={df2[c].notna().sum()}')
    else:
        print(f'  {c}: 列不存在')

print()
print('df1 列名:', list(df1.columns))
print()
print('df2 列名:', list(df2.columns))