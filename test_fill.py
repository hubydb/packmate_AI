# -*- coding: utf-8 -*-
import os, pandas as pd

df_main = pd.read_csv('data/test_input_total_new.csv', encoding='utf-8-sig')
df_dim  = pd.read_csv('data/零件尺寸_97Y_C_局部.csv', encoding='utf-8-sig')

fill_map = {
    '零件重量KG': '零件重量',
    '零件尺寸L':  '零件尺寸L',
    '零件尺寸L.1':'零件尺寸W',
    '零件尺寸L.2':'零件尺寸H',
    '来料包装L':  '来料包装L',
    '来料包装L.1':'来料包装W',
    '来料包装L.2':'来料包装H',
}

src_cols_needed = list(fill_map.values())
merge_df = df_dim[['零件名称'] + src_cols_needed].copy()
rename_map = {src: src + '_src' for src in src_cols_needed}
merge_df = merge_df.rename(columns=rename_map)

merged = df_main.merge(merge_df, on='零件名称', how='left')

filled_count = {}
for tgt, src in fill_map.items():
    src_col = src + '_src'
    mask = (
        (df_main[tgt].isna() | (df_main[tgt] == '')) &
        merged[src_col].notna() &
        (merged[src_col] != '')
    )
    count = int(mask.sum())
    df_main.loc[mask, tgt] = merged.loc[mask, src_col]
    filled_count[tgt] = count

print('填充结果:')
for tgt, cnt in filled_count.items():
    print(f'  {tgt}: {cnt} 条')

df_main.to_csv('data/test_input_total_new_filled.csv', index=False, encoding='utf-8-sig')
print(f'\n已保存到 data/test_input_total_new_filled.csv ({len(df_main)} 行)')