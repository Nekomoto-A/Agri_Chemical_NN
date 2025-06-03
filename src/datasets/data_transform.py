import pandas as pd

asv_data = pd.read_csv('/home/nomura/Agri_Chemical_NN/data/raw/taxon_data/feature-table.tsv',sep='\t',index_col='OTU ID').T
print(asv_data)

# 各列の 0 の割合を計算
zero_ratio = (asv_data == 0).sum() / len(asv_data)

# 0 の割合が 99%以上の列を除外
df_filtered = asv_data.loc[:, zero_ratio < 0.90]
df_filtered = df_filtered.reset_index()
df_filtered.rename(columns={'OTU ID': 'index'}, inplace=True)
print(df_filtered)
#non_float_columns = df_filtered.select_dtypes(exclude='float').columns
#print(df_filtered.dtypes)
#print(non_float_columns)
#print(df_filtered.select_dtypes(include=').columns)
df_filtered.to_csv('/home/nomura/Agri_Chemical_NN/data/raw/taxon_data/lvasv.csv')
