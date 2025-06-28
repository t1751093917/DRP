import pandas as pd

file_name = "Boosting"
result_file = f"C:\\Users\\JSW\\Desktop\\results\\{file_name}.csv"

COLUMN_SCHEMA = [
    ("Init_Acc", "float64"),
    ("Method", "str"),
    ("Rate", "str"),
    ("Pruned_Acc", "float64"),
    ("Best_Acc", "float64"),
    ("Best_epoch", "int32"),
    ("tag", "str")
]

df = pd.read_csv(result_file, header=None,
                names=[col[0] for col in COLUMN_SCHEMA]).astype(
                    {col[0]: col[1] for col in COLUMN_SCHEMA}
                )
print(df.head(2))
# df = df.iloc[:, :-1]
print(df.head(2))
df_group = df.groupby(["tag", "Rate", "Method"], as_index=False).agg(
    {"Init_Acc": 'first', "Pruned_Acc": 'mean', "Best_Acc": 'mean', "Best_epoch": 'first'})[df.columns]
print(df_group.head(5))

save_file = f"C:\\Users\\JSW\\Desktop\\results\\{file_name}_group.csv"
df_group.to_csv(save_file, index=False, header=True, encoding='utf-8-sig')

# import time
#
# print(time.time())
