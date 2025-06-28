import pandas as pd
import matplotlib.pyplot as plt

result_file = "C:\\Users\\JSW\\Desktop\\resnet_results\\ablation-model-lambda.csv"

# [cfg.log.split('-')[0], init_acc, cfg.prune_rate, cfg.theta, cfg.delta, cfg.k, prune_info[-1], best_acc, best_epoch, round(best_accuracy - accuracy_init, 3)]
COLUMN_SCHEMA = [
    ("Weight_id", "int32"),
    ("Init_Acc", "float64"),
    ("Rate", "float32"),
    ("lambda", "float32"),
    ("delta", "int32"),
    ("k", "int32"),
    ("Pruned_Acc", "float64"),
    ("Best_Acc", "float64"),
    ("Best_epoch", "float64"),
    ("Increase", "float64")
]

df = pd.read_csv(result_file, header=None,
                names=[col[0] for col in COLUMN_SCHEMA]).astype(
                    {col[0]: col[1] for col in COLUMN_SCHEMA}
                )
print(df.head(2))
# df = df.iloc[:, :-1]
# print(df.head(2))
key = "lambda"; label = "λ"
# key = "Rate"; label = "rate"
df_group = df.groupby([key, "Weight_id"], as_index=False).agg(
    {"Init_Acc": 'first', "Pruned_Acc": 'mean', "Best_Acc": 'mean', "Increase": 'mean'})
print(df_group)

colors = ['blue', 'green', 'yellow', 'orange', 'red']
# plt.figure(figsize=(8, 6))
# label: lambda, x: model, y: accuracy
keys = list(set(df_group[key].values))
keys.sort()
for i, value in enumerate(keys):
    sub_df = df_group[df_group[key] == value]
    print(sub_df)
    value =round(value, 2) if value < 1 else int(value)
    plt.plot(sub_df['Weight_id'], sub_df['Increase'], label=f'{label}={value}', color=colors[i], linewidth=1, marker='o')
    # plt.plot(x, [label] * len(y), color=color, linestyle='--')

plt.xticks(ticks=list(set(df_group['Weight_id'])))
plt.legend()
# plt.title("Multiple Line Segments", fontsize=16)
plt.xlabel("Overfitting model(%)")  # , fontsize=12
plt.ylabel("Accuracy increase")  # , fontsize=12

plt.grid(True)  # 显示网格
# plt.savefig(result_file[:result_file.rfind('.')] + '.png')
# plt.show()
