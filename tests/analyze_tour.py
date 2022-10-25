import pandas as pd
import numpy as np

file_path = "./tour-results.csv"

df = pd.read_csv(file_path)  # , index_col=0
print(df.head())
dn = df.to_numpy(na_value=0)


col = {
    "Game":0,
    "Total Day":1,
    "Spawn Rate":2,
    "p1":3,
    "p2":4,
    "p3":5,
    "p4":6,
    "p1 Score":7,
    "p2 Score":8,
    "p3 Score":9,
    "p4 Score":10,
    "Total Score":11,
}


print(dn)

norm_score = np.zeros((dn.shape[0], 4))
norm_score[:, 0] = dn[:, col["p1 Score"]] / dn[:, col["Total Score"]]
norm_score[:, 1] = dn[:, col["p2 Score"]] / dn[:, col["Total Score"]]
norm_score[:, 2] = dn[:, col["p3 Score"]] / dn[:, col["Total Score"]]
norm_score[:, 3] = dn[:, col["p4 Score"]] / dn[:, col["Total Score"]]
print(norm_score)

# avg overall
avg_scores = []
for gr in range(1, 9):
    mask_g1 = dn[:, col["p1"]:col["p4"]+1] == gr
    score_g1 = norm_score[mask_g1]
    avg_scores.append(score_g1.mean())


# for gr in range(8):
#     print(f"Group {gr+1}: {avg_scores[gr]:.2f}")
# print(avg_scores)



# avg spawn 1
spawn_avg_scores = {
    1: [], 5: [], 10: [], 20: [],
}
for spawn_rate in [1, 5, 10, 20]:
    mask_spawn = dn[:, col["Spawn Rate"]] == spawn_rate
    dn_ = dn[mask_spawn]
    norm_score_ = norm_score[mask_spawn]
    for gr in range(1, 9):
        mask_g1 = dn_[:, col["p1"]:col["p4"]+1] == gr
        score_g1 = norm_score_[mask_g1]
        spawn_avg_scores[spawn_rate].append(score_g1.mean())


for spawn_rate in [1, 5, 10, 20]:
    print(f"Spawn Rate: {spawn_rate}")
    for gr in range(8):
        print(f"  Group {gr+1}: {spawn_avg_scores[spawn_rate][gr]:.2f}")
print(spawn_avg_scores)


# Make a dict of all the scores
all_scores = {
    "Spawn Rate":[],
    "Group 1":[],
    "Group 2":[],
    "Group 3":[],
    "Group 4":[],
    "Group 5":[],
    "Group 6":[],
    "Group 7":[],
    "Group 8":[],
}
all_scores["Spawn Rate"].append(-1)
for gr in range(8):
    all_scores[f"Group {gr+1}"].append(avg_scores[gr].round(3))

for spawn_r in [1, 5, 10, 20]:
    all_scores["Spawn Rate"].append(spawn_r)
    for gr in range(8):
        all_scores[f"Group {gr + 1}"].append(spawn_avg_scores[spawn_r][gr].round(3))
df2 = pd.DataFrame(all_scores)
df2.to_csv("tour_avg_scores.csv")


# Create list of commands to recreate tour

mask_g1 = dn[:, col["p1"]:col["p4"]+1] == 1
mask_g1 = mask_g1.sum(axis=1).astype(bool)
dn_ = dn[mask_g1]

all_cmds = []
for idx in range(len(dn_)):
    row = dn_[idx].astype(int)
    base_cmd = f'python main.py --last {row[col["Total Day"]]} ' \
               f'-p1 {row[col["p1"]]} -p2 {row[col["p2"]]} -p3 {row[col["p3"]]} -p4 {row[col["p4"]]} ' \
               f'--spawn {row[col["Spawn Rate"]]} --dump_state --no_gui --game_name {row[col["Game"]]}'
    all_cmds.append(base_cmd)
path_cmds = "tour_recreate_cmds.txt"
with open(path_cmds, "w") as fd:
    for cmd_ in all_cmds:
        fd.write(cmd_ + "\n")

print("Done")
