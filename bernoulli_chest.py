# Welcome! This isn't necessarily the prettiest code, nor is it the most efficient, but feel free to refactor/modify it to your heart's content!

import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os
import shutil
import re
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

timestamp = time.time()

N = 200
R = 50000
GR = 100

min_key_count = 0
max_key_count = 99

affix_y = False

default_preset = ["control","3_key_advantage","moms_key","paper_clip","daemons_tail","contract_from_below","rusted_key","gilded_key"] # + ["fix_pile"]
key_gif = ["control"] + [f"{i+1}_key_advantage" for i in range(GR)]

preset_used = default_preset
save_as_GIF = True
calc_rsq = False

# options: "MAX", "LAST", "TIME_BANKRUPT", "TOTAL_C", "TOTAL_K", "BOTH_TOTALS"
fVal_types = ["MAX","LAST","TIME_BANKRUPT","BOTH_TOTALS"]

real_values = {
    "control": {
        "MAX": [0,11,7,5,8],
        "LAST": [0,11,3,4,2],
        "TIME_BANKRUPT": [200,37,120,82,92]
    },
    "3_key_advantage": {
        "MAX": [12,27,24,15,12],
        "LAST": [11,27,24,5,5],
        "TIME_BANKRUPT": [0,0,75,0,0]
    },
    "moms_key": {
        "MAX": [22,23,32,22,37],
        "LAST": [22,23,32,22,36],
        "TIME_BANKRUPT": [3,8,0,23,0]
    },
    "paper_clip": {
        "MAX": [39,35,41,40,33],
        "LAST": [39,35,41,40,33],
        "TIME_BANKRUPT": [1,10,7,2,3]
    },
    "daemons_tail": {
        "MAX": [20,20,28,16,14],
        "LAST": [19,20,28,12,13],
        "TIME_BANKRUPT": [38,16,11,94,2]
    },
    "contract_from_below": {
        "MAX": [5,12,4,16,10],
        "LAST": [4,2,2,8,9],
        "TIME_BANKRUPT": [97,30,159,47,57]
    },
    "rusted_key": {
        "MAX": [29,23,35,10,27],
        "LAST": [29,19,32,10,27],
        "TIME_BANKRUPT": [1,21,26,21,1]
    },
    "gilded_key": {
        "MAX": [1,1,5,1,2],
        "LAST": [0,0,0,0,0],
        "TIME_BANKRUPT": [199,199,166,197,189]
    }
}

try:
    shutil.rmtree("figs")
except Exception as e:
    pass

os.mkdir("figs")

first_full_lift = 0
bin_caps = [80, 80, 80, 80, 120, 200]
counts_below_N = [[] for _ in range(0,4)]

for var in preset_used:

    clean_var = var.replace("_"," ").upper()
    print(clean_var)

    initial_key_count = 0

    fix_pile = False
    moms_key = False
    paper_clip = False
    daemons_tail = False
    contract_from_below = False
    rusted_key = False
    gilded_key = False

    try:
        initial_key_count = int(re.findall(r"\d+", var)[0])
    except Exception as e:
        exec(f"{var} = True")

    # Yes, I know this is TERRIBLE practice, but it's concise, so I'm rolling with it :p

    gold_chance = 1 if gilded_key else 0.5
    keychest_chance = 0.1 if gilded_key else 0.15
    I = 2 if contract_from_below else 1

    last_vals = []
    maxVals = []
    minVals = []
    totalC_vals = []
    totalK_vals = []
    bankrupt_times = []

    max_val = min_key_count

    for rep in tqdm(range(R)):
        time_bankrupt = 0
        chests_opened = 0
        total_keys = initial_key_count+0

        gold_pile = 0
        key_count = initial_key_count+0
        key_counts = []

        for i in range(N):
            golden = random.random() <= gold_chance

            r = random.random()
            if r >= 0.22 and ((not contract_from_below) or (contract_from_below and random.random() >= 0.33)):
                r = r - 0.22

                if rusted_key and random.random() <= 0.1:
                    key_count += I
                    total_keys += I
                elif r >= 0.33:
                    if golden:
                        gold_pile += I
                        if fix_pile:
                            gold_pile = min(gold_pile, I)
                    else:
                        if random.random() <= 0.78:
                            for n in range(2 if moms_key else 1):
                                for p in range(random.randrange(2,4)):
                                    if random.random() <= keychest_chance or (daemons_tail and random.random() <= (keychest_chance*0.8)):
                                        I2 = 2 if (rusted_key and random.random() <= 1/3) else 1
                                        key_count += I2
                                        total_keys += I2

                    while gold_pile > 0 and key_count > min_key_count:
                        if not paper_clip:
                            key_count -= 1

                        gold_pile -= 1
                        chests_opened += 1
                        if random.random() <= 0.58:
                            for n in range(2 if moms_key else 1):
                                for p in range(random.randrange(2,7)):
                                    if random.random() <= keychest_chance or (daemons_tail and random.random() <= (keychest_chance*0.8)):
                                        I2 = 2 if (rusted_key and random.random() <= 1/3) else 1
                                        key_count += I2
                                        total_keys += I2
                        pass
                else:
                    if random.random() <= (0.28 if rusted_key else 0.2) or (daemons_tail and random.random() <= ((0.28 if rusted_key else 0.2)*0.8)):
                        key_count += I
                        total_keys += I

            if key_count == 0:
                time_bankrupt += 1

            key_count = min(key_count, max_key_count)
            key_counts.append(key_count)

        if rep <= 500:
            plt.plot(range(N), key_counts)
        final_val = key_counts[-1]
        max_val = max(max_val, final_val)

        last_vals.append(key_counts[-1])
        maxVals.append(max(key_counts))
        minVals.append(min(key_counts))
        totalC_vals.append(chests_opened)
        totalK_vals.append(total_keys)
        bankrupt_times.append(time_bankrupt)

    plt.title(f"{clean_var} - Key Count vs Time (t={N}, {R} repetitions)")
    plt.xlabel("Time")
    plt.ylabel("Key Count")

    if not os.path.isdir(f"figs/{var.replace("_"," ")}"): os.mkdir(f"figs/{var.replace("_"," ")}")

    plt.ylim(0, max([25] + maxVals))
    plt.savefig(f"figs/{var.replace("_"," ")}/raw.png")
    plt.clf()

    for i,fvt in enumerate(fVal_types):

        if fvt == "LAST":
            final_vals = last_vals
            if min(minVals) > 0 and save_as_GIF and first_full_lift == 0:
                first_full_lift = initial_key_count
            for i in range(0,4):
                counts_below_N[i].append(sum([1 if k <= i else 0 for k in minVals])/len(minVals))

        elif fvt == "MAX":
            final_vals = maxVals
        elif fvt == "TOTAL_C":
            final_vals = totalC_vals
        elif fvt == "TOTAL_K":
            final_vals = totalK_vals
        elif fvt == "BOTH_TOTALS":
            final_vals = totalK_vals
        elif fvt == "TIME_BANKRUPT":
            final_vals = bankrupt_times

        final_vals = sorted(final_vals)

        bin_caps[i] = max(bin_caps[i], max(bankrupt_times) if fvt == "TIME_BANKRUPT" else (max(totalC_vals + totalK_vals) if fvt == "BOTH_TOTALS" else max(last_vals + maxVals)))
        bin_cap = bin_caps[i]

        # bin_align = [uv - 0.5 for uv in sorted(list(set(final_vals)))] + [sorted(list(set(final_vals))).pop() + 0.5]
        bin_align = [uv - 0.5 for uv in range(bin_cap)] + [bin_cap + 0.5]

        if affix_y: plt.ylim(0, 1)

        if fvt == "BOTH_TOTALS":
            plt.hist(totalC_vals, bins=bin_align, density=True, alpha=0.7, color="skyblue")
            plt.hist(totalK_vals, bins=bin_align, density=True, alpha=0.7, color="salmon")
            plt.axvline(sum(totalC_vals)/len(totalC_vals), color='blue', linestyle='dashed', linewidth=1, label=f"Mean Chests Opened: {round(sum(totalC_vals)/len(totalC_vals),2)}")
            plt.axvline(sum(totalK_vals)/len(totalK_vals), color='red', linestyle='dashed', linewidth=1, label=f"Mean Total Keys Obtained: {round(sum(totalK_vals)/len(totalK_vals),2)}")
        else:
            plt.hist(final_vals, bins=bin_align, density=True, alpha=0.7, edgecolor='black', linewidth=1.2*30/len(bin_align))
            try:
                for i,pt in enumerate(real_values[var][fvt]):
                    if not save_as_GIF:
                        plt.axvline(pt, color='black', linestyle='dashed', linewidth=0.5, label=f"Attempt {i}: {pt}")
            except Exception as e:
                plt.axvline(sum(final_vals)/len(final_vals), color='blue', linestyle='dashed', linewidth=1, label=f"Mean: {round(sum(final_vals)/len(final_vals),2)}")
                plt.axvline(final_vals[len(final_vals)//2], color='red', linestyle='dashed', linewidth=1, label=f"Median: {final_vals[len(final_vals)//2]}")

        plt.legend()
        plt.title(f"{clean_var} - Histogram of {fvt.capitalize()} (t={N}, {R} repetitions)")
        plt.xlabel("Time Bankrupt" if fvt == "TIME_BANKRUPT" else "Key Count")
        plt.ylabel("Probability (normalized)")
        plt.savefig(f"figs/{var.replace("_"," ")}/{fvt.lower()}.png")
        plt.clf()

    canvas = Image.new('RGB', (640*len(fVal_types)+1, 480))
    for x,image in enumerate([f"figs/{var.replace("_"," ")}/{fvt.lower()}.png" for fvt in (["raw"] + fVal_types)]):
        canvas.paste(Image.open(image), (640*x, 0))
    canvas.save(f"figs/{var.replace("_"," ")}/{var}_spread.png")

    if calc_rsq:

        x = np.array([i for i in range(N)]).reshape(-1,1)
        y = np.array(key_counts)

        model = LinearRegression()
        model.fit(x,y)

        y_pred = model.predict(x)

        lr_score = r2_score(y, y_pred)
        
        print(var, "R2 score: " + str(lr_score))

folders = [f.replace("_"," ") for f in preset_used]
subfolders = ["raw", "last", "max", "both_totals", "time_bankrupt"]

canvas = Image.new('RGB', (640*len(subfolders), 480*len(folders)))

pixs = [[] for _ in subfolders]

for y,folder in enumerate(folders):
    for x,subfolder in enumerate(subfolders):
        pix = Image.open(f"figs/{folder}/{subfolder}.png")
        pixs[x].append(pix)
        canvas.paste(pix, (640*x, 480*y))
        # print(subfolder)

try:
    canvas.save("full_canvas.png")
except Exception as e:
    pass

for i, images in enumerate(pixs):
    if save_as_GIF:
        images[0].save(
            f"canvas_animated_{subfolders[i]}.gif",
            save_all=True,
            append_images=images[1:],
            duration=100,  # 100 milliseconds per frame
            loop=0,
        )

if save_as_GIF: print(f"First Full Lift: {first_full_lift}")

for i in range(0,4):
    plt.plot(range(first_full_lift+1), counts_below_N[i][:(first_full_lift+1)], label=f"N = {i}")

plt.xlabel("Starting Key Counts")
plt.ylabel("Probability")
plt.title(f"Percentage of Min Values above N")
plt.axvline(first_full_lift, color='black', linestyle='dashed', linewidth=1, label=f"First Full Lift: {first_full_lift}")
for i in range(0,11,1):
    plt.axhline(i/10, color='black', linewidth=1, alpha=0.4)
for i in range(0,first_full_lift,5):
    plt.axvline(i, color='black', linewidth=1, alpha=0.2)
plt.legend()

if save_as_GIF: plt.savefig(f"canvas_N_counts.png")
plt.clf()

print(str(time.time() - timestamp) + " seconds")