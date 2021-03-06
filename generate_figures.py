import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.style as style
import os

"""
    function to import data from saved CSV files and generate plots for papers.

    -- Sean Morrison, 2018
"""

style.use("seaborn-deep")
curr_dir = os.getcwd()
directory = curr_dir + "/data"

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Policy Return")
ax.set_xlabel("Episodes")
ax.set_ylabel("Reward")
fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.3)

# loop through data folder and plot data
for f in os.listdir(directory):
    if f.endswith(".csv"):
         print(os.path.join(directory, f))
         data = pd.read_csv(os.path.join(directory, f))
         name = f.split(".")[0]
         print(name)
         xs = data["episode"]
         ys = data["reward"]
         ax.plot(xs, ys, label=name)
plt.legend()         
plt.show()
fig.savefig(curr_dir + "/figures/reward_gain.pdf", bbox_inches="tight")
print("Figure saved")
