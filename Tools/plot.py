import seaborn as sns
import matplotlib.pyplot as plt
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-ne", "--num_env", help = "Number of the env",
                    type=int, required = True)
parser.add_argument("-na", "--num_algo", help = "Number of the algorithm",
                    action='append')
parser.add_argument("-s", "--seed", help = "The seed",
                    action='append')
parser.add_argument("-f", "--final", help = "Plot only the final results",
                    action="store_true")
parser.add_argument("-n", "--n_steps", help = "Number of step we consider",
                    type=int)
args = parser.parse_args()

ids = ['cartstemcontact-v0', 'cartstem-v0', "stempendulum-v0",
        "catchtheobject-v0", "multigaitrobot-v0", "abstractmultigait-v0"]
id = ids[args.num_env]

name_algo = ['SAC', 'PPO','PPO_AVEC','SAC_AVEC']
colors = ['blue', 'orange', 'green', 'red']
legend = []

for id_name in args.num_algo:
    data = {"Iteration": [], "Reward": []}
    for seed in args.seed:
        name = name_algo[int(id_name)]+ "_"  + id + "_" + str(int(seed)*10)
        if args.final:
            file = "./Results_benchmark/" +  name + "/final_rewards_"+id+".txt"
        else:
            file = "./Results_benchmark/" +  name + "/rewards_"+id+".txt"

        print(">> Load: ", file)
        with open(file, 'r') as fp:
            loaded_data = json.load(fp)
            data["Reward"]+= loaded_data[0][:args.n_steps]
            data["Iteration"]+= loaded_data[1][:args.n_steps]
    fig = sns.lineplot(x="Iteration", y="Reward", data=data, color = colors[int(id_name)])
    legend.append(name_algo[int(id_name)])

if args.final:
    print(">> Print: final_rewards_"+id+".txt")
else:
    print(">> Print: rewards_"+id+".txt")

fig.legend(tuple(legend))
plt.show()
