import os
import sys
import evaluate
import numpy as np
import json
import pickle


results = {}
for algo in ["ppo"]:
    for action in ["continuous"]:
        for safety in ["safe"]:
            print(f"{algo} - {action} - {safety}")
            complete_rates = []
            collision_rates = []
            config_file = f"benchmark/agents/experiments/{algo}_{action}_{safety}.yaml"

            base_dir = f"../../results/{algo}/{action}/{safety}/evaluation/"
            checkpoint_folders = os.scandir(base_dir)
            for checkpoint_fold in checkpoint_folders:
                checkpoint = checkpoint_fold.path + "/checkpoint_100/checkpoint-100"
                agent_mets = evaluate.main("scenarios/double_merge/cross/", [config_file], checkpoint, ".log/results", num_episodes=10, headless=True)

                complete_rate = agent_mets['']['Completion Rate']
                collision_rate = agent_mets['']['Average Collision Rate']

                print("complete rate: {}".format(complete_rate))
                print("collision rate: {}".format(collision_rate))

                complete_rates.append(complete_rate)
                collision_rates.append(collision_rate)

            comp_mean = np.mean(complete_rates)
            comp_std = np.std(complete_rates)

            coll_mean = np.mean(collision_rates)
            coll_std = np.std(collision_rates)

            comp_str = f"completion rate = {comp_mean} +- {1.95 * comp_std / np.sqrt(10)}"
            coll_str = f"collision rate = {coll_mean} +- {1.95 * coll_std / np.sqrt(10)}"
            results[f"{algo} - {action} - {safety}"] = comp_str, coll_str

            print(comp_str)
            print(coll_str)


print(json.dumps(results, sort_keys=True, indent=4))

with open("eval.pickle", "wb") as f:
    pickle.dump(results, f)


