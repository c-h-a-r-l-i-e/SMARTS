import os
import sys
import evaluate
import numpy as np

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

# TODO: figure out how to display these results!

for algo in ["ppo"]:
    for action in ["discrete", "continuous"]:
        for safety in ["safe", "nosafe"]:
            print(f"{algo} - {action} - {safety}")
            complete_rates = []
            collision_rates = []
            config_file = f"benchmark/agents/experiments/{algo}_{action}_{safety}.yaml"

            base_dir = f"../../results/{algo}/{action}/{safety}/evaluation/"
            checkpoint_folders = os.scandir(base_dir)
            for checkpoint_fold in checkpoint_folders:
                checkpoint = checkpoint_fold.path + "/checkpoint_100/checkpoint-100"
                blockPrint()
                agent_mets = evaluate.main("scenarios/double_merge/cross/", [config_file], checkpoint, ".log/results", num_episodes=1, headless=True)
                enablePrint()

                complete_rate = agent_mets['']['Completion Rate']
                collision_rate = agent_mets['']['Average Collision Rate']

                complete_rates.append(complete_rate)
                collision_rates.append(collision_rate)

            comp_mean = np.mean(complete_rates)
            comp_std = np.std(complete_rates)


            coll_mean = np.mean(collision_rates)
            coll_std = np.std(collision_rates)

            print(f"completion rate = {comp_mean} +- {1.95 * comp_std / np.sqrt(10)}")
            print(f"collision rate = {coll_mean} +- {1.95 * coll_std / np.sqrt(10)}")







