import os
import sys
import json

import numpy as np

def evaluate_submission_score_as_codalab():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    print('input_dir: {}'.format(input_dir))
    print('output_dir: {}'.format(output_dir))

    submit_dir = os.path.join(input_dir)
    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    score = 0
    duration = 0
    n_gen = 5
    n_load = 11
    n_lines = 20

    input_dir = os.path.join(input_dir, "saved_experiment")
    if os.path.exists(input_dir):

        for el in os.listdir(input_dir):
            if not os.path.isdir(os.path.join(input_dir, el)):
                continue
            if not os.path.exists(os.path.join(input_dir, el, "episode_meta.json")):
                continue
            if not os.path.exists(os.path.join(input_dir, el, "episode_times.json")):
                continue

            with open(os.path.join(input_dir, el, "episode_meta.json"), "r") as f:
                meta = json.load(f)
            with open(os.path.join(input_dir, el, "episode_times.json"), "r") as f:
                timings = json.load(f)
            tmp_sc = float(meta["cumulative_reward"]) * (int(meta["nb_timestep_played"]) == int(meta["chronics_max_timestep"]))

            print("Score for scenario {}: {}".format(el, tmp_sc))
            score += tmp_sc
            duration += float(timings["Agent"]["total"])
    else:
        print("Your submission is not valid.")
        score = -1
        duration = 99999
        raise RuntimeError("Nothing found at {} where the data should be located".format(input_dir))

    output_filename = os.path.join(output_dir, 'scores.txt')
    with open(output_filename, 'w') as f:
        f.write("score: {:.6f}\n".format(score))
        f.write("duration: {:0.6f}\n".format(duration))
        f.close()


if __name__ == "__main__":
    evaluate_submission_score_as_codalab()
    print("Your submission is valid, you may proceed with the next steps")

