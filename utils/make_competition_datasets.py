#!/usr/bin/env python3

import sys
import os
import json
import shutil

import grid2op

def split_ds(indir, outdir, start_d, end_d):
    # Prepare output directories
    chronics_outdir = os.path.join(outdir, "chronics")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(chronics_outdir, exist_ok=True)

    # Copy env files: configs, jsons, ...
    for f in os.listdir(indir):
        f_path = os.path.join(indir, f)
        if os.path.isdir(f_path):
            continue
        shutil.copy(f_path, outdir)

    # Split the chronics
    env = grid2op.make(indir)
    env.chronics_handler.real_data.split_and_save(start_d,
                                                  end_d,
                                                  chronics_outdir)

def test_intervals():
    # Declare datetimes intervals to extract for test set
    start_test = {
        "Scenario_january_28": "2012-01-21 23:55",
        "Scenario_february_40": "2012-02-21 23:55",
        "Scenario_march_07": "2012-03-16 23:55",
        "Scenario_april_42": "2012-04-08 23:55",
        "Scenario_may_17": "2012-05-03 23:55",
        "Scenario_june_01": "2012-06-06 23:55",
        "Scenario_august_01": "2012-08-06 23:55",
        "Scenario_october_21": "2012-10-01 23:55",
        "Scenario_november_34": "2012-11-05 23:55",
        "Scenario_december_12": "2012-12-08 23:55"
    }
    end_test = {
        "Scenario_january_28": "2012-01-25 00:00",
        "Scenario_february_40": "2012-02-25 00:00",
        "Scenario_march_07": "2012-03-20 00:00",
        "Scenario_april_42": "2012-04-14 00:00",
        "Scenario_may_17": "2012-05-07 00:00",
        "Scenario_june_01": "2012-06-10 00:00",
        "Scenario_august_01": "2012-08-10 00:00",
        "Scenario_october_21": "2012-10-04 00:00",
        "Scenario_november_34": "2012-11-09 00:00",
        "Scenario_december_12": "2012-12-14 00:00"
    }
    return start_test, end_test

def validation_intervals():
    # Declare datetimes intervals to extract for validation set
    start_valid = {
        "Scenario_january_32": "2012-01-21 23:55",
        "Scenario_february_20": "2012-02-21 23:55",
        "Scenario_march_39": "2012-03-16 23:55",
        "Scenario_april_19": "2012-04-08 23:55",
        "Scenario_may_24": "2012-05-03 23:55",
        "Scenario_june_14": "2012-06-06 23:55",
        "Scenario_august_02": "2012-08-06 23:55",
        "Scenario_october_05": "2012-10-01 23:55",
        "Scenario_november_46": "2012-11-05 23:55",
        "Scenario_december_16": "2012-12-08 23:55"
    }    
    end_valid = {
        "Scenario_january_32": "2012-01-24 00:00",
        "Scenario_february_20": "2012-02-24 00:00",
        "Scenario_march_39": "2012-03-20 00:00",
        "Scenario_april_19": "2012-04-14 00:00",
        "Scenario_may_24": "2012-05-07 00:00",
        "Scenario_june_14": "2012-06-10 00:00",
        "Scenario_august_02": "2012-08-10 00:00",
        "Scenario_october_05": "2012-10-04 00:00",
        "Scenario_november_46": "2012-11-08 00:00",
        "Scenario_december_16": "2012-12-14 00:00"
    }
    return start_valid, end_valid
    
if __name__ == "__main__":
    dir_input_rel = "/home/tezirg/data_grid2op/case118_l2rpn_neurips_2.5x"
    dir_input = os.path.abspath(dir_input_rel)
    ds = os.path.basename(dir_input)
    dir_test = os.path.join("/tmp", ds, "test")
    dir_valid = os.path.join("/tmp", ds, "validation")
    
    # Create test set
    print("Creating test set..")
    start_test, end_test = test_intervals()
    split_ds(dir_input, dir_test, start_test, end_test)
    print("Test set located at {}".format(dir_test))

    # Create validation set
    print("Creating validation set..")
    start_valid, end_valid = validation_intervals()
    split_ds(dir_input, dir_valid, start_valid, end_valid)
    print("Validation set located at {}".format(dir_valid))

    
