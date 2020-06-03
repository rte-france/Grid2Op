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
        "Scenario_januray_00": "2012-01-01 00:00",
        "Scenario_february_00": "2012-02-01 00:00",
        "Scenario_march_00": "2012-03-01 00:00",
        "Scenario_april_00": "2012-04-01 00:00",
        "Scenario_may_00": "2012-05-01 00:00",
        "Scenario_june_00": "2012-06-01 00:00",
        "Scenario_april_00": "2012-04-01 00:00",
        "Scenario_august_00": "2012-08-01 00:00",
        "Scenario_november_00": "2012-11-01 00:00",
        "Scenario_december_00": "2012-12-01 00:00"
    }    
    end_test = {
        "Scenario_januray_00": "2012-01-03 00:00",
        "Scenario_february_00": "2012-02-03 00:00",
        "Scenario_march_00": "2012-03-03 00:00",
        "Scenario_april_00": "2012-04-03 00:00",
        "Scenario_may_00": "2012-05-03 00:00",
        "Scenario_june_00": "2012-06-03 00:00",
        "Scenario_april_00": "2012-04-03 00:00",
        "Scenario_august_00": "2012-08-03 00:00",
        "Scenario_november_00": "2012-11-03 00:00",
        "Scenario_december_00": "2012-12-03 00:00"
    }
    return start_test, end_test

def validation_intervals():
    # Declare datetimes intervals to extract for validation set
    start_valid = {
        "Scenario_januray_00": "2012-01-01 00:00",
        "Scenario_february_00": "2012-02-01 00:00",
        "Scenario_march_00": "2012-03-01 00:00",
        "Scenario_april_00": "2012-04-01 00:00",
        "Scenario_may_00": "2012-05-01 00:00",
        "Scenario_june_00": "2012-06-01 00:00",
        "Scenario_april_00": "2012-04-01 00:00",
        "Scenario_august_00": "2012-08-01 00:00",
        "Scenario_november_00": "2012-11-01 00:00",
        "Scenario_december_00": "2012-12-01 00:00"
    }    
    end_valid = {
        "Scenario_januray_00": "2012-01-03 00:00",
        "Scenario_february_00": "2012-02-03 00:00",
        "Scenario_march_00": "2012-03-03 00:00",
        "Scenario_april_00": "2012-04-03 00:00",
        "Scenario_may_00": "2012-05-03 00:00",
        "Scenario_june_00": "2012-06-03 00:00",
        "Scenario_april_00": "2012-04-03 00:00",
        "Scenario_august_00": "2012-08-03 00:00",
        "Scenario_november_00": "2012-11-03 00:00",
        "Scenario_december_00": "2012-12-03 00:00"
    }
    return start_valid, end_valid
    
if __name__ == "__main__":
    dir_input_rel = "/home/tezirg/data_grid2op/case118_l2rpn_wcci_48years"
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

    
