./inspect_action_space.py --path_data ~/data_grid2op/rte_case14_realistic

./train.py  --num_pre_steps 256 --num_train_steps 131072 --path_data ~/data_grid2op/rte_case14_realistic/ --name dqn-XX.0.0.0 --num_frames 4

./eval.py --path_data ~/data_grid2op/rte_case14_realistic/ --path_model ./dqn-XX.0.0.0 --path_logs ./logs_eval --nb_episode 5

rm -rf ./logs-dqn-TopologyChangeAndDispatch.0.0.1/ && ./eval.py --path_data ~/data_grid2op/rte_case14_realistic/ --path_model ./dqn-TopologyChangeAndDispatch.0.0.1.h5 --path_logs ./logs-dqn-TopologyChangeAndDispatch.0.0.1/ --nb_episode 10 --max_steps=8000 --verbose

