import numpy as np
from grid2op.Agent import BaseAgent

class DeltaRedispatchRandomAgent(BaseAgent):        
    def __init__(self, action_space,
                 n_gens_to_redispatch=2,
                 redispatching_delta=1.0):
        """
        Agent constructor

        Parameters
        ----------
        :action_space: :class:`grid2op.Action.ActionSpace`
             the Grid2Op action space

        :n_gens_to_redispatch: `int`
          The maximum number of dispatchable generators to play with 

        :redispatching_delta: `float`
          The redispatching MW value used in both directions
        """
        super().__init__(action_space)
        self.desired_actions = []

        # Get all generators IDs
        gens_ids = np.arange(self.action_space.n_gen, dtype=int)
        # Filter out non resipatchable IDs
        gens_redisp = gens_ids[self.action_space.gen_redispatchable]
        # Cut if needed
        if len(gens_redisp) > n_gens_to_redispatch:
            gens_redisp = gens_redisp[0:n_gens_to_redispatch]

        # Register do_nothing action
        self.desired_actions.append(self.action_space({}))

        # Register 2 actions per generator
        # (increase or decrease by the delta)
        for gen_id in gens_redisp:
            # Create action redispatch by opposite delta
            act1 = self.action_space({
                "redispatch": [
                    (gen_id, -float(redispatching_delta))
                ]
            })
            
            # Create action redispatch by delta
            act2 = self.action_space({
                "redispatch": [
                    (gen_id, float(redispatching_delta))
                ]
            })

            # Register this generator actions
            self.desired_actions.append(act1)
            self.desired_actions.append(act2)

        
    def act(self, observation, reward, done=False):
        """
        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the 
            :class:`grid2op.Environment.Environment`
        reward: ``float``
            The current reward. 
            This is the reward obtained by the previous action
        done: ``bool``
            Whether the episode has ended or not. 
            Used to maintain gym compatibility
        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The action chosen by agent.
        """
        
        return self.space_prng.choice(self.desired_actions)
