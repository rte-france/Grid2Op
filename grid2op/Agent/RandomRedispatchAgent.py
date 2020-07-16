from grid2op.Agent import BaseAgent

class RandomRedispatchAgent(BaseAgent):        
    def __init__(self, action_space,
                 n_gens_to_redispatch=2,
                 redispatching_increment=1):
        """
        Initialize agent
        :param action_space: the Grid2Op action space
        :param n_gens_to_Redispatch: 
          the maximum number of dispatchable generators to play with 
        :param redispatching_increment: 
          the redispatching MW value to play with (both Plus or Minus)
        """
        super().__init__(action_space)
        self.desired_actions = []

        gens_ids = np.arange(self.action_space.n_gen)
        gens_redisp = gens_ids[self.action_space.gen_redispatchable == True]
        if len(gens_redisp) > n_gens_to_redispatch:
            gens_redisp = gens_redisp[0:n_gens_to_redispatch]

        # Register do_nothing action
        self.desired_actions.append(self.action_space({}))

        # Register 2 actions per generator
        # (increase or decrease by the increment)
        for i in gens_redisp:
            # Create action redispatch by opposite increment
            act1 = self.action_space({
                "redispatch": [
                    (i, -float(redispatching_increment))
                ]
            })
            
            # Create action redispatch by increment
            act2 = self.action_space({
                "redispatch": [
                    (i, float(redispatching_increment))
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
            The action chosen by the bot / controller / agent.
        """
        
        return self.space_prng.choice(self.desired_actions)
