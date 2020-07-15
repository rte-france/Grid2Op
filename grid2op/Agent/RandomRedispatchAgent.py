from grid2op.Agent import BaseAgent

class RandomRedispatchAgent(BaseAgent):
        
    def __init__(self, action_space,
                 n_gens_to_Redispatch=2,
                 redispatching_increment=1):
        """
        Initialize agent
        :param action_space: the Grid2Op action space
        :param n_gens_to_Redispatch: 
          the maximum number of dispatchable generators to play with 
        :param redispatching_increment: 
          the redispatching MW value to play with (both Plus or Minus)
        """
        super().__init__(self, action_space)
        self.desired_actions = []

        gen_redisp = self.action_space.gen_redispatchable
        #we create a dictionnary of redispatching actions we want to play with
        GensToRedipsatch=[i for i in range(len(gen_redisp)) if gen_redisp[i]]
        if len(GensToRedipsatch) > n_gens_to_Redispatch:
            GensToRedipsatch=GensToRedipsatch[0:n_gens_to_Redispatch]
        
        #action dic will have 2 actions par generator (increase or decrease by the increment) + do nothing
        self.desired_actions.append(self.action_space({}))# do_nothing action
        
        for i in GensToRedipsatch:

            #redispatching decreasing the production by the increment
            act1=self.action_space({"redispatch": [(i,-redispatching_increment)]})
            self.desired_actions.append(act1)
            
            #redispatching increasing the production by the increment
            act2=self.action_space({"redispatch": [(i,+redispatching_increment)]})
            self.desired_actions.append(act2)

        
    def act(self, observation, reward, done=False):
        """
        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment.Environment`
        reward: ``float``
            The current reward. This is the reward obtained by the previous action
        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility
        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The action chosen by the bot / controller / agent.
        """
        
        return self.space_prng.choice(self.desired_actions)
