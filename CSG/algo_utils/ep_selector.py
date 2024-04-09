
class AllowAllEpisodes():
    
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def select(self, *args, **kwargs):
        return True
    
    def update_conditionals(self, *args, **kwargs):
        pass
    def get_stats(self):
        return {}
    
    def flush_data(self):
        pass


class AllowNoEpisodes():
    
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def select(self, *args, **kwargs):
        return False
    
    def update_conditionals(self, *args, **kwargs):
        pass
    def get_stats(self):
        return {}
    def flush_data(self):
        pass

    
    
class AllowUniqueEpisodes(AllowAllEpisodes):
    
    def __init__(self, config) -> None:
        self.expr_limit = config.EXPRESSION_LIMIT
        self.expr_list = [None, ] * self.expr_limit
        self.expr_idx = 0
    
    def flush_data(self):
        
        self.expr_list = [None, ] * self.expr_limit
        self.expr_idx = 0
        
    def select(self, info, update_expr=True, *args, **kwargs):
        # avoid repeats though
        
        expr_set = set(self.expr_list)
        expression_str = self.get_expr_str(info)
        expression_conditional = not expression_str in expr_set
        if expression_conditional:
            if update_expr:
                self.add_expr(info)
            return True
        else:
            return False
    def get_expr_str(self, info, use_im=False):
        
        expression = info['predicted_expression']
        expression_str = "".join(expression)
        if use_im:
            expr = hash(bytes(info['target_canvas'].data))
            expression_str += expr
        return expression_str
        
    def get_stats(self):
        stats = {
            'selector_expr_size': len(self.expr_list)
        }
        return stats
    def add_expr(self, info):
        
        expression_str = self.get_expr_str(info)
        self.expr_list[self.expr_idx] = expression_str
        self.expr_idx = (self.expr_idx + 1) % self.expr_limit


class AllowUniqueIDs(AllowUniqueEpisodes):
    
        
    def get_expr_str(self, info, use_im=False):
        
        expression = (info['slot_id'], info['target_id'])
        return expression

class AllowUniqueEpisodeAndID(AllowUniqueEpisodes):
    
        
    def get_expr_str(self, info, use_im=False):
        
        expression = info['predicted_expression']
        expression = "".join(expression)
        expression_str = expression + str(info['slot_id']) + str(info['target_id'])
        return expression_str

class RewardBasedSelector(AllowUniqueEpisodes):
    
    def __init__(self, config) -> None:
        
        super(RewardBasedSelector, self).__init__(config)
        # Reward Config:
        self.init_reward_thres = config.INIT_REWARD_THRES
        self.reward_thres = self.init_reward_thres
        self.max_reward_thres = config.MAX_REWARD_THRES
        self.reward_thres_multiplier = config.REWARD_THRES_MULTIPLIER
        self.reward_moving_coef = config.REWARD_MOVING_COEF
        
    
    def select(self, info, reward, *args, **kwargs):
        expression_conditional = super(RewardBasedSelector, self).select(info, update_expr=False)
        reward_conditional = reward > self.reward_thres
        if expression_conditional and reward_conditional:
            self.add_expr(info)
            return True
        else:
            return False
    
    def get_stats(self):
        stats = {
            'reward_thres': self.reward_thres,
            'selector_expr_size': len(self.expr_list)
        }
        return stats
        
        
    def update_conditionals(self, *args, **kwargs):
        self.update_reward_thres(*args, **kwargs)
    
    def update_reward_thres(self, rollout_avg_reward, *args, **kwargs):
        new_reward_thres = self.reward_thres_multiplier * rollout_avg_reward
        new_reward_thres = min(max(self.init_reward_thres, new_reward_thres), self.max_reward_thres)
        
        self.reward_thres = self.reward_moving_coef * self.reward_thres + (1 - self.reward_moving_coef) * new_reward_thres

    
class AdvantageBasedSelector(AllowUniqueEpisodes):
    
    def __init__(self, config) -> None:
        
        super(AdvantageBasedSelector, self).__init__(config)
    
    
    def select(self, info, reward, rollout_buffer, env_id, **kwargs):
        cur_pos = rollout_buffer.pos
        value = rollout_buffer.values[cur_pos - 1, env_id]
        expression_conditional = super(AdvantageBasedSelector, self).select(info, update_expr=False)
        advantage_conditional = reward > value
        if expression_conditional and advantage_conditional:
            self.add_expr(info)
            return True
        else:
            return False
            
EP_SELECTOR = {
    'AllowAllEpisodes': AllowAllEpisodes,
    'AllowUniqueEpisodes': AllowUniqueEpisodes,
    'RewardBasedSelector': RewardBasedSelector,
    'AdvantageBasedSelector': AdvantageBasedSelector,
    "AllowNoEpisodes": AllowNoEpisodes,
    "AllowUniqueIDs": AllowUniqueIDs,
    "AllowUniqueEpisodeAndID": AllowUniqueEpisodeAndID
}
