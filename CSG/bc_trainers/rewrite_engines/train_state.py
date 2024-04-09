import numpy as np

class BCTrainState():

    def __init__(self, epoch=0, n_epochs=100, n_iters_per_epoch=100):

        self.best_score = -np.inf
        self.cur_score = -np.inf
        self.best_epoch = 0
        self.cur_epoch = epoch

        self.n_epochs = n_epochs
        self.n_iters_per_epoch = n_iters_per_epoch
        self.n_forwards = 0
        self.n_updates = 0
        self.tensorboard_step = epoch * self.n_iters_per_epoch
        self.state_attrs = ["best_score", "cur_score", "best_epoch", "cur_epoch", "n_epochs",
                            "n_iters_per_epoch", "n_forwards", "n_updates", "tensorboard_step"]

    def get_state_stats(self,):
        stats_dict_it = {}
        stats_dict_it['Total Epochs'] = self.n_epochs
        stats_dict_it['Epoch'] = self.cur_epoch
        stats_dict_it['Best Epoch'] = self.best_epoch
        stats_dict_it['Iters/per Epoch'] = self.n_iters_per_epoch
        return stats_dict_it

    def get_state(self):
        state_dict = {}
        for key in self.state_attrs:
            value = getattr(self, key)
            state_dict[key] = value
        return state_dict
    
    def set_state(self, state_dict):
        for key, value in state_dict.items():
            setattr(self, key, value)

class PladTrainState(BCTrainState):

    def __init__(self, *args, **kwargs):

        super(PladTrainState, self).__init__(*args, **kwargs)
        self.poor_epochs = 0
        self.n_search = 0
        self.post_search_epoch = 0
        self.best_search_loss = np.inf
        self.epoch_avg_loss = self.best_search_loss

        new_state_attrs = ["poor_epochs","n_search","post_search_epoch","best_search_loss","epoch_avg_loss"]
        self.state_attrs.extend(new_state_attrs)
    
    
    def get_state_stats(self,):
        stats_dict_it = super(PladTrainState, self).get_state_stats()
        stats_dict_it['Search Count'] = self.n_search
        stats_dict_it['Poor Epochs'] = self.poor_epochs
        stats_dict_it['Post Search Epochs'] = self.post_search_epoch
        stats_dict_it['Epoch Avg. L'] = self.epoch_avg_loss
        stats_dict_it['Best Search L'] = self.best_search_loss
        return stats_dict_it

class WSTrainState(BCTrainState):

    def __init__(self, *args, **kwargs):
        super(WSTrainState, self).__init__(*args, **kwargs)
        self.ws_iteration = 0
        self.poor_epochs = 0
        self.epoch_loss = np.inf
        self.best_epoch_loss = np.inf

        new_state_attrs = ["ws_iteration","poor_epochs","epoch_loss","best_epoch_loss"]
        self.state_attrs.extend(new_state_attrs)

    def get_state_stats(self,):
        stats_dict_it = super(WSTrainState, self).get_state_stats()
        stats_dict_it['Poor Epochs'] = self.poor_epochs
        stats_dict_it['Epoch L'] = self.epoch_loss
        stats_dict_it['Best Epoch L'] = self.best_epoch_loss
        return stats_dict_it
