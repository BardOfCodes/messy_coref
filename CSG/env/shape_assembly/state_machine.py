"""
What actions are valid at the current state?
"""



# Action space:
#  5 Cuboids
#  R float
#  3 axis
#  6 face
#  3 sym count -> (1, 2, 3, 4)
#  2 leaf mode
#  5 command_type
#  1 stop
# -- Total
# 25 + Resolution
# State has 16 flags:

import numpy as np


class SAStateMachine:
    def __init__(self, master_min_prim, master_max_prim, sub_min_prim, sub_max_prim, action_space, n_cuboid_ind_states=6, hierarchy_allowed=True, max_sub_progs=3):

        self.program_state = 0 # "S" after finish
        self.n_cuboids = 0
        self.cuboid_attach_count = []
        self.cuboid_macro_count = []
        self.action_space = action_space
        self.n_cuboid_ind_states = n_cuboid_ind_states 
        self.max_sub_progs  = max_sub_progs
        self.hierarchy_allowed = hierarchy_allowed

        self.master_min_prim = master_min_prim
        self.master_max_prim = master_max_prim
        self.sub_min_prim = sub_min_prim
        self.sub_max_prim = sub_max_prim # set to 2
        self.min_max_prim = {
            0: [self.master_min_prim, self.master_max_prim],
            1: [self.sub_min_prim, self.sub_max_prim],
            2: [self.sub_min_prim, self.sub_max_prim],
            3: [self.sub_min_prim, self.sub_max_prim],
            4: [self.sub_min_prim, self.sub_max_prim],
            5: [self.sub_min_prim, self.sub_max_prim],
            6: [self.sub_min_prim, self.sub_max_prim],
        }

        self.create_all_states()
        # Global counter
        # self.max_total_cuboids = 15
        # self.max_attaches = 20
        # self.max_subprogram = 4
        # self.max_macros = 10

        # self.global_cuboid_count = 0
        # self.global_attach_count = 0
        # self.global_subprogram_count = 0
        # self.macro_count = 0

    def create_all_states(self):

        self._FLOAT = 0
        self._CUBOID = 1 # (3f) or (3f, 1i) 
        self._ATTACH = 2 # (c1, 6f)
        self._SQUEEZE = 3 # (c_1, c_2, face, 2f)
        self._TRANSLATE = 4 # (Axis, n_count, f)
        self._REFLECT = 5 # (Axis)
        self._AXIS = 6
        self._FACE = 7
        self._SYM_COUNT = 8
        self._LEAF_TYPE_EMPTY = 9
        self._LEAF_TYPE_SUBPR = 10
        
        PRE_CUBOID_IND = 11

        cur_id= 11
        for i in range(self.n_cuboid_ind_states):
            setattr(self, "_CUBOID_ID_%d" % i, PRE_CUBOID_IND + i)
            cur_id += 1

        self._SUBPROGRAM_STOP = PRE_CUBOID_IND + self.n_cuboid_ind_states
        self._STOP = PRE_CUBOID_IND + self.n_cuboid_ind_states + 1
        empty_state = [0 for x in range(PRE_CUBOID_IND + self.n_cuboid_ind_states + 2)]
        _ONLY_FLOAT_STATE = empty_state.copy()
        _ONLY_FLOAT_STATE[0] = 1
        self._ONLY_FLOAT_STATE =  _ONLY_FLOAT_STATE
        _ONLY_LEAF_TYPE_STATE = empty_state.copy()
        _ONLY_LEAF_TYPE_STATE[9] = 1
        _ONLY_LEAF_TYPE_STATE[10] = 1
        self._ONLY_LEAF_TYPE_STATE = _ONLY_LEAF_TYPE_STATE
        _ONLY_CUBOID_ID_STATE = empty_state.copy()
        _ONLY_CUBOID_ID_STATE[PRE_CUBOID_IND : PRE_CUBOID_IND + self.n_cuboid_ind_states] = [1 for i in range(self.n_cuboid_ind_states)]
        self._ONLY_CUBOID_ID_STATE = _ONLY_CUBOID_ID_STATE
        _ONLY_SYM_COUNT_STATE = empty_state.copy()
        _ONLY_SYM_COUNT_STATE[8] = 1
        self._ONLY_SYM_COUNT_STATE = _ONLY_SYM_COUNT_STATE
        _ONLY_FACE_ID_STATE = empty_state.copy()
        _ONLY_FACE_ID_STATE[7] = 1
        self._ONLY_FACE_ID_STATE = _ONLY_FACE_ID_STATE
        _ONLY_AXIS_STATE = empty_state.copy()
        _ONLY_AXIS_STATE[6] = 1
        self._ONLY_AXIS_STATE = _ONLY_AXIS_STATE
        _ONLY_CUBOID_STATE = empty_state.copy()
        _ONLY_CUBOID_STATE[1] = 1
        _ONLY_CUBOID_STATE[-1] = 1
        _ONLY_CUBOID_STATE[-2] = 1
        self._ONLY_CUBOID_STATE = _ONLY_CUBOID_STATE
        _ONLY_ATT_SQ_STATE = empty_state.copy()
        _ONLY_ATT_SQ_STATE[2:4] = [1, 1]
        self._ONLY_ATT_SQ_STATE = _ONLY_ATT_SQ_STATE
        _ONLY_CU_MAC_STATE = empty_state.copy()
        _ONLY_CU_MAC_STATE[1] = 1
        _ONLY_CU_MAC_STATE[4:6] = [1, 1]
        _ONLY_CU_MAC_STATE[-1] = 1
        _ONLY_CU_MAC_STATE[-2] = 1
        self._ONLY_CU_MAC_STATE = _ONLY_CU_MAC_STATE
        _POST_FIRST_ATTACH_STATE = empty_state.copy()
        _POST_FIRST_ATTACH_STATE[1:3] = [1, 1]
        _POST_FIRST_ATTACH_STATE[4:6] = [1, 1]
        _POST_FIRST_ATTACH_STATE[-1] = 1
        _POST_FIRST_ATTACH_STATE[-2] = 1
        self._POST_FIRST_ATTACH_STATE =  _POST_FIRST_ATTACH_STATE
        _POST_SEC_ATTACH_STATE = empty_state.copy()
        _POST_SEC_ATTACH_STATE[1] = 1
        _POST_SEC_ATTACH_STATE[4:6] = [1, 1]
        _POST_SEC_ATTACH_STATE[-1] = 1
        _POST_SEC_ATTACH_STATE[-2] = 1
        self._POST_SEC_ATTACH_STATE =_POST_SEC_ATTACH_STATE
        self._NO_ACTION_STATE = empty_state.copy()
        _ONLY_STOP_STATE = empty_state.copy()
        _ONLY_STOP_STATE[-1] = 1
        _ONLY_STOP_STATE[-2] = 1
        self._ONLY_STOP_STATE = _ONLY_STOP_STATE

        action_to_state_req_dict = {
            self._FLOAT: [],
            self._CUBOID : [_ONLY_FLOAT_STATE, _ONLY_FLOAT_STATE, _ONLY_FLOAT_STATE, _ONLY_LEAF_TYPE_STATE],
            self._ATTACH : [_ONLY_CUBOID_ID_STATE, _ONLY_FLOAT_STATE, _ONLY_FLOAT_STATE, _ONLY_FLOAT_STATE, 
                    _ONLY_FLOAT_STATE, _ONLY_FLOAT_STATE, _ONLY_FLOAT_STATE],
            self._SQUEEZE : [_ONLY_CUBOID_ID_STATE, _ONLY_CUBOID_ID_STATE, _ONLY_FACE_ID_STATE, _ONLY_FLOAT_STATE, _ONLY_FLOAT_STATE],
            self._TRANSLATE : [_ONLY_AXIS_STATE, _ONLY_SYM_COUNT_STATE, _ONLY_FLOAT_STATE],
            self._REFLECT : [_ONLY_AXIS_STATE],
            self._AXIS : [],
            self._FACE :  [],
            self._SYM_COUNT : [],
            self._LEAF_TYPE_EMPTY : [],
            self._LEAF_TYPE_SUBPR : [],
            self._SUBPROGRAM_STOP: [],
            self._STOP : [],
        }

        for i in range(self.n_cuboid_ind_states):
            action_to_state_req_dict[i + PRE_CUBOID_IND] = []
        
        self.action_to_state_req_dict = action_to_state_req_dict
        self.reset()
        
    def reset(self):
        # self.state_stack = [FinishState(), BaseState(0, 1, self.init_boolean_limit)]
        self.program_state = 0 # "S" after finish
        self.n_cuboids = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6:0}
        self.cuboid_attach_count = {x:[] for x in self.n_cuboids.keys()}
        self.cuboid_macro_count = {x:[] for x in self.n_cuboids.keys()}
        self.n_sub_programs = 0
        self.state = self._ONLY_FLOAT_STATE.copy()
        self.fulfill_list = [self._ONLY_CUBOID_STATE.copy()]

    def update_state(self, action):
        action_type = self.get_action_type(action)
        states_to_fulfill = [x.copy() for x in self.action_to_state_req_dict[action_type]]
        self.fulfill_list.extend(states_to_fulfill)
        # print(action, action_type, len(states_to_fulfill))
        # For the next state  

        if action_type == self._CUBOID:
            self.n_cuboids[self.program_state] += 1
            self.cuboid_attach_count[self.program_state].append(0)
            self.cuboid_macro_count[self.program_state].append(0)
            if not self.hierarchy_allowed:
                self.fulfill_list.pop()
            else:
                if self.program_state > 0:
                    # Remove requirement of Leaf type
                    self.fulfill_list.pop()
            self.fulfill_list.append(self._ONLY_ATT_SQ_STATE.copy())
            
        elif action_type == self._ATTACH:
            # Correct the ID pointer:
            n_cuboids = self.n_cuboids[self.program_state]
            valid = [int(x < n_cuboids) for x in range(self.n_cuboid_ind_states)]
            self.fulfill_list[0][11: 11 + self.n_cuboid_ind_states] = valid

            cur_attach_count = self.cuboid_attach_count[self.program_state][-1]
            if cur_attach_count == 0:
                self.fulfill_list.append(self._POST_FIRST_ATTACH_STATE.copy())
            else:
                self.fulfill_list.append(self._POST_SEC_ATTACH_STATE.copy())
            self.cuboid_attach_count[self.program_state][-1] += 1
        elif action_type == self._SQUEEZE:
            n_cuboids = self.n_cuboids[self.program_state]
            valid = [int(x < n_cuboids) for x in range(self.n_cuboid_ind_states)]
            self.fulfill_list[0][11: 11 + self.n_cuboid_ind_states] = valid
            self.fulfill_list[1][11: 11 + self.n_cuboid_ind_states] = valid

            self.cuboid_attach_count[self.program_state][-1] += 2
            self.fulfill_list.append(self._ONLY_CU_MAC_STATE.copy())
        elif action_type in [self._TRANSLATE, self._REFLECT]:
            # Post macro
            if self.cuboid_macro_count[self.program_state][-1] == 0:
                # Can do one more 
                self.fulfill_list.append(self._ONLY_CU_MAC_STATE.copy())
            else:
                self.fulfill_list.append(self._ONLY_CUBOID_STATE.copy())
            self.cuboid_macro_count[self.program_state][-1] += 1
                
        elif action_type == self._LEAF_TYPE_EMPTY:
            #In master program:
            self.n_sub_programs += 0
        elif action_type == self._LEAF_TYPE_SUBPR:
            self.n_sub_programs += 1

        elif action_type == self._SUBPROGRAM_STOP:
            self.program_state += 1
            self.fulfill_list.append(self._ONLY_CUBOID_STATE.copy())
        elif action_type == self._STOP:
            self.fulfill_list.append(self._NO_ACTION_STATE.copy())


        if len(self.fulfill_list) > 0:
            cur_state = self.fulfill_list[0]
            self.fulfill_list = self.fulfill_list[1:]
        else:
            # Now if there are cubes
            raise ValueError("This should not happen?")

        # If CU then check if valid
        cur_cuboid_allowed = cur_state[1]
        if cur_cuboid_allowed:
            # Check if allowed:
            n_cuboids = self.n_cuboids[self.program_state]
            if n_cuboids == self.master_max_prim:
                # gets converted to _ONLY_STOP_STATE
                cur_state[1] = 0

        cur_stop_allowed = cur_state[-1]
        cur_subprog_stop_allowed = cur_state[-2]
        if cur_subprog_stop_allowed or cur_stop_allowed:
            n_cuboids = self.n_cuboids[self.program_state]
            if n_cuboids < self.min_max_prim[self.program_state][0]:
                cur_state[-1] = 0
                cur_state[-2] = 0
            # Check if allowed:
            if not self.hierarchy_allowed:
                cur_state[-2] = 0
            else:
                if self.program_state == self.n_sub_programs:
                    # Cant subpr stop
                    cur_state[-2] = 0
                elif self.program_state < self.n_sub_programs:
                    # Cant stop
                    cur_state[-1] = 0
        
        leaf_type_empty = cur_state[9]     
        leaf_type_subpr = cur_state[10]
        if leaf_type_empty or leaf_type_subpr:
            if self.n_sub_programs == self.max_sub_progs:
                cur_state[10] = 0
            if not self.hierarchy_allowed:
                cur_state[9] = 0
                cur_state[10] = 0


        
        self.state = cur_state
        valid = np.sum(self.state)
        # print(cur_state)
        if valid == 0:
            print("Reached end state")
            # raise ValueError("This should not happen?")
                
    def get_action_type(self, action):
        ## TODO: This is linked to the action space.
        action_type = self.action_space.index_to_command[action]
        return action_type
        
    def get_state(self):
        return self.state.copy()     
    
    def get_internal_state(self):
         return [self.program_state, {x:y for x, y in self.n_cuboids.items()}, 
                 {x:y.copy() for x, y in self.cuboid_attach_count.items()}, 
                 {x:y.copy() for x, y in self.cuboid_macro_count.items()}, 
                 self.n_sub_programs, self.state, [x.copy() for x in self.fulfill_list]]

    def set_internal_state(self, internal_state):
         self.program_state = internal_state[0]
         self.n_cuboids = {x:y for x, y in internal_state[1].items()}
         self.cuboid_attach_count = {x:y.copy() for x, y in internal_state[2].items()}
         self.cuboid_macro_count = {x:y.copy() for x, y in internal_state[3].items()}
         self.n_sub_programs = internal_state[4]
         self.state = internal_state[5]
         self.fulfill_list = [x.copy() for x in internal_state[6]]
        #  self.program_state, self.n_cuboids, self.cuboid_attach_count, \
        #     self.cuboid_macro_count, self.n_sub_programs, self.state, self.fulfill_list = internal_state
          