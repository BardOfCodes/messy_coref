
_NUMBER = 0
_PTYPE = 1
_TRANSFORM = 2
_BOOL = 3
_STOP = 4
MAX_TRANSFORM_BOOL_RATIO = 3.0
class BaseState:
    
    def __init__(self, number_count, primitive_count, boolean_count, transform_count):
        self.boolean_count = boolean_count
        self.primitive_count = primitive_count
        self.number_count = number_count
        self.transform_count = transform_count
        self.over = False
        
    @property
    def state(self):
        if self.over:
            number_state = 0
            primitive_state = 0
            transform_state = 0
            bool_state = 0
            stop_state = 0
        elif self.number_count > 0:
            number_state = 1
            primitive_state = 0
            transform_state = 0
            bool_state = 0
            stop_state = 0
        else:
            number_state = 0
            if self.primitive_count > 0:
                primitive_state = 1
                stop_state = 0
            else:
                primitive_state = 0
                stop_state = 1
                self.boolean_count = 0
                
            bool_state = int(self.boolean_count > 0)

            if (bool_state==0) and (primitive_state == 0):
                transform_state = 0
            else:
                transform_state = int(self.transform_count > 0)
                
        self.action_state ={
            _NUMBER: number_state,
            _PTYPE: primitive_state,
            _TRANSFORM: transform_state,
            _BOOL: bool_state,
            _STOP: stop_state
        } 
        state = [number_state, primitive_state, transform_state, bool_state, stop_state] 
        return state
        
    def get_internal_state(self):
        return [self.boolean_count, self.primitive_count, self.number_count, self.over, self.transform_count]
    
    def set_internal_state(self, state):
        self.boolean_count = state[0]
        self.primitive_count = state[1]
        self.number_count = state[2]
        self.over = state[3]
        self.transform_count = state[4]
    
class MCSGStateMachine:
    def __init__(self, boolean_limit, action_space):
        self.init_boolean_limit = boolean_limit
        self.init_transform_limit = int(MAX_TRANSFORM_BOOL_RATIO * self.init_boolean_limit)
        self.state = None
        self.reset()
        self.action_space = action_space
        
    def reset(self):
        # self.state_stack = [FinishState(), BaseState(0, 1, self.init_boolean_limit)]
        self.state = BaseState(0, 1, self.init_boolean_limit, self.init_transform_limit)
        
    def get_action_type(self, action):
        ## TODO: This is linked to the action space.
        action_type, parameter_count = self.action_space.index_to_command[action]
        return action_type, parameter_count
        
    def update_state(self, action):
        action_type, parameter_count = self.get_action_type(action)
        
        cur_state = self.state
        if action_type == _NUMBER:
            self.state.number_count -= 1
        elif action_type == _PTYPE:
            self.state.number_count = parameter_count # varies
            self.state.primitive_count -= 1
            # self.state_stack.append(BaseState(new_number_count, new_primitive_count, new_boolean_count, False))
        elif action_type == _TRANSFORM:
            self.state.number_count = parameter_count
            self.state.transform_count -= 1
        elif action_type == _BOOL:
            self.state.primitive_count += 1
            self.state.boolean_count -= 1
            # Put previous state back in:
        elif action_type == _STOP:
            self.state.boolean_count = 0
            self.state.over = True
                
    def get_state(self):
        # cur_state = self.state_stack[-1]
        return self.state.state      
    
    def get_internal_state(self):
         return self.state.get_internal_state()

    def set_internal_state(self, internal_state):
         return self.state.set_internal_state(internal_state)
          

class FCSGState(BaseState):

    @property
    def state(self):
        state = super(FCSGState, self).state
        state[2] = 0
        # No transforms allowed.
        return state

class FCSGStateMachine(MCSGStateMachine):

    def reset(self):
        # self.state_stack = [FinishState(), BaseState(0, 1, self.init_boolean_limit)]
        self.state = FCSGState(0, 1, self.init_boolean_limit, 0)