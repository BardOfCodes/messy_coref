from .action_2d import RestrictedAction, RefactoredActionSpace, MultiRefactoredActionSpace, OpRefactoredAction, OpRestrictedAction
from .action_2d import ACTION_SPACE, MULTI_ACTION_SPACE
from .csg3d.action_space import MCSGAction3D, HCSGAction3D, FCSGAction3D, PCSGAction3D, MCSGActionExtended3D
from .shape_assembly.action_space import HSA3DAction, PSA3DAction
from .csg2d.action_space import MCSGAction2D, HCSGAction2D, FCSGAction2D, PCSGAction2D

def get_action_space(action_space_type, *args, **kwargs):
    if action_space_type == "RestrictedAction":
        n_action = 400
        return RestrictedAction(n_action)
    elif action_space_type == "RefactoredActionSpace":
        return RefactoredActionSpace(ACTION_SPACE)
    elif action_space_type == "MultiRefactoredActionSpace":
        return MultiRefactoredActionSpace(MULTI_ACTION_SPACE)
    elif action_space_type == "ComplexityAction":
        return RestrictedAction(10)
    elif action_space_type == "OpRestrictedAction":
        return OpRestrictedAction(400)
    elif action_space_type == "MCSG3DAction":
        resolution = kwargs['resolution']
        return MCSGAction3D(resolution)
        # return MCSGActionExtended3D(resolution)
    elif action_space_type == "HCSG3DAction":
        resolution = kwargs['resolution']
        return HCSGAction3D(resolution)
    elif action_space_type == "FCSG3DAction":
        resolution = kwargs['resolution']
        return FCSGAction3D(resolution)
    elif action_space_type == "PCSG3DAction":
        resolution = kwargs['resolution']
        return PCSGAction3D(resolution)
    elif action_space_type == "HSA3DAction":
        resolution = kwargs['resolution']
        n_cuboid_ind_states = kwargs['n_cuboid_ind_states']
        return HSA3DAction(resolution, n_cuboid_ind_states)
    elif action_space_type == "PSA3DAction":
        resolution = kwargs['resolution']
        n_cuboid_ind_states = kwargs['n_cuboid_ind_states']
        return PSA3DAction(resolution, n_cuboid_ind_states)
    elif action_space_type == "MCSG2DAction":
        resolution = kwargs['resolution']
        return MCSGAction2D(resolution)
    elif action_space_type == "HCSG2DAction":
        resolution = kwargs['resolution']
        return HCSGAction2D(resolution)
    elif action_space_type == "FCSG2DAction":
        resolution = kwargs['resolution']
        return FCSGAction2D(resolution)
    elif action_space_type == "PCSG2DAction":
        resolution = kwargs['resolution']
        return HCSGAction2D(resolution)
        