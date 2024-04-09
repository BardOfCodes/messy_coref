# from .green.compiler import Compiler3DCSG
# from .green.state_machine import CSG3DStateMachine
# from .green.draw import DiffDraw3D as OldDiffDraw
# from .green.parser import Parser3DCSG, 
# from .green.cnr_csg3d import CNRCompiler3DCSG, CNRParser3DCSG
# from .green.nr_csg3d import NRCompiler3DCSG, NRParser3DCSG
# from .green.nt_csg3d import NTCompiler3DCSG, NTParser3DCSG, NTCSG3DStateMachine, nt_draw_commands
# from .green.mnr_csg3d import MNRCompiler3DCSG, MNRParser3DCSG, mnr_macro_commands

from .compiler import MCSG3DCompiler
from .parser import MCSG3DParser
from .parser_utils import boolean_commands, draw_commands, transform_commands
from .sub_parsers import HCSG3DParser, FCSG3DParser, PCSG3DParser
from .draw import DiffDraw3D
from .state_machine import MCSGStateMachine, FCSGStateMachine
from .graph_compiler import GraphicalMCSG3DCompiler

language_map = {
    # "CSG3Dx": {'parser': Parser3DCSG, 'compiler': Compiler3DCSG, 'state_machine': CSG3DStateMachine},
    # "NTCSG3D": {'parser': NTParser3DCSG, 'compiler': NTCompiler3DCSG, 'state_machine': NTCSG3DStateMachine},
    # "NRCSG3D": {'parser': NRParser3DCSG, 'compiler': NRCompiler3DCSG, 'state_machine': CSG3DStateMachine},
    # "CNRCSG3D": {'parser': CNRParser3DCSG, 'compiler': CNRCompiler3DCSG, 'state_machine': CSG3DStateMachine},
    # "MNRCSG3D": {'parser': MNRParser3DCSG, 'compiler': MNRCompiler3DCSG, 'state_machine': CSG3DStateMachine},
    # New Languages
    "MCSG3D": {'parser': MCSG3DParser, 'compiler': MCSG3DCompiler, 'state_machine': MCSGStateMachine},
    "HCSG3D": {'parser': HCSG3DParser, 'compiler': MCSG3DCompiler, 'state_machine': MCSGStateMachine},
    "FCSG3D": {'parser': FCSG3DParser, 'compiler': MCSG3DCompiler, 'state_machine': FCSGStateMachine},
    "PCSG3D": {'parser': PCSG3DParser, 'compiler': MCSG3DCompiler, 'state_machine': FCSGStateMachine},
    "MCSG2D": {'parser': MCSG3DParser, 'compiler': MCSG3DCompiler, 'state_machine': MCSGStateMachine},
    "HCSG2D": {'parser': HCSG3DParser, 'compiler': MCSG3DCompiler, 'state_machine': MCSGStateMachine},
    "FCSG2D": {'parser': FCSG3DParser, 'compiler': MCSG3DCompiler, 'state_machine': FCSGStateMachine},
    "PCSG2D": {'parser': PCSG3DParser, 'compiler': MCSG3DCompiler, 'state_machine': FCSGStateMachine},
}