
from .compiler import MCSG2DCompiler
from .parser import MCSG2DParser
from .parser_utils import boolean_commands, draw_commands, transform_commands
from .sub_parsers import HCSG2DParser, FCSG2DParser, PCSG2DParser
from .draw import DiffDraw2D
from CSG.env.csg3d.state_machine import MCSGStateMachine, FCSGStateMachine
from .graph_compiler import GraphicalMCSG2DCompiler

language_map = {
    "MCSG2D": {'parser': MCSG2DParser, 'compiler': MCSG2DCompiler, 'state_machine': MCSGStateMachine},
    "HCSG2D": {'parser': HCSG2DParser, 'compiler': MCSG2DCompiler, 'state_machine': MCSGStateMachine},
    "FCSG2D": {'parser': FCSG2DParser, 'compiler': MCSG2DCompiler, 'state_machine': FCSGStateMachine},
    "PCSG2D": {'parser': PCSG2DParser, 'compiler': MCSG2DCompiler, 'state_machine': FCSGStateMachine},
}