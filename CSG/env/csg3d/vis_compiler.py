import networkx as nx
import torch as th
import numpy as np
from .compiler import MCSG3DCompiler
from .compiler_utils import get_reward
from .graph_compiler import GraphicalMCSG3DCompiler

class VisMCSG3DCompiler(GraphicalMCSG3DCompiler):
    """For doing the things that are related to program tree construction.
    Program Tree is innately tied to the compile process. 
    """
    def __init__(self, *args, **kwargs):
        super(VisMCSG3DCompiler, self).__init__(*args, **kwargs)
        # For Tree Crops
        self.threshold = 0.025
        self.threshold_diff = 0.05
        self.mode = "3D"

    def vis_tree(self, command_list, target=None, reset=True, enable_subexpr_targets=False, add_sweeping_info=False, add_splicing_info=False):
        # first create simple graph
        graph = self.command_tree(command_list, target=None, reset=reset, enable_subexpr_targets=False, add_sweeping_info=False, add_splicing_info=False)
        # Then store expression at each node
        n_nodes = len(graph.nodes)


        for node_id in range(n_nodes):
            cur_node = graph.nodes[node_id]
            cur_command_list = self.get_node_commands(graph, cur_node)
            cur_node['subexpr_info'] = {}
            cur_node['subexpr_info']['commands'] = cur_command_list
            cur_node['name'] = cur_node['symbol']
            # self._compile(cur_command_list)
            # canvas = self._output <= 0
            # cur_node['subexpr_info']['expr_shape'] = canvas
            # Check if it has child nodes with >1 children.
            draw_flag = self.get_draw_flag(graph, cur_node)
            cur_node['subexpr_info']['draw'] = draw_flag
        # calculate tree, and repr for each. 
        # Remove the graph parts which we don't have to draw:
        graph = self.remove_undrawn_nodes(graph)
        # set the level counter:
        graph, max_level = self.set_levels(graph)
        print("Max level is", max_level)

        return graph
    
    def remove_undrawn_nodes(self, graph):

        cur_node = graph.nodes[0]
        cur_node['level_counter'] = 0
        state_list = [cur_node]
        while(state_list):
            selected_node = state_list[0]
            state_list = state_list[1:]
            node_id = selected_node['node_id']
            children_id = selected_node['children']
            # children_id.sort()
            for child_id in children_id[::-1]:
                new_child = graph.nodes[child_id]
                state_list.insert(0, new_child)
            if not selected_node['subexpr_info']['draw']:
                parent_id = selected_node['parent']
                if not parent_id is None:
                    if parent_id in graph.nodes:
                        parent_node = graph.nodes[parent_id]
                        parent_node['children'].remove(node_id)
                graph.remove_node(node_id)

        return graph
        
    def set_levels(self, graph):
        cur_node = graph.nodes[0]
        cur_node['level_counter'] = 0
        state_list = [cur_node]
        max_level = 0
        while(state_list):
            selected_node = state_list[0]
            state_list = state_list[1:]

            children_id = selected_node['children']
            # children_id.sort()
            for child_id in children_id[::-1]:
                new_child = graph.nodes[child_id]
                new_level = selected_node["level_counter"] +1
                new_child['level_counter'] = new_level
                max_level = max(max_level, new_level)
                state_list.insert(0, new_child)
        return graph, max_level
        
    def get_draw_flag(self, graph, cur_node):
        # Treat the tree node as starting point. 
        draw_flag = False
        state_list = [cur_node]
        while(state_list):
            selected_node = state_list[0]
            state_list = state_list[1:]
            children_id = selected_node['children']
            # children_id.sort()
            if len(children_id)>=2:
                draw_flag = True
            for child_id in children_id[::-1]:
                new_child = graph.nodes[child_id]
                state_list.insert(0, new_child)
        if draw_flag is False:
            parent_id = cur_node['parent']
            if not parent_id is None:
                parent_node = graph.nodes[parent_id]
                if len(parent_node['children']) >=2:
                    draw_flag = True
                    cur_node['name'] = "_".join([x['symbol'][:5] for x in cur_node['subexpr_info']['commands']])
                    if "_" in parent_node['name']:
                        parent_node['name'] = parent_node['symbol']
                elif parent_node['symbol'] == 'mirror':
                    draw_flag = True
                    cur_node['name'] = "_".join([x['symbol'][:5] for x in cur_node['subexpr_info']['commands']])
                    if "_" in parent_node['name']:
                        parent_node['name'] = parent_node['symbol']

        return draw_flag