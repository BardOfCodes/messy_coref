from .restricted_env import RestrictedCSG
from .cad_env import CADCSG
import numpy as np
import networkx as nx

## TODO: Handle Restricted and CADCSG in both.
class ModifierCSG(CADCSG):
    
    
    def refactor_expression(self, pred_expression, target_canvas=None):
        
        if target_canvas is None:
            target_canvas = self.obs[0] * 0
        parsed_exp = self.parser.parse_full_expression(pred_expression)
        
        self.action_sim.reset()
        program_tree = self.get_program_tree(parsed_exp, target_canvas)
        
        self.action_sim.reset()
        non_terminals = []
        n_nodes = len(program_tree.nodes)
        non_terminals = [program_tree.nodes[i] for i in range(n_nodes) if program_tree.nodes[i]['children'] is not None]
        max_reward = np.max([n['reward'] for n in non_terminals])
        max_reward_nodes = [n for n in non_terminals if n['reward'] == max_reward]
        min_program_len = np.min([len(node['expr']) for node in max_reward_nodes])
        ml_mr_node = [node for node in max_reward_nodes if len(node['expr']) == min_program_len][0]
        
        # Modify to remove links to other parts of the program:
        ml_mr_node['parent'] = None
            
        # Find smallest program with best reward.
        # Top-Down Pass
        # Find largest uesless child.
        # new_graph = self.top_down_refactor(ml_mr_node)

        node_list = [ml_mr_node]
        terminal_nodes = []
        expr_failed = False
        
        while node_list:
            cur_node = node_list[0]
            cur_id = cur_node['node_id']
            children_id = cur_node['children']
            parent_id = cur_node['parent']
            if parent_id:
                parent = program_tree.nodes[parent_id]
            else:
                parent = None
            if children_id:
                child_0, child_1 = [program_tree.nodes[i] for i in children_id]
            
                if cur_node['empty_canvas']:
                    if not parent:
                        # If it has no parents but is empty -> Its garbage.
                        expr_failed = True
                        break
                        
                    # REMOVE the entire chain.
                    # print('found empty canvas at expr %d. Removing %d and %d.' % (cur_id, children_id[0], children_id[1]))
                    parent['children'].remove(cur_id)
                    # idx = parent['children'].index(cur_id)
                    sibling_ind = parent['children'][0] # [x for x in [0, 1] if not x == idx][0]
                    sibling = program_tree.nodes[sibling_ind]
                    new_expr = sibling['expr']
                    parent['expr'] = new_expr
                    parent['children'] = None
                    self.update_parents(parent, program_tree)
                    # program_tree.remove_edge(parent_id, cur_id)
                    # No addition to the list.
                elif cur_node['match_0']:
                    # print('Match 0: %d matches %d. Removing %d Adding %d.' % (cur_id, children_id[0], children_id[1], children_id[0]))
                    # If matches with 1, 2 and operation did not have any effect.
                    # remove op node, and join child_2 to parent.
                    if not parent:
                        expr_failed = True
                        break
                    else:
                        idx = parent['children'].index(cur_id)
                        parent['children'][idx] = children_id[0]
                        child_0['parent'] = parent_id
                        self.update_parents(child_0, program_tree)
                        
                    node_list.append(child_0)
                elif cur_node['match_1']:
                    # print('Match 1: %d matches %d. Removing %d Adding %d.' % (cur_id, children_id[1], children_id[0], children_id[1]))
                    if not parent:
                        expr_failed = True
                        break
                    else:
                        idx = parent['children'].index(cur_id)
                        parent['children'][idx] = children_id[1]
                        child_1['parent'] = parent_id
                        self.update_parents(child_1, program_tree)
                    node_list.append(child_1)
                else:
                    # Everything is all right.
                    # print('Everything is all right at expr %d. Adding %d & %d.' % (cur_id, children_id[0], children_id[1]))
                    if children_id:
                        node_list.append(child_0)
                        node_list.append(child_1)
            else:
                # print("No children for %d" % cur_id)
                terminal_nodes.append(cur_node)
            node_list.remove(cur_node)
        # New tree which has to be redone. 
        # TODO: Find bug and remove this hack fix.
        for node in terminal_nodes:
            self.update_parents(node, program_tree)
        
        if expr_failed: 
            expression = ""
        else:
            expression = ml_mr_node['expr']
            expression.extend(pred_expression[-1])
        return expr_failed, expression 
            
    def get_refactored_experience(self, pred_expression, target_canvas):
        
        
        expr_failed, new_expression = self.refactor_expression(pred_expression, target_canvas)    
            
        if expr_failed:
            new_expression, refactored_observations, \
                refactored_actions, refactored_rewards = None, None, None, None
            # print("Refactor failed due to poor program")
        else:
            refactored_observations, refactored_actions, refactored_rewards = self.generate_experience(new_expression, target_canvas)
        # self.reset()
            
        return new_expression, refactored_observations, \
                refactored_actions, refactored_rewards
                
    def generate_experience(self, new_expression, target_canvas, slot_id, target_id):
        
        
        # refactored_actions = np.array([self.unique_draw.index(x) for x in new_expression])
        # refactored_rewards[-1] = 1.0
        refactored_actions = np.array([self.action_space.expression_to_action(x, self.unique_draw) for x in new_expression])
        # new experience:
        refactored_observations = []
        self.action_sim.reset()
        # parsed_exp = self.parser.parse_full_expression(new_expression)
        refactored_observations, refactored_rewards = self.sim_refactored_obs(refactored_actions, target_canvas, slot_id, target_id)
        
        refactored_actions = refactored_actions[:,None]
        self.action_sim.reset()
        return refactored_observations, refactored_actions, refactored_rewards
    
    
    def sim_refactored_obs(self, actions, target_canvas, slot_id, target_id):
        # Need to run sim.
        info = {
            'target_expression': [self.unique_draw[0] for x in range(self.max_len)],
            'target_canvas': target_canvas,
            'slot_id': slot_id,
            'target_id': target_id
        }
        refac_obs = self.reset_from_info(info)
        self.obs[0:1] = target_canvas
        rewards = np.zeros(actions.shape[0])
        # change to usable fornat:
        refac_obs['obs'] = refac_obs['obs'][None,]
        refac_obs['draw_allowed'] = np.array(refac_obs['draw_allowed'])[None,None]
        refac_obs['stop_allowed'] = np.array(refac_obs['stop_allowed'])[None,None]
        refac_obs['op_allowed'] = np.array(refac_obs['op_allowed'])[None,None]
        for ind, action in enumerate(actions):
            obs_dict, reward, done, info = self.step(action)
            rewards[ind] = reward
            if done:
                break
            for key, value in obs_dict.items():
                if isinstance(value, int):
                    value = np.array(value)[None,]
                refac_obs[key] = np.concatenate([refac_obs[key], value[None,]], 0)
        
        # predicted_canvas = refac_obs['obs'][-1, 1:2]
        # refac_obs['obs'][:, :1] = target_canvas 
        
        return refac_obs, rewards
    
    def update_parents(self, parent, program_tree):
        
        cur_node = parent
        while(cur_node['parent']):
            parent_id = cur_node['parent']
            parent = program_tree.nodes[parent_id]
            self_ind = parent['children'].index(cur_node['node_id'])
            
            sibling_ind = (self_ind +1) %2
            
            self_expr = cur_node['expr'].copy()
            sibling_id = parent['children'][sibling_ind]
            sibling = program_tree.nodes[sibling_id]
            sibling_expr = sibling['expr'].copy()
            new_expr = []
            if self_ind == 0:
                new_expr.extend(self_expr)
                new_expr.extend(sibling_expr)
                new_expr.extend(parent['current_op'])
            else:
                new_expr.extend(sibling_expr)
                new_expr.extend(self_expr)
                new_expr.extend(parent['current_op'])
            parent['expr'] = new_expr
                
            cur_node = parent
            
    def get_program_tree(self, parsed_exp, target_canvas):
        
        graph = nx.DiGraph()
        counter = 0
        counter_list = []
        name_list = []
        for index, p in enumerate(parsed_exp):
            if p["type"] == "draw":
                x = float(p["param"][0]) * self.canvas_shape[0] / 64
                y = float(p["param"][1]) * self.canvas_shape[1] / 64
                scale = float(p["param"][2]) * self.canvas_shape[0] / 64
                # Copy to avoid over-write
                layer = self.action_sim.draw[p["value"][0]]([x, y], scale)
                self.action_sim.stack.push(layer)
                reward = self.reward(layer[None, :], target_canvas, True)
                # img = np.stack([layer, ] * 3, -1)
                # img = Image.fromarray(np.uint8(img))
                graph.add_node(counter, node_id=counter, expr=[p['name']], production=layer, reward=reward,
                               children=None, current_op=p['name'], parent=None, resolved=True)
                name_list.append([p['name']])
                counter_list.append(counter)
                counter += 1
            elif p["type"] == "stop":
                pass
            else:
                # operate
                obj_2 = self.action_sim.stack.pop()
                obj_1 = self.action_sim.stack.pop()
                layer = self.action_sim.op[p["value"]](obj_1, obj_2)
                empty_canvas, match_0, match_1 = self.valid_operation(layer, obj_1, obj_2)
                self.action_sim.stack.push(layer)
                
                
                reward = self.reward(layer[None,:], target_canvas, True)
                
                l2, l1 = name_list.pop(), name_list.pop()
                l1.extend(l2)
                l1.append(p['name'])
                expr = l1
                
                prev_counters = [counter_list.pop(), counter_list.pop()]
                children = prev_counters[::-1]
                graph.add_node(counter, node_id=counter, expr=expr, production=layer, reward=reward,
                               current_op=p['name'], parent=None,
                               empty_canvas=empty_canvas, match_0=match_0, match_1=match_1,
                               children=children, resolved=False)
                # Add parent:
                for _prev in prev_counters:
                    graph.add_edge(_prev, counter)
                    graph.nodes[_prev]['parent'] = counter
                # add edges:
                graph.add_edge(prev_counters[1], counter)
                graph.add_edge(prev_counters[0], counter)
                
                name_list.append(expr)
                counter_list.append(counter)
                counter += 1
                
        # write_dot(graph,'test.dot')
        return graph
    
    def valid_operation(self, output, top, bottom, threshold=1):
            
        val_1 = np.sum(output)
        val_2 = np.sum(output ^ top)
        val_3 = np.sum(output ^ bottom) 
        cond_1 = val_1 < threshold # < 40*40
        # Also not too small
        cond_2 = val_2 < threshold
        cond_3 = val_3 < threshold
        # output = cond_1 and cond_2 and cond_3
        return cond_1, cond_2, cond_3
