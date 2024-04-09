import torch as th
import time
import numpy as np
from CSG.env.shape_assembly.graph_compiler import GraphicalSACompiler
from .graph_sweep_helper import REWARD_THRESHOLD
import _pickle as cPickle
MASKED_MATCH_REQUIREMENT = 0.05
THRESHOLD = 100

def parallel_sa_gs(proc_id, mini_prog_dict, temp_env, save_loc, td_mode, reward_threshold, length_alpha=0, count_limit=9998):

    th.backends.cudnn.benchmark = True
    # compiler = temp_env.program_generator.compiler
    compiler = GraphicalSACompiler(resolution=temp_env.program_generator.compiler.resolution,
                                scale=temp_env.program_generator.compiler.scale,
                                draw_mode=temp_env.program_generator.compiler.draw.mode)

    temp_env.program_generator.set_execution_mode(th.device("cuda"), th.float16)
    parser = temp_env.program_generator.parser

    compiler.set_to_cuda()
    compiler.set_to_half()
    compiler.reset()


    previous_rewards = []
    updated_rewards = []
    previous_lengths = []
    updated_lengths = []
    bu_delta = []
    td_delta = []
    
    start_time = time.time()
    prog_objs= []

    for iteration, (key, value) in enumerate(mini_prog_dict):
        cur_slot = key[0]
        cur_target = key[1]
        obs = temp_env.reset_to_target(cur_slot, cur_target)
        target_np = obs['obs']
        target = th.from_numpy(target_np).cuda().bool()
        key = (cur_slot, cur_target)
        cur_expression = value['expression']
        prev_reward = value['reward']
        command_list = parser.parse(cur_expression)

        with th.no_grad():
            # with th.cuda.amp.autocast():
                graph = compiler.command_tree(command_list, target=target, enable_subexpr_targets=True, add_splicing_info=True)
        td_delta.append(0)
        bu_delta.append(0)
        previous_rewards.append(prev_reward)
        previous_lengths.append(len(cur_expression))

        # now traverse the graph and add indices that you can remove:
        removable = [False for x in cur_expression]
        token_to_program_id = dict()
        program_limits = []
        init = 0
        program_id = 0
        for ind, expr in enumerate(cur_expression):
            token_to_program_id[ind] = program_id
            if "$" in expr:
                program_id += 1
                program_limits.append((init, ind +1))
                init = ind + 1
            

        # Now annotate the targets and target masks:# Now annotate the targets and target masks:
        state_list = [graph.nodes[0]]
        while(state_list):
            cur_node = state_list[0]
            state_list = state_list[1:]
            # Perform processing according to the node type
            node_type = cur_node['type']

            if node_type in ["B", "D"]:
                sa_info = cur_node['sa_info']
                if sa_info['deletable']:
                    # check masked matching:
                    if cur_node['subexpr_info']['sa_sweep_metric'] < THRESHOLD:
                        # can be deleted:
                        cmd_ids = sa_info['command_ids']
                        # first check if program has valid tokens left in subprogram:
                        cur_prog = token_to_program_id[cmd_ids[0]]
                        prog_lims = program_limits[cur_prog]
                        temp_removable = [x for x in removable]
                        for j in range(cmd_ids[0], cmd_ids[1]):
                            temp_removable[j] = True

                        reminder_program = [x for ind, x in enumerate(cur_expression[prog_lims[0]:prog_lims[1]]) if not temp_removable[ind + prog_lims[0]]]
                        n_cuboids = np.sum(['cuboid(' in x for x in reminder_program])
                        if n_cuboids >= 2:
                            removable = [x for x in temp_removable]
                            # print("removing", cmd_ids)
                            # print("removing as fully masked and removable", cur_node['subexpr_info']['masked_matching'])
                        else:
                            pass
                            # print("will result in empty subprogram")

            cur_children_id = cur_node['children']
            for child_id in cur_children_id[::-1]:
                child = graph.nodes[child_id]
                state_list.insert(0, child)

        
        # Now make the new expression:
        if any(removable):
            initial_pointer = 0
            new_expression = []
            for ind, expr in enumerate(cur_expression):
                if not removable[ind]:
                    new_expression.append(expr)

            # Nwo correct the naming:
                                
            program_limits = []
            init = 0
            program_id = 0
            corrected_expr = []
            for ind, expr in enumerate(new_expression):
                if "$" in expr:
                    program_id += 1
                    program_limits.append((init, ind +1))
                    init = ind + 1
            for cur_prog_inds in program_limits:
                cur_program = new_expression[cur_prog_inds[0]:cur_prog_inds[1]]
                n_cuboids = [x for x in cur_program if "cuboid(" in x]
                n_cuboids = [x.split("=")[0].strip() for x in n_cuboids]
                n_cuboids = n_cuboids[1:]
                correct_names = ["re_cube_%d"%x for x in range(len(n_cuboids))]
                name_pairs = [(x, correct_names[ind]) for ind, x in enumerate(n_cuboids) if x != correct_names[ind]] 
                for ind, statement in enumerate(cur_program):
                    for n_p in name_pairs[::-1]:
                        statement = statement.replace(n_p[0], n_p[1])
                    cur_program[ind] = statement
                for ind, statement in enumerate(cur_program):
                    statement = statement.replace("re_cube_", "cube")
                    cur_program[ind] = statement
                corrected_expr.extend(cur_program)


            actions = temp_env.action_space.expression_to_action(corrected_expr) 
            real_expression = temp_env.action_space.action_to_expression(actions) 
            real_pred_canvas = temp_env.program_generator.execute(real_expression, return_numpy=True)
            real_reward = temp_env.reward(real_pred_canvas, target_np, True, [])
            real_reward = real_reward + length_alpha * len(real_expression)

            if real_reward >= (prev_reward - REWARD_THRESHOLD) and len(real_expression) < len(cur_expression):
                updated_lengths.append(len(real_expression))
                updated_rewards.append(real_reward)
                prog_obj = dict(expression=real_expression,
                                slot_id=cur_slot,
                                target_id=cur_target,
                                reward=real_reward,
                                origin="GS",
                                do_fail=False,
                                cs_fail=False,
                                log_prob=0)
                prog_objs.append(prog_obj)
            else:
                # Add lengths as well
                updated_rewards.append(prev_reward)
                updated_lengths.append(len(cur_expression))
        else:
            # Add lengths as well
            updated_rewards.append(prev_reward)
            updated_lengths.append(len(cur_expression))

    if len(prog_objs) > count_limit:
        prog_objs.sort(key=lambda x: x['reward'], reverse=True)
        prog_objs = prog_objs[:int(count_limit)]
            
    if not previous_rewards:
        previous_rewards = [0]
        updated_rewards = [0]
        previous_lengths = [1]
        updated_lengths = [1]
    
    with open(save_loc + "/gs_%d.pkl" % proc_id, 'wb') as f:
        cPickle.dump([prog_objs, updated_rewards, previous_rewards, updated_lengths, previous_lengths, 
                      td_delta, bu_delta], f)
  