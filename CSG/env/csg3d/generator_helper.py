
from collections import defaultdict

import numpy as np
import torch
import _pickle as cPickle
from .languages import boolean_commands
from .constants import VALIDATION_THRESHOLD as THRESHOLD, VALIDATION_THRESHOLD_DIFF as DIFF_THRESHOLD, TRANSFORM_SAMPLING_RATE, MIRROR_SAMPLING_RATE, HIER_TFORM_RATE
# from CSG.bc_trainers.rewrite_engines.graph_sweep_helper import gs_singular_parse
def get_permutation_index_tuple(k):
    diag = ~np.eye(k, dtype=np.bool)
    coords = np.stack(np.where(diag), -1)
    return coords
        
def sample_random_primitive(parser, compiler, valid_draws,
                            valid_transforms, lang_type="CSG3D", 
                            allow_mirror=False, allow_mirror_macro=False):
    
    count = 0
    while(True):
        if lang_type == "CSG3Dx":
            expr = parser.sample_only_primitive(valid_draws)
            program = parser.parse(expr)
            compiler.reset()
            compiler._compile(program)
            primitive = compiler.get_output_sdf()
            if compiler.is_valid_primitive(primitive):
                bbox = compiler.draw.return_bounding_box(primitive)
                transform_expr = parser.get_random_transforms(valid_transforms, min_count=2, bbox=bbox)
                expr = transform_expr + expr
        elif lang_type in ["NRCSG3D", "MNRCSG3D", "MCSG3D", "HCSG3D"]:
            expr = parser.sample_only_primitive(valid_draws)
            bbox = None
            for transform_type in valid_transforms:
                if np.random.sample() <= TRANSFORM_SAMPLING_RATE[transform_type]:
                    program = parser.parse(expr)
                    compiler.reset()
                    compiler._compile(program)
                    primitive = compiler.get_output_sdf()
                    bbox = compiler.draw.return_bounding_box(primitive)
                    transform_expr, bbox = parser.get_transform(transform_type, bbox=bbox)
                    expr = transform_expr + expr
            if lang_type == "MCSG3D":
                if np.random.sample() <= MIRROR_SAMPLING_RATE:
                    if allow_mirror:
                        transform_expr, bbox = parser.get_mirror_transform(bbox=bbox)
                        expr = transform_expr + expr
                    elif allow_mirror_macro:
                        transform_expr, bbox = parser.get_macro_mirror(bbox=bbox)
                        expr = transform_expr + expr
        elif lang_type == "NTCSG3D":
            expr = parser.sample_random_primitive(valid_draws, valid_transforms)
        elif lang_type in ["PCSG3D", "FCSG3D"]:
            expr = parser.sample_random_primitive(valid_draws)
        elif lang_type in ["HCSG3D", "MCSG3D"]:
            expr = parser.sample_random_primitive(valid_draws)

        program = parser.parse(expr)
        compiler.reset()
        compiler._compile(program)
        primitive = compiler.get_output_sdf()
        count +=1 
        if compiler.is_valid_primitive(primitive):
            # print("Found after %d attempts" % count)
            break
    return primitive, expr

def get_valid_actions(f1, f2, compiler, threshold, threshold_diff):
    
    op_list = [] # self.action_sim.op[op_sym], op_sym]]
    for ind, op_sym in enumerate(boolean_commands):
        op = compiler.boolean_to_execute[op_sym]
        
        output = op(f1, f2)
        if valid_operation(output, f1, f2, threshold, threshold_diff):
            op_list.append([op, op_sym])
    return op_list

# TODO: Refactor this as its used at multiple places.
def valid_operation(output, top, bottom, threshold, threshold_diff):
    total = output.nelement()
    output_shape = (output <= 0)
    top_shape = (top <= 0)
    bottom_shape = (bottom <= 0)
    val_1 = torch.sum(output_shape) / total
    val_2 = torch.sum(output_shape ^ top_shape) / torch.sum(output_shape)
    val_3 = torch.sum(output_shape ^ bottom_shape) / torch.sum(output_shape)
    cond_1 = val_1 > threshold # < 40*40
    # Also not too small
    # print("\% occupied after operation", val_1)
    # print("\% difference", val_2, val_3)
    cond_2 = val_2 > threshold_diff
    cond_3 = val_3 > threshold_diff
    output = cond_1 and cond_2 and cond_3
    return output
    
def execute(expr, parser, compiler, return_numpy=False, return_shape=True):
    program = parser.parse(expr)
    compiler._compile(program)
    if return_shape:
        primitive = compiler.get_output_shape()
    else:
        primitive = compiler.get_output_sdf()
    if return_numpy:
        primitive = primitive.cpu().numpy().copy()
    return primitive

def is_valid_expr(expr, parser, compiler):
    sdf = execute(expr, parser, compiler,return_shape=False)
    return compiler.is_valid_sdf(sdf)
        
def generate_programs_func(proc_id, program_lengths, parser, compiler, action_space, save_files, total_samples,
                           valid_draws, valid_transforms, lang_type="CSG3D", 
                           allow_mirror=False, allow_mirror_macro=False, max_action_length=144):
    

    programs = defaultdict(list)
    reject_count = 0
    random_lens = np.random.choice(program_lengths, size=total_samples)
    main_ind = 0
    while(main_ind < len(random_lens)):
        cur_len = random_lens[main_ind]
        if main_ind % 100 == 0:
            print("Proc ID %d, Generating Program %d of %d " % (proc_id, main_ind, total_samples))
            print("Proc ID %d, Rejected %d , Accepted: %d" % (proc_id, reject_count, main_ind))
        created = False
        # don't stop until you create one of the length:
        while not created:
            created = False
            cur_program = []
            primitive_set = []
            expression_list = []
            # first sample the shapes:
            for j in range(cur_len +1):
                ## sample primitive:
                primitive, expr = sample_random_primitive(parser, compiler, valid_draws, valid_transforms, lang_type, allow_mirror, allow_mirror_macro)
                primitive_set.append(primitive)
                expression_list.append(expr)
                
            # Now try random combinations:
            size = len(primitive_set)
            permutation_index_tuple = get_permutation_index_tuple(size)
            attempts = 0
            while len(primitive_set)>1:
                # breaking condition:
                size = len(primitive_set)
                if attempts >= len(permutation_index_tuple):
                    print("Tried All %d permutations but none possible." % int(attempts))
                    created = False
                    break
                n1, n2 = permutation_index_tuple[attempts]
                if n1 > n2:
                    f1 = primitive_set.pop(n1)
                    exp_1 = expression_list.pop(n1)
                    
                    f2 = primitive_set.pop(n2)
                    exp_2 = expression_list.pop(n2)
                else:
                    f2 = primitive_set.pop(n2)
                    exp_2 = expression_list.pop(n2)
                    f1 = primitive_set.pop(n1)
                    exp_1 = expression_list.pop(n1)

                op_list = get_valid_actions(f1, f2, compiler, THRESHOLD, DIFF_THRESHOLD)
                
                if op_list:
                    ind = np.random.choice(range(len(op_list)))
                    op, op_sym = op_list[ind]
                    new_expr = [op_sym] + exp_1  + exp_2
                    program = parser.parse(new_expr)
                    compiler.reset()
                    compiler._compile(program)
                    primitive = compiler.get_output_sdf()
                    if lang_type in ["CSG3D", "NRCSG3D", "HCSG3D", "MCSG3D"]:
                        bbox = compiler.draw.return_bounding_box(primitive)
                        if np.random.uniform() < HIER_TFORM_RATE:
                            transform_expr = parser.get_random_transforms(valid_transforms, min_count=0, bbox=bbox)
                            new_expr = transform_expr + new_expr
                        if np.random.sample() < MIRROR_SAMPLING_RATE:
                            if allow_mirror:
                                transform_expr, bbox = parser.get_mirror_transform()
                                new_expr = transform_expr + new_expr
                            elif allow_mirror_macro:
                                transform_expr, bbox = parser.get_macro_mirror()
                                new_expr = transform_expr + new_expr

                    if not is_valid_expr(new_expr, parser, compiler):
                        new_expr = [op_sym] + exp_1  + exp_2

                    output = execute(new_expr, parser, compiler, return_shape=False)
                    primitive_set.append(output)
                    expression_list.append(new_expr)
                    attempts = 0
                    size = len(primitive_set)
                    # make set of all possible permutations of index
                    permutation_index_tuple = get_permutation_index_tuple(size)


                else:
                    primitive_set.append(f1)
                    primitive_set.append(f2)
                    expression_list.append(exp_1)
                    expression_list.append(exp_2)
                    attempts +=1 
                size = len(primitive_set)
                
            if len(primitive_set) == 1:
                created = True
            # Finally if its a good program add it to the stack
            if created:
                final_canvas = primitive_set[0]
                best_expression = expression_list[0]
                output_shape = (final_canvas <= 0)
                # best_expression, td_delta_sin, bu_delta_sin = gs_singular_parse(best_expression, output_shape, parser, compiler, reward_threshold=0)
                # final_canvas = self.action_sim.stack.pop()
                td_delta_sin, bu_delta_sin = 0 , 0
                cond_1 = output_shape.float().mean() > THRESHOLD
                # Do some other checks here:
                cond_2 = (td_delta_sin + bu_delta_sin) == 0
                # Also have a length check:

                actions = action_space.expression_to_action(best_expression)
                # print(len(actions))
                action_valid = len(actions)< max_action_length
                if cond_1 and cond_2 and action_valid:
                    main_ind += 1
                    programs[cur_len].append(best_expression)
                else:
                    # print("Rejected since occupancy of %f is < threshold %f" % (val_1, threshold))
                    created = False
                    reject_count += 1
            else:
                created = False
                reject_count += 1
        
    print("Proc id %d: Saving programs!" % proc_id)
    for key, value in programs.items():
        gen_programs = value
        save_file = save_files[key]
        print('Proc id %d: Saving %s' % (proc_id, save_file))
        with open(save_file, "w") as f:
            for cur_program in gen_programs:
                strng = "__".join(cur_program) + "\n"
                f.write(strng)
