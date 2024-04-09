from collections import defaultdict
import numpy as np
import sys
import random
import torch
import torch as th

from .parser import convert_sa_to_valid_hsa
import itertools
device = torch.device('cuda')

# 3D CSG Executor Logic
MIN_UNIQUE_VOXELS = 4
MAX_ACTION_LENGTH = 192
MAX_MASTER_PROGRAM_SIZE = {
    "HSA3D": 60,
    "PSA3D": 144
    }
MAX_SUB_PROGRAM_SIZE = 40


# Define Language Tokens + Expected Voxel Dimension

DIM = 32
MAX_PRIMS = 10
MAX_SYM = 4

a = (torch.arange(DIM).float() / (DIM-1.)) - .5
b = a.unsqueeze(0).unsqueeze(0).repeat(DIM, DIM, 1)
c = a.unsqueeze(0).unsqueeze(2).repeat(DIM, 1, DIM)
d = a.unsqueeze(1).unsqueeze(2).repeat(1, DIM, DIM)
pts = torch.stack((b,c,d), dim=3).view(-1, 3).to(device)

FACES = ['left','right','bot','top','back','front']
BBOX_FACE_DIST = np.array([0.,0.,0.5,0.5,0.,0.])
GEN_FACE_DIST = np.array([.125,.125,0.25,0.25,.125,.125])

AXES = ['X','Y','Z']
AXES_DIST = np.array([0.6,.2,.2])

PART_TYPES = ['Prim,Move,Reflect', 'Prim,Move,Translate', 'Prim,Move']
PART_DIST = np.array([0.25, 0.1, 0.65])

MOVE_TYPES = ['Attach','Attach,Attach','Squeeze']
MOVE_DIST = np.array([0.4,0.25,0.35])


START_TOKEN = 'START'

# TOKENS describe the semantics of the language
# {name: (input types, output type)}
# name -> name of the token
# input types -> expected types of the arguments to the token
# output types -> the output type of the token

TOKENS = {
    START_TOKEN: ('Part', '', ''),
    'END': ('', 'Part', ''),
        
    'Cuboid': ('cuboid_fnum,cuboid_fnum,cuboid_fnum','Prim','Cuboid('),
    'Attach': ('cind,face,fpos,fpos','Att','attach('),
    'Squeeze': ('cind,cind,face,fpos','Sq','squeeze('),
    'Reflect': ('axis','Sym','reflect('),
    'Translate': ('axis,snum,fnum','Sym','translate('),
}
TOKENS[str(DIM)] = ('', 'fnum', 1.0)

for i in range(1,DIM):
    TOKENS[f'{i}'] = ('', 'fnum', (i * 1.)/DIM)

TOKENS[f'bbox'] = ('', 'cind', f'bbox')
for i in range(0,MAX_PRIMS):
    TOKENS[f'cube{i}'] = ('', 'cind', f'cube{i}')


for i in range(1,10):
    for j in range(1,10):
        TOKENS[f'fpos_{i}_{j}'] = ('', 'fpos', (
            (i * 1.)/10., (j * 1.)/10.
        ))

for face in FACES:
    TOKENS[face] = ('', 'face',face)

for axis in AXES:
    TOKENS[axis] = ('', 'axis',axis)

for i in range(1, MAX_SYM+1):
    TOKENS[f'snum_{i}'] = ('', 'snum', i)
    
I2T = {i:t for i,t in enumerate(TOKENS)}
T2I = {t:i for i,t in I2T.items()}



# START SYNTHETIC PROG CREATION LOGIC

O2T = {}

for token, (inp_types, out_type, out_values) in TOKENS.items():
    if token in ['START','END']:
        continue
    
    if out_type not in O2T:
        O2T[out_type] = []

    O2T[out_type].append(token)

O2T['cuboid_fnum'] = O2T['fnum']
# Start LANG Execution Logic

def makeCuboidLine(params, cc):
    x = float(params[0]) *1. / DIM
    y = float(params[1]) *1. / DIM
    z = float(params[2]) *1. / DIM
    return f'cube{cc} = Cuboid({x}, {y}, {z}, False)'

def makeAttParams(face,fpos1,fpos2,flip):
    _,x1,y1 = fpos1.split('_')
    _,x2,y2 = fpos2.split('_')
    x1 = float(x1) / 10.
    y1 = float(y1) / 10.
    x2 = float(x2) / 10.
    y2 = float(y2) / 10.
    
    att = [None,None,None,None,None,None]

    if face == 'left':
        I = 3
        J = 0
        A = 1
        B = 2
        C = 4
        D = 5
        
    elif face == 'right':
        I = 0
        J = 3
        A = 1
        B = 2
        C = 4
        D = 5

    elif face == 'bot':
        I = 4
        J = 1
        A = 0
        B = 2
        C = 3
        D = 5

    elif face == 'top':
        I = 1
        J = 4
        A = 0
        B = 2
        C = 3
        D = 5

    elif face == 'back':
        I = 5
        J = 2
        A = 0
        B = 1
        C = 3
        D = 4

    elif face == 'front':
        I = 2
        J = 5
        A = 0
        B = 1
        C = 3
        D = 4

    if flip and face == 'bot':
        att[I] = 0.
        att[J] = 0.
    elif flip and face == 'top':
        att[I] = 1.
        att[J] = 1.
    else:
        att[I] = 1.0
        att[J] = 0.0
        
    att[A] = x1
    att[B] = y1
    att[C] = x2
    att[D] = y2
        
    return att
    
def makeAttachLine(params, cc):
    cind = params[0]
    att_prms = makeAttParams(params[1],params[2],params[3],cind == 'bbox')
    
    return f'attach(cube{cc-1}, {cind}, {att_prms[0]}, {att_prms[1]}, {att_prms[2]}, {att_prms[3]}, {att_prms[4]}, {att_prms[5]})' 

def makeSqueezeLine(params, cc):

    cind1 = params[0]
    cind2 = params[1]
    face = params[2]
    _,x1,y1 = params[3].split('_')
    x = float(x1) / 10.
    y = float(y1) / 10.
    
    return f'squeeze(cube{cc-1}, {cind1}, {cind2}, {face}, {x}, {y})'

def makeReflectLine(params, cc):
    return f'reflect(cube{cc-1}, {params[0]})'

def makeTranslateLine(params, cc):
    num = params[1].split('_')[1]
    d = float(params[2]) * 1. / DIM
    return f'translate(cube{cc-1}, {params[0]}, {num}, {d})'

def tokens_to_lines(tokens):
    lines = []

    start = 0

    cc = 0
    
    while start < len(tokens):        
        t = tokens[start]

        if t == 'START':
            bbDim = float(tokens[start+1]) * 1. / DIM
            lines.append(f'bbox = Cuboid(1., {bbDim}, 1., False)')
            start += 2
            
        elif t == 'Cuboid':
            lines.append(makeCuboidLine(tokens[start+1:start+4], cc))
            start += 4
            cc += 1
            
        elif t == 'Attach':
            lines.append(makeAttachLine(tokens[start+1:start+5], cc))
            start += 5

        elif t == 'Squeeze':
            lines.append(makeSqueezeLine(tokens[start+1:start+5], cc))
            start += 5

        elif t == 'Reflect':
            lines.append(makeReflectLine(tokens[start+1:start+2], cc))
            start += 2

        elif t == 'Translate':
            lines.append(makeTranslateLine(tokens[start+1:start+4], cc))
            start += 4

        elif t == 'END':
            break
            
        else:
            assert False
                        
    return lines                                     

# Helper function to sample a bounded normal distribution
def norm_sample(mean, std, mi, ma):
    v = None

    if mi == ma:
        return mi
    
    while True:        
        v = round(max(mean,1) + (np.random.randn() * max(std,1)))
        if v >= mi and v <= ma:
            break

    return v


def samplePartType():
    return np.random.choice(PART_TYPES,p=PART_DIST)

def sampleMoveType():    
    return np.random.choice(MOVE_TYPES,p=MOVE_DIST)

def sampleCubInd(prev_prims, last_cind):
    l = ['bbox']
    for i in [f'cube{i}' for i  in range(prev_prims)]:
        if i != last_cind:
            l.append(i)

    return random.sample(l,1)[0]

def sampleFace(last, last_face):
    if last == 'bbox':
        dist = BBOX_FACE_DIST
    else:
        dist = GEN_FACE_DIST 

    if last_face is not None:
        dist = dist.copy()
        dist[FACES.index(last_face)] = 0.
        dist /= dist.sum()
        
    return np.random.choice(FACES, p=dist)

def sampleAxis():
    return np.random.choice(AXES, p=AXES_DIST)

def sampleBBDim():

    d = norm_sample(
        20,
        8,
        8,
        DIM - 1
    )
    
    return str(d)
        

def sampleFNum():
    H = DIM//2
    
    center = norm_sample(H,H/3,2,DIM-2)    
    
    d = norm_sample(
        min(DIM-center, center) * 0.75,
        min(DIM-center, center) / 2,
        2, 
        min(DIM-center,center) * 2 - 2,
    )    
    return str(d)


def sampleCuboidFNum(cuboid_type):
    if cuboid_type == 0:
        H = DIM//2
        
        center = norm_sample(H,H/3,2,DIM-2)    
        
        d = norm_sample(
            min(DIM-center, center) * 0.75,
            min(DIM-center, center) / 2,
            2, 
            min(DIM-center,center) * 2 - 2,
        )    
    else:
        H = DIM//2
        d = 0
        while (d < DIM //3):
            center = norm_sample(H,H/3,2,DIM-2)    
            d = norm_sample(
                min(DIM-center, center) * 0.75,
                min(DIM-center, center) / 2,
                2, 
                min(DIM-center,center) * 2 - 2,
            )    

    return str(d)

def sampleFPos(paired):
    r = random.random() <= 0.5

    if r:
        if paired:
            return f'fpos_5_5'
        else:
            return random.sample(O2T['fpos'],1)[0]


    i = norm_sample(5,2,1,9)
    j = norm_sample(5,2,1,9)

    return f'fpos_{i}_{j}'

def samplePart(prev_prims, cuboid_type):    
    q = [samplePartType()]

    r = []

    c = 0

    last_cind = None
    last_face = None
    lastT = None
    
    while len(q) > 0:
        p = q.pop(0)
        
        c += 1
        
        if ',' in p:
            q = p.split(',') + q
            
        elif p in TOKENS:
            r.append(p)
            ninp = TOKENS[p][0]

            if ninp == '':
                continue
            
            _q = []            
            for i in ninp.split(','):
                _q.append(i)
            q = _q + q
                
        elif p in O2T:
            # State dependant command
                        
            if p == 'cind':
                n = sampleCubInd(prev_prims, last_cind)
                last_cind = n
                
            elif p == 'face':
                n = sampleFace(last_cind, last_face)
                last_face = n
                
            elif p == 'axis':
                n = sampleAxis()

            elif p == 'fnum':
                n = sampleFNum()
            elif p == 'cuboid_fnum':
                n = sampleCuboidFNum(cuboid_type)

            elif p == 'fpos':
                n = sampleFPos(lastT == 'fpos')
                
            else:
                n = random.sample(O2T[p],1)[0]
                
            q = [n] + q
            lastT = p
            
        else:
            assert p == 'Move'
            n = sampleMoveType()
            q = [n] + q

    return r

# Master Program
def sample_prog(max_tokens, num_prims, num_sub_programs, sampleBB=True):
    
    cuboid_types = [0 for x in range(num_prims)]
    subpr = random.sample(range(num_prims), num_sub_programs)
    for i in subpr:
        cuboid_types[i] = 1
    while(True):
        if sampleBB:
            tokens = ['START', sampleBBDim()]
        else:
            tokens = ['START', DIM]

        if sampleBB:
            for i in range(num_prims):
                tokens += samplePart(i, cuboid_types[i])    
        else:
            for i in range(num_prims):
                tokens += samplePart(i, 1)       
            
        tokens.append('END')
        
        if len(tokens) < max_tokens:
            break
    lines = tokens_to_lines(tokens)
    valid_sa = convert_sa_to_valid_hsa(lines) # Adds 0 to cuboids and adds end token.
    # Add hierarchy information
    hier = 1
    for ind, cuboid_type in enumerate(cuboid_types):
        if cuboid_type > 0:
            line = ""
            search_phrase = "cube%d =" %  ind
            for jind, line in enumerate(lines):
                if search_phrase in line:
                    lines[jind] = line[:-2] + "%d)" % hier
                    hier += 1
                    break

    return valid_sa

# Main entrypoint to generate a dataste/batch of synthetic programs
# Returns data, either a list of tokens, or a list of tupled (voxels, tokens)
# Possible hierarchies for different number of primitives.
def get_all_hier_forms(max_cuboids=12, master_min_cuboids=4, master_max_cuboids=6, max_n_subprograms=4, suprogram_counts=[2, 3, 4]):
    all_forms = defaultdict(list)
    total_combos= 0
    for master_cuboids in range(master_min_cuboids, master_max_cuboids):
        for n_subpr in range(0, max_n_subprograms):
            n_combinations = list(itertools.combinations_with_replacement(suprogram_counts, n_subpr))
            for cur_combo in n_combinations:
                total_cuboids = master_cuboids - n_subpr + np.sum(cur_combo)
                total_cuboids = int(total_cuboids)
                if total_cuboids < max_cuboids:
                    all_forms[total_cuboids].append((master_cuboids, list(cur_combo)))
                    total_combos += 1
    print("Total %d hierarchical forms" % total_combos)
    return all_forms

def generate_programs_func(proc_id, program_lengths, parser, compiler, action_space, save_files, total_samples, allow_sub_program=True, max_action_length=MAX_ACTION_LENGTH):
    

    HIERARCHY_FORMS = get_all_hier_forms()
    programs = defaultdict(list)
    reject_count = 0
    random_lens = np.random.choice(program_lengths, size=total_samples)
    main_ind = 0
    compiler.set_to_half()
    compiler.reset()
    parser.set_device("cuda")
    parser.set_tensor_type(th.float16)

    unique_count = []
    action_count = []
    occ_count = []

    
    while(main_ind < len(random_lens)):
        cur_len = random_lens[main_ind]
        if main_ind % 100 == 0:
            print("Proc ID %d, Generating Program %d of %d " % (proc_id, main_ind, total_samples))
            print("Proc ID %d, Rejected %d , Accepted: %d" % (proc_id, reject_count, main_ind))
        # don't stop until you create one of the length:
        # Now given a random ind, we need to select the kind of break down:
        if cur_len <= 5:
            master_program_prims = cur_len
            n_sub_programs = 0
            sub_program_prims = []
            sel = (cur_len, [])
        elif cur_len >= 6:
            if allow_sub_program:
                possibilities = HIERARCHY_FORMS[cur_len]
                sel = random.choice(possibilities)
                master_program_prims = sel[0]
                n_sub_programs = len(sel[1])
                sub_program_prims = sel[1]
            else:
                master_program_prims = cur_len
                n_sub_programs = 0
                sub_program_prims = []
                sel = (cur_len, [])
        
        # Now we need to create master program & subprogram & link them
        # Find a way to specify large cubes for sub-programs. 
        # Generate master program:

        created = False
        if allow_sub_program:
            max_program_size = MAX_MASTER_PROGRAM_SIZE["HSA3D"]
        else:
            max_program_size = MAX_MASTER_PROGRAM_SIZE["PSA3D"]
        while not created:
            master_program = sample_prog(max_program_size, master_program_prims, n_sub_programs)
            sub_programs = []
            for sub_pr_prim in sub_program_prims:
                sub_pr = sample_prog(MAX_SUB_PROGRAM_SIZE, sub_pr_prim, 0, sampleBB=False)
                sub_programs.extend(sub_pr)
            
            best_expression = master_program + sub_programs
            # Check validity:

            try:
                command_list = parser.parse(best_expression)
                compiler._compile(command_list)
                shape = compiler._output
                occupancy = (shape<=0).float().mean().item()
                occupancy_valid = (occupancy < 0.5)  and (occupancy > 0.05)

                actions = action_space.expression_to_action(best_expression)
                # print(len(actions))
                action_valid = len(actions)< max_action_length
                # unique bbox:
                uniqueness_valid = compiler.per_level_check(command_list, MIN_UNIQUE_VOXELS)
                
                expression_valid = occupancy_valid and action_valid and uniqueness_valid

                unique_count.append(uniqueness_valid)
                action_count.append(action_valid)
                occ_count.append(occupancy_valid)
                if expression_valid: 
                    main_ind += 1
                    created = True
                    programs[cur_len].append(best_expression)
                else:
                    created = False
                    reject_count += 1
            except: 
                reject_count += 1

        
    print("Overall", "occupancy", np.mean(occ_count), "action", np.mean(action_count), "unique", np.mean(unique_count))
    print("Proc id %d: Saving programs!" % proc_id)
    for key, value in programs.items():
        gen_programs = value
        save_file = save_files[key]
        print('Proc id %d: Saving %s' % (proc_id, save_file))
        with open(save_file, "w") as f:
            for cur_program in gen_programs:
                strng = "__".join(cur_program) + "\n"
                f.write(strng)
