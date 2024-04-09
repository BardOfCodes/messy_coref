
from collections import OrderedDict
import torch as th
# from .ShapeAssembly import Program, Cuboid
from .SA_mod import Program, Cuboid
import time
from collections import defaultdict
import numpy as np

class TypedCuboid(Cuboid):
    """
    Add a type for hierarchy
    """
    def __init__(self, name, aligned = False, vis = True, cuboid_type=0, device="cuda", dtype=th.float32, *args, **kwargs):

        super(TypedCuboid, self).__init__(name, aligned, vis, device=device, dtype=dtype,*args, **kwargs)
        self.cuboid_type = cuboid_type
        self.ref_counter = 0
        self.trans_counter = 0
        self.macro_parent = None
        self.macro_list = []
        self.has_subprogram = False
        self.subpr_sa_commands = []
        self.subpr_hcsg_commands = []
        self.sa_action_length = 0
        self.canonical_shape = None
    
class SAProgram(Program):

    def __init__(self, cuboids={}, device="cuda", dtype=th.float32):
        self.device = device
        self.dtype = dtype
        # For the cuboid
        self.CUBE_dims =  th.tensor([1.0, 1.0, 1.0], device=self.device, dtype=self.dtype)
        self.CUBE_pos = th.tensor([0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        self.CUBE_r_mat = th.eye(3, dtype=self.dtype, device=self.device)
        self.CUBE_noisy_eye = th.eye(3, dtype=self.dtype, device=self.device) * th.finfo(self.dtype).resolution
        self.CUBE_unit_zero = th.tensor(0., device=self.device, dtype=self.dtype)
        self.CUBE_dim_to_scale = {            
            "length": th.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype),
            "height": th.tensor([0.0, 1.0, 0.0], device=self.device, dtype=self.dtype),
            "width": th.tensor([0.0, 0.0, 1.0], device=self.device, dtype=self.dtype)
        }
        
        self.ft = {
            'right': (th.tensor([1.0, 0.5, 0.5], device=self.device, dtype=self.dtype), 0, 0.),
            'left': (th.tensor([0.0, 0.5, 0.5], device=self.device, dtype=self.dtype), 0, 1.),
            'top': (th.tensor([.5, 1.0, 0.5], device=self.device, dtype=self.dtype), 1, 0.),
            'bot': (th.tensor([.5, 0.0, 0.5], device=self.device, dtype=self.dtype), 1, 1.),
            'front': (th.tensor([.5, 0.5, 1.0], device=self.device, dtype=self.dtype), 2, 0.),
            'back': (th.tensor([.5, 0.5, 0.0], device=self.device, dtype=self.dtype), 2, 1.),
        }
        self.zeros_3 = th.zeros(3, device=self.device, dtype=self.dtype)
        # super(SAProgram, self).__init__(cuboids=cuboids, device=device, dtype=dtype)
        
        self.cuboids = self.getBoundBox()
        self.cuboids.update(cuboids)
        self.commands = []
        self.parameters = []
        self.att_points = {}
        
        # self.CUBE_rfnorm = th.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        # self.CUBE_tfnorm = th.tensor([0.0, 1.0, 0.0], device=self.device, dtype=self.dtype)
        # self.CUBE_ffnorm = th.tensor([0.0, 0.0, 1.0], device=self.device, dtype=self.dtype)
        self.command_to_func = {
            "sa_cuboid": self.executeCuboid,
            "sa_attach": self.executeAttach,
            "sa_squeeze": self.executeSqueeze,
            "sa_translate": self.executeTranslate,
            "sa_reflect": self.executeReflect
        }



    # Each program starts off with an invisible bounding box
    def getBoundBox(self):
        bbox = TypedCuboid("bbox", aligned = True, vis=False, device=self.device, dtype=self.dtype,
                            dims=self.CUBE_dims.clone(),
                            pos=self.CUBE_pos.clone(),
                            r_mat=self.CUBE_r_mat.clone(),
                            dim_to_scale=self.CUBE_dim_to_scale,
                            noisy_eye=self.CUBE_noisy_eye.clone(),
                            unit_zero=self.CUBE_unit_zero.clone(),)
        return OrderedDict(bbox=bbox)

    def execute_with_params(self, sa_command, params):
        self.command_to_func[sa_command](params)
    # Construct a new cuboid, add it to state
    def executeCuboid(self, parse):
        name = parse[0]
        cuboid_type = parse[4]
        x, y, z = parse[1], parse[2], parse[3]
        if name in self.cuboids:
            c = self.cuboids[name]            
            c.dims = th.stack((x, y, z))
            
        else:            
            c = TypedCuboid(
                name,
                aligned = False,
                cuboid_type=cuboid_type,
                device=self.device,
                dtype=self.dtype,
                dims=self.CUBE_dims.clone(),
                pos=self.CUBE_pos.clone(),
                r_mat=self.CUBE_r_mat.clone(),
                dim_to_scale=self.CUBE_dim_to_scale,
                noisy_eye=self.CUBE_noisy_eye.clone(),
                unit_zero=self.CUBE_unit_zero.clone(),
            )
            
            c.scaleCuboid(th.stack((x, y, z)))

            self.cuboids.update({
                name: c
            })
            
    
    # Executes a reflect line by making + executing new Cuboid and attach lines
    def executeReflect(self, parse):        
        c = self.cuboids[parse[0]]  
        cuboid_type = c.cuboid_type      
        assert c.name != "bbox", 'tried to move the bbox'
        
        rdir = self.getRefDir(parse[1])
        cnum = len(self.cuboids) - 1

        ref_name = c.name + "_ref_%s_%d" % (parse[1], c.ref_counter)
        c.ref_counter += 1

        self.executeCuboid([ref_name, c.dims[0].clone(), c.dims[1].clone(), 
                            c.dims[2].clone(), cuboid_type])
                        
        self.cuboids[ref_name].parent = c.name
        self.cuboids[ref_name].parent_axis = parse[1]
        # self.cuboids[ref_name].macro_parent = c.name
        # c.macro_list.append([ref_name])
        
        atts = c.move_atts
        for att in atts:
            
            if parse[1] == 'X':
                x = 1 - att[0].x.clone()
                y = att[0].y.clone()
                z = att[0].z.clone()
            elif parse[1] == "Y":
                x = att[0].x.clone()
                y = 1 - att[0].y.clone()
                z = att[0].z.clone()
            
            elif parse[1] == "Z":
                x = att[0].x.clone()
                y = att[0].y.clone()
                z = 1 - att[0].z.clone()

            n = att[2]

            cpt = att[0].getPos().clone()
            rpt = self.reflect_point(cpt, self.cuboids['bbox'].pos.clone(), rdir)
            
            rrpt = self.cuboids[n].getRelPos(rpt, True)
            
            self.executeAttach([ref_name, f'{n}', x, y, z, rrpt[0], rrpt[1], rrpt[2]])
            # what if we just instead copy variables and change them?
            
    # Executes a translate line by making + executing new Cuboid and attach lines
    def executeTranslate(self, parse):
        
        c = self.cuboids[parse[0]]
        cuboid_type = c.cuboid_type     
        assert c.name != "bbox", 'tried to move the bbox'
        tdir, td = self.getTransDir(parse[1])

        N = parse[2]
        scale = (td * parse[3]) / float(N)
        # all_names = []
        for i in range(1, N+1):
        
            new_name = c.name + "_trans_%d_%d" % (c.trans_counter, i)

            self.executeCuboid([new_name, c.dims[0].clone(), c.dims[1].clone(), c.dims[2].clone(), cuboid_type]) 
            
            atts = c.move_atts
            for att in atts:
                x = att[0].x
                y = att[0].y
                z = att[0].z
                n = att[2]

                cpt = att[0].getPos()
                rpt = cpt + (tdir * scale * i)

                rrpt = self.cuboids[n].getRelPos(rpt, True)

                self.executeAttach([new_name, f'{n}', x, y, z, rrpt[0], rrpt[1], rrpt[2]])
            # check the positions and other variables.
            
        c.trans_counter += 1
        # c.macro_list.append(all_names)
    def runProgram(self, param_lines):
        raise ValueError("Execution will give errors (Cuboid init is different)")
