import copy
import numpy as np
# import meshplot as mp
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
# Random visualize predictions
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import torch as th
import cv2


def viz_inverse_points(sdf_pc, sdf_mask, colors=None, plot=None, point_size=0.25, bbox=None):
    m = [-1, -1, -1]
    ma = [1, 1, 1]
    top_bbox = np.array([m , ma]) 
    v_box, f_box = bbox_to_draw_array(top_bbox)
    # Corners of the bounding box
    sdf_mask_col = np.random.rand(*sdf_mask.shape)
    sdf_mask_col[:,0] = 255
    sdf_mask_col[:,2] = 0 
    sdf_pc += np.random.uniform(size=sdf_pc.shape) * 0.01
    if colors is None:
        sdf_pc_col = np.random.rand(*sdf_pc.shape)
        sdf_pc_col[:,0] = 0
        sdf_pc_col[:,2] = 255 
    if plot is None:
        all_points = np.concatenate([sdf_pc, sdf_mask], 0)
        all_colors = np.concatenate([sdf_pc_col, sdf_mask_col], 0)
        plot = mp.plot(all_points, c=all_colors, shading={"point_size": point_size}, return_plot=True)
    else:
        mp.plot(all_points, c=all_colors, shading={"point_size": point_size}, return_plot=False)
#         mp.plot(sdf_pc, c=colors, shading={"point_size": 0.5}, return_plot=False)
    plot.add_edges(v_box, f_box, shading={"line_color": "red", "line_width": 1});
    if not bbox is None:
        if isinstance(bbox, list):
            for cur_bbox in bbox:
                sel_v_box, sel_f_box = bbox_to_draw_array(cur_bbox)
                plot.add_edges(sel_v_box, sel_f_box, shading={"line_color": "red", "line_width": 3});
        else:
            sel_v_box, sel_f_box = bbox_to_draw_array(bbox)
            plot.add_edges(sel_v_box, sel_f_box, shading={"line_color": "red", "line_width": 5});
    return plot

def render_stl(stl_file, image_file, ):
    try:
        figure = plt.figure(figsize=(10, 10))
        axes = figure.add_subplot(projection='3d')

        # Load the STL files and add the vectors to the plot
        your_mesh = mesh.Mesh.from_file(stl_file)
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

        # Auto scale to the mesh size
        bbox_mesh = mesh.Mesh.from_file("bbox_file.stl")
        scale = bbox_mesh.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        # Show the plot to the screen
        pyplot.savefig(image_file,  transparent=True)
    except:
        image = np.zeros([512, 512, 3])
        cv2.imwrite(image_file, image)

    pyplot.close("all")


def bbox_to_draw_array(bbox):
    
    m = bbox[0]
    ma = bbox[1]
    v_box = np.array([[m[0], m[1], m[2]], [ma[0], m[1], m[2]], [ma[0], ma[1], m[2]], [m[0], ma[1], m[2]],
                      [m[0], m[1], ma[2]], [ma[0], m[1], ma[2]], [ma[0], ma[1], ma[2]], [m[0], ma[1], ma[2]]])

    # Edges of the bounding box
    f_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], 
                      [7, 4], [0, 4], [1, 5], [2, 6], [7, 3]], dtype=np.int)
    
    return v_box, f_box

def viz_points(sdf_pc, colors=None, plot=None, point_size=0.25, bbox=None):

    if sdf_pc.shape[0] ==0:
        print("NO POINTS")
        return 0
    m = [-1, -1, -1]
    ma = [1, 1, 1]
    top_bbox = np.array([m , ma]) 
    v_box, f_box = bbox_to_draw_array(top_bbox)
    # Corners of the bounding box
    if colors is None:
        colors = np.random.rand(*sdf_pc.shape)
    if plot is None:
        plot = mp.plot(sdf_pc, c=colors, shading={"point_size": point_size}, return_plot=True)
    else:
        mp.plot(sdf_pc, c=colors, shading={"point_size": point_size}, return_plot=False)
#         mp.plot(sdf_pc, c=colors, shading={"point_size": 0.5}, return_plot=False)
    plot.add_edges(v_box, f_box, shading={"line_color": "red", "line_width": 1});
    if not bbox is None:
        if isinstance(bbox, list):
            for cur_bbox in bbox:
                sel_v_box, sel_f_box = bbox_to_draw_array(cur_bbox)
                plot.add_edges(sel_v_box, sel_f_box, shading={"line_color": "red", "line_width": 3});
        else:
            sel_v_box, sel_f_box = bbox_to_draw_array(bbox)
            plot.add_edges(sel_v_box, sel_f_box, shading={"line_color": "red", "line_width": 5});
    return plot

def draw_graph(graph, label_dict, figsize=(15, 15)):
    # same layout using matplotlib with no labels
    pos = graphviz_layout(graph, prog='dot')
    pos = {node: (x, y) for node, (x,y) in pos.items()}
    fig, ax = plt.subplots(figsize=figsize)
    # fig.set_figheight(15)
    # fig.set_figwidth(15)
    expr_sizes = [len(y) for x, y in label_dict.items()]
    node_sizes = [x * 200 for x in expr_sizes]
    nx.draw(graph, pos, 
            ax=ax,
            arrows=True,
            arrowstyle="-",
#             min_source_margin=15,
#             min_target_margin=15,
            font_size=12,
            node_size=node_sizes,
            labels=label_dict, with_labels=True)
    tr_figure = ax.transData.transform
    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform
    icon_size = 0.1 # (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025
    icon_center = icon_size / 2.0

    # Add the respective image to each node
    for n in graph.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        # get overlapped axes and plot icon
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        # a.imshow(graph.nodes[n]["image"])
        temp_c = graph.nodes[n]["node_id"]
    #     a.imshow(image_dict[temp_c])
        # a.set_title(graph.nodes[n]['symbol'])
        a.axis("off")
    return fig
    
def draw_graph_with_images(graph, label_dict, image_dict, figsize=(15, 15), figure_title="CSG TREE"):
    # same layout using matplotlib with no labels
    pos = graphviz_layout(graph, prog='dot')
    pos = {node: (x, y) for node, (x,y) in pos.items()}
    fig, ax = plt.subplots(figsize=figsize)
    fig.tight_layout()
    ax.set_title(figure_title, y=-0.01)
    ax.title.set_size(30)
    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])
    expr_sizes = [len(y) for x, y in label_dict.items()]
    node_sizes = []
    for n in graph.nodes:
        if "image" in graph.nodes[n].keys():
            node_sizes.append(5000)
        else:
            node_sizes.append(0)
    # print(node_sizes)
    # node_sizes = [x * 500 for x in expr_sizes]
    small_label_dict = {x:y[:5] for x, y in label_dict.items()}
    nx.draw(graph, pos, 
            ax=ax,
            arrows=True,
            arrowstyle="-",
            min_source_margin=0,
            min_target_margin=0,
            font_size=20,
            node_size=node_sizes,
            labels=small_label_dict, with_labels=True)
    tr_figure = ax.transData.transform
    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform
    # The levels
    print("here", (ax.get_xlim()[1] - ax.get_xlim()[0]))
    icon_size = 0.15# (ax.get_xlim()[1] - ax.get_xlim()[0]) #* 0.025
    icon_center = icon_size / 2.0

    # Add the respective image to each node
    z = graph.nodes# list(range(len(graph.nodes)))
    for n in z:
        if n in image_dict.keys():
            xf, yf = tr_figure(pos[n])
            xa, ya = tr_axes((xf, yf))
            size_multiplier = graph.nodes[n]['height'] * 3 / 10
            # get overlapped axes and plot icon
            cur_icon_center = max(icon_center * size_multiplier, 0.05)

            cur_icon_size = max(icon_size * size_multiplier, 0.1)
            print('Sizes', [xa - cur_icon_center, ya - cur_icon_center, cur_icon_size, cur_icon_size])
            a = plt.axes([xa - cur_icon_center, ya - cur_icon_center, cur_icon_size, cur_icon_size])
            image = image_dict[n]
            a.imshow(image)
            a.margins(x=-.1)
            a.margins(y=-.1)
            # plt.gca().set_position((0, 0, 1, 1))
            subfig = plt.gcf()
            subfig.tight_layout()
            a.set_title(graph.nodes[n]['name'], y=0.1, x=0.75 )
            a.title.set_size(15)
            a.axis("off")
    return fig

def visualize_tree_with_figures(expression, target, compiler, parser, rotation_param=[0, 0, 0], size_ratio=4, figure_title="CSG TREE"):
    # part 1: Create complex graph:
    rotation_command = {"type": "T", "symbol": "rotate", "param": rotation_param}

    target = th.from_numpy(target).cuda()
    compiler.reset()
    command_list = parser.parse(expression)
    size_ratio = len(command_list) /3.
    # complex_graph = compiler.command_tree(command_list, target.clone(), add_splicing_info=True)
    complex_graph = compiler.vis_tree(command_list, target.clone(), add_splicing_info=False)
    # Get the list of all booleans
    complex_graph.nodes[1]["subexpr_info"]['commands'] = copy.deepcopy(command_list)
    # complex_graph.nodes[1]["subexpr_info"]['commands'] = command_list

    # generate_images(complex_graph)
    image_dict = {}
    print(complex_graph.nodes)
    for node_id in complex_graph.nodes:
        cur_node = complex_graph.nodes[node_id]
        if cur_node["subexpr_info"]:
            if cur_node["subexpr_info"]['draw']:
                commands = cur_node["subexpr_info"]['commands']
                commands.insert(0, rotation_command)
                # print(node_id, commands)
                compiler.write_to_stl(commands, "tmp.stl")
                render_stl("tmp.stl", "tmp%d.png" % node_id)
                image = cv2.imread("tmp%d.png" % node_id)
                image_dict[node_id] = image.copy()

    target = th.stack([target, target * 0 + 1], -1)
    target = compiler.draw.shape_rotate(rotation_param, target, inverted=False)
    target = target[:, :, :, 0]
    image = get_target_image(target, compiler)
    image_dict[0] = image.copy()
    complex_graph.nodes[0]["subexpr_info"]['commands'] = [rotation_command] + command_list
    # complex_graph.nodes[0]["subexpr_info"]['commands'] = command_list

    # simple_graph = compiler.command_tree(command_list, target.clone(), add_splicing_info=False)

    simple_graph = compiler.vis_tree(command_list, target.clone(), add_splicing_info=False)
    # n_list = list(range(len(simple_graph.nodes)))
    # print(complex_graph.nodes)
    for n in simple_graph.nodes:
        if n not in image_dict:
            # simple_graph.remove_node(n)
            cur_node['height'] = 0.0001
            cur_node['width'] = 0.0001
        else:
            cur_node = simple_graph.nodes[n]
            complex_node = complex_graph.nodes[n]
            if complex_node['subexpr_info']:
                commands = complex_node['subexpr_info']['commands']
                size = max(len(commands)/ size_ratio, 3)
            else:
                size = 3
            # size = size * 2
            cur_node['height'] = size
            cur_node['width'] = size
    # For Root add the image of target:
    # rotate shape by the same amount:
    label_dict = {simple_graph.nodes[x]['node_id']: "_".join([simple_graph.nodes[x]['symbol'],str(simple_graph.nodes[x]['node_id'])])   for x in simple_graph.nodes}

    for key, value in image_dict.items():
        print(key, value.shape)
    fig = draw_graph_with_images(simple_graph, label_dict, image_dict, figsize=(30, 15), figure_title=figure_title)
    return fig


def get_target_image(target, compiler):
    # Post rotation, bool th:
    target = -(target.float() - 0.1)

    image_file = "tmp0.png"
    pred_points = compiler.draw.return_inside_coords(target)
    colors = pred_points[:,:]  + 0.05 * np.random.uniform(size=pred_points.shape)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')
    ax.scatter(pred_points[:,1], pred_points[:,0] * -1, pred_points[:,2], s=0.5, c=pred_points[:,2], cmap='copper')
    ax.set_xlim(left=-1.0, right=1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    # plt.show()
    plt.tight_layout()
    plt.savefig(image_file,  transparent=True)
    plt.close("all")
    image = cv2.imread(image_file)
    return image



def merge_two_points(v1, v2):
    sdf_pc = np.concatenate([v1, v2], 0)
    c_1 = np.random.rand(*v1.shape)
    c_1[:,0] = 255
    c_1[:,2] = 0 
    c_2 = np.random.rand(*v2.shape)
    c_2[:,2] = 255
    c_2[:,0] = 0
    sdf_pc += np.random.uniform(size=sdf_pc.shape) * 0.01
    colors = np.concatenate([c_1, c_2], 0)
    return sdf_pc, colors