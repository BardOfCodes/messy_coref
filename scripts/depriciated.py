
def measure_cd():
    ply_path = "/home/aditya/eval_meshes_pr/"
    pointcloud_path = "/home/aditya/eval_points/"

    item_names = os.listdir(pointcloud_path)

    distances_1 = []
    distances_2 = []

    num_surface_points = 2048

    for ind, item in enumerate(item_names):
        print(ind)
        gt_name = os.path.join(pointcloud_path, "%s" % item)
        pred_name = os.path.join(ply_path, "%s" % item)
        
        
        points = o3d.io.read_point_cloud(gt_name)
        points = np.asarray(points.points)# .points
        surface_selection_index = np.random.randint(0, points.shape[0], num_surface_points)
        
        gt_points = points[surface_selection_index].astype(np.float32)
        # gt_points = np.asarray(pcd.points, dtype=np.float32)
        
        mesh = o3d.io.read_triangle_mesh(pred_name)
        pcd = mesh.sample_points_poisson_disk(num_surface_points)
        pred_points = np.asarray(pcd.points, dtype=np.float32)
        
        # calculate CD
        source_cloud = th.tensor(gt_points).unsqueeze(0).cuda()
        target_cloud = th.tensor(pred_points).unsqueeze(0).cuda()
        chamferDist = ChamferDistance()
        distance_1, distance_2 = chamferDist(source_cloud, target_cloud)
        distance_1 = distance_1.mean()
        distance_2 = distance_2.mean()
        distances_1.append(distance_1.item())
        distances_2.append(distance_2.item())
        
        cd1 = np.sum(np.stack(distances_1))/len(distances_1)
        cd2 = np.sum(np.stack(distances_2))/len(distances_2)
        cd = (cd1+cd2)*500
        print("CD", cd)
        
