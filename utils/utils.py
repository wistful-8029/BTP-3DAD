import numpy as np
import os
import random
import shutil
import torch
import torch.distributed as dist
import torch.autograd as autograd

from PIL import ImageFilter
from easydict import EasyDict
import yaml
import open3d as o3d

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config
def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
              
        new_config = yaml.load(f, Loader=yaml.FullLoader)
                 
                                       
    merge_new_config(config=config, new_config=new_config)
    return config

def get_model(model):
    if isinstance(model, torch.nn.DataParallel)\
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(state, is_best, output_dir):
    if is_main_process():
        ckpt_path = '{}/checkpoint_{}.pt'.format(output_dir, state['epoch'])
        best_path = f'{output_dir}/checkpoint_best.pt'
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copyfile(ckpt_path, best_path)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def scaled_all_reduce(tensors, is_scale=True):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    world size.
    """
    world_size = get_world_size()
                                                            
    if world_size == 1:
        return tensors
                          
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
                                   
    for reduction in reductions:
        reduction.wait()
                       
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
                                
    world_size = get_world_size()
                                                            
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False                   
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
                                
    world_size = get_world_size()
                                                            
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_dataset(train_transform, tokenizer, args, dataset_name=None):
    from models.ulip_legacy.dataset_3d import Dataset_3D

    dataset_3d = Dataset_3D(args, tokenizer, dataset_name, train_transform)
    return dataset_3d.dataset


def compute_patch_FPFH(points: torch.Tensor,
                       patch_idx: torch.Tensor,
                       radius_normal=0.05,
                       radius_feature=0.1,
                       max_nn=30,
                       max_nn_feature=100,
                       agg='mean') -> torch.Tensor:
    """
    Compute patch-level FPFH features for point clouds.

    Args:
        points: torch.Tensor, [B, N, 3] point coordinates
        patch_idx: torch.Tensor, [B, num_patches, points_per_patch] patch indices
        radius_normal: float, normal-estimation radius
        radius_feature: float, FPFH radius
        max_nn: int, maximum neighbors for normal estimation
        max_nn_feature: int, maximum neighbors for FPFH
        agg: str, aggregation method, 'mean' or 'max'

    Returns:
        patch_feats: torch.Tensor, [B, num_patches, 33] patch features
    """
    device = points.device
    B, N, _ = points.shape
    num_patches, points_per_patch = patch_idx.shape[1], patch_idx.shape[2]
    patch_feats_list = []

    for b in range(B):
                                           
        pts_np = points[b].detach().cpu().numpy()         

                                      
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_np)

                           
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn)
        )

                       
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn_feature)
        )
        fpfh_np = np.asarray(fpfh.data).T.copy()          
        fpfh_tensor = torch.from_numpy(fpfh_np).float().to(device)

                                                                
        idx_expand = patch_idx[b].unsqueeze(-1).expand(-1, -1, 33)                                       
        fpfh_expand = fpfh_tensor.unsqueeze(0).expand(num_patches, -1, -1)                        

        patch_feat = torch.gather(fpfh_expand, 1, idx_expand)                                       

                                      
        if agg == 'mean':
            patch_feat = patch_feat.mean(dim=1)
        elif agg == 'max':
            patch_feat, _ = patch_feat.max(dim=1)
        else:
            raise ValueError(f"Unknown aggregation type: {agg}")

        patch_feats_list.append(patch_feat)

    patch_feats = torch.stack(patch_feats_list, dim=0).to(device)                        
    return patch_feats


def compute_patch_geofeatures(points: torch.Tensor,
                               patch_idx: torch.Tensor,
                               radius_normal=0.05,
                               radius_feature=0.1,
                               max_nn=30,
                               max_nn_feature=100,
                               agg='mean') -> torch.Tensor:
    """
    Compute patch-level geometric features with FPFH, normals, and curvature statistics.

    Args:
        points: [B, N, 3]
        patch_idx: [B, num_patches, points_per_patch]
        radius_normal: normal-estimation radius
        radius_feature: FPFH radius
        agg: 'mean' or 'max'

    Returns:
        patch_feats: [B, num_patches, F_total]
    """
    device = points.device
    B, N, _ = points.shape
    num_patches, points_per_patch = patch_idx.shape[1], patch_idx.shape[2]
    patch_feats_list = []

    for b in range(B):
        pts_np = points[b].detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_np)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn)
        )
        normals_np = np.asarray(pcd.normals)         

              
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn_feature)
        )
        fpfh_np = np.asarray(fpfh.data).T.copy()          

                                                                   
        idx_expand = patch_idx[b].unsqueeze(-1).expand(-1, -1, 33).to(device)                        
        fpfh_expand = torch.from_numpy(fpfh_np).float().to(device).unsqueeze(0).expand(num_patches, -1, -1)                      
        patch_fpfh = torch.gather(fpfh_expand, 1, idx_expand.to(torch.long))                       

                                                             
        normals_expand = torch.from_numpy(normals_np).float().to(device).unsqueeze(0).expand(num_patches, -1, -1)
        patch_normals = torch.gather(normals_expand, 1, patch_idx[b].unsqueeze(-1).expand(-1,-1,3).to(device).to(torch.long))

                                           
        patch_pts = points[b][patch_idx[b]]                       
        patch_centered = patch_pts - patch_pts.mean(dim=1, keepdim=True)
        cov = torch.matmul(patch_centered.transpose(1,2), patch_centered) / points_per_patch                     
        eigvals, _ = torch.linalg.eigh(cov)                   
        lam0, lam1, lam2 = eigvals[:,0], eigvals[:,1], eigvals[:,2]                                 
        linearity = ((lam2 - lam1) / (lam2 + 1e-6)).unsqueeze(-1)
        planarity = ((lam1 - lam0) / (lam2 + 1e-6)).unsqueeze(-1)
        sphericity = (lam0 / (lam2 + 1e-6)).unsqueeze(-1)
        curvature = (lam0 / (lam0 + lam1 + lam2 + 1e-6)).unsqueeze(-1)
        patch_curv = torch.cat([linearity, planarity, sphericity, curvature], dim=-1)                   

                                             
        if agg == 'mean':
            patch_fpfh = patch_fpfh.mean(dim=1)
            patch_normals = patch_normals.mean(dim=1)
        elif agg == 'max':
            patch_fpfh,_ = patch_fpfh.max(dim=1)
            patch_normals,_ = patch_normals.max(dim=1)
        else:
            raise ValueError(f"Unknown agg: {agg}")

                               
        patch_feat = torch.cat([patch_fpfh, patch_normals, patch_curv], dim=-1)                            
        patch_feats_list.append(patch_feat)

    patch_feats = torch.stack(patch_feats_list, dim=0).to(device)                      
    return patch_feats


def compute_patch_scores(layer_feats, text_normal_embed, text_anomaly_embed):
    """
    Compute patch-level anomaly scores for one feature layer.

    Args:
        layer_feats: torch.Tensor, [B, P, C] single-layer features
        text_normal_embed: torch.Tensor, [1, C] normalized normal-text feature
        text_anomaly_embed: torch.Tensor, [1, C] normalized anomaly-text feature

    Returns:
        raw_scores: torch.Tensor, [B, P] raw anomaly scores
        soft_scores: torch.Tensor, [B, P] softmax anomaly probabilities
    """
    sim_normal = torch.matmul(layer_feats, text_normal_embed.T).squeeze(-1)          
    sim_anomaly = torch.matmul(layer_feats, text_anomaly_embed.T).squeeze(-1)          

    raw_scores = sim_anomaly - sim_normal

    logits = torch.stack([sim_normal, sim_anomaly], dim=-1)             
    probs = torch.softmax(logits, dim=-1)
    soft_scores = probs[..., 1]

    return raw_scores, soft_scores

def patch_scores_to_point_scores(patch_scores, patch_idx, N):
    """
    Map patch-level anomaly scores back to original points.

    Args:
        patch_scores: torch.Tensor, [B, 512] aggregated patch-level scores
        patch_idx: torch.LongTensor, [B, 512, 32] original point indices for each patch
        N: int, number of original points, e.g. 2048

    Returns:
        point_scores: torch.Tensor, [B, N] anomaly score for each point
    """
    B, P, K = patch_idx.shape
    device = patch_scores.device

                                   
    point_scores_sum = torch.zeros(B, N, device=device)
    point_counts = torch.zeros(B, N, device=device)

                               
    for b in range(B):
                              
                              
                                                             
        scores_expand = patch_scores[b].unsqueeze(-1).expand(-1, K)          
        idx = patch_idx[b]          

                            
        point_scores_sum[b].index_add_(0, idx.flatten(), scores_expand.flatten())
        point_counts[b].index_add_(0, idx.flatten(), torch.ones_like(scores_expand.flatten()))

                                                            
    point_scores = point_scores_sum / (point_counts + 1e-6)
    return point_scores

def compute_global_anomaly_score(pts_embed, text_normal_embed, text_anomaly_embed, temperature=0.07):
    """
    Compute global point-cloud anomaly scores from normalized features.

    Args:
        pts_embed: [B, C] normalized global point-cloud features
        text_normal_embed: [1, C] normalized normal-text embedding
        text_anomaly_embed: [1, C] normalized anomaly-text embedding
        temperature: temperature coefficient

    Returns:
        raw_scores: [B] raw anomaly scores (anomaly - normal)
        soft_scores: [B] sigmoid-normalized anomaly probabilities
    """
                                           
    sim_normal = torch.sum(pts_embed * text_normal_embed, dim=-1) / temperature       
    sim_anomaly = torch.sum(pts_embed * text_anomaly_embed, dim=-1) / temperature       

                         
    raw_scores = sim_anomaly - sim_normal                               

                                 
    soft_scores = torch.sigmoid(raw_scores)

    return raw_scores, soft_scores

