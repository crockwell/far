import numpy as np
from collections import OrderedDict

import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_readers.factory import dataset_factory
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F

#import lietorch
#from lietorch import SE3

# network
from src.model import ViTEss
from src.logger import Logger

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import random
from datetime import datetime
import os

def setup_ddp(gpu, args):
    dist.init_process_group(                                   
    	backend='nccl',                                 
   		init_method='env://',     
    	world_size=args.world_size,                              
    	rank=gpu)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v

# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out

def compute_pose_from_rotation_matrix(T_pose, r_matrix):
    batch=r_matrix.shape[0]
    joint_num = T_pose.shape[0]

    r_matrix = torch.from_numpy(r_matrix).cuda().float()

    r_matrices = r_matrix.view(batch,1, 3,3).expand(batch,joint_num, 3,3).contiguous().view(batch*joint_num,3,3)
    src_poses = T_pose.view(1,joint_num,3,1).expand(batch,joint_num,3,1).contiguous().view(batch*joint_num,3,1)
        
    out_poses = torch.matmul(r_matrices, src_poses) #(batch*joint_num)*3*1
        
    return out_poses.view(batch, joint_num,3)

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

def train(gpu, args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    if not args.no_ddp:
        setup_ddp(gpu, args)
    rng = np.random.default_rng(12345)
    random.seed(0)

    thiscuda = 'cuda:%d' % gpu
    map_location = {'cuda:%d' % 0: thiscuda}
    args.map_location = map_location
    if args.no_ddp: 
        args.map_location = ''
        thiscuda = 'cuda:0'

    T_pose_np  = np.array([[1,0,0],[0,1,0], [0,0,1]])
    T_pose = torch.FloatTensor(T_pose_np).cuda()
    args.T_pose = T_pose

    # for normalizing 6D depending on dataset
    if args.dataset == 'matterport':
        global_pose_mean = torch.tensor([-0.06979753, 0.03417105, -0.17588863, 0.50275223, 0.03533648, -0.18179045, -0.03533648, 0.98189617, 0.09313615]).cuda()
        global_pose_std = torch.tensor([0.38802881, 0.07354026, 0.37663504, 0.51837117, 0.12717603, 0.65426397, 0.12717603, 0.0188729, 0.09709263]).cuda()
    elif args.dataset == 'interiornet':
        if args.streetlearn_interiornet_type == 'T':
            global_pose_mean = torch.tensor([0,0,0,0.92456496, -0.00201821, -0.00987212, -0.00019313, 0.72139406, -0.00184757]).cuda()
            global_pose_std = torch.tensor([1,1,1,0.07689704, 0.17564303, 0.32912105, 0.1753406, 0.27482772, 0.6109926]).cuda()
        else:
            global_pose_mean = torch.tensor([0,0,0,0.9275364, -0.00368287, -0.00655767, 0.00045095, 0.7385428, -0.00683342]).cuda()
            global_pose_std = torch.tensor([1,1,1,0.07534314, 0.1704135, 0.32389316, 0.17006727, 0.27120626, 0.5933235]).cuda()
    elif args.dataset == 'streetlearn':
        if args.streetlearn_interiornet_type == 'T':
            global_pose_mean = torch.tensor([0,0,0,0.828742, 0.00034936, -0.00100069, -0.00250733,  0.7001684, -0.00283758]).cuda()
            global_pose_std = torch.tensor([1,1,1,0.16392577, 0.2663457, 0.46407992, 0.26599622, 0.27905113, 0.60093635]).cuda()
        else:
            global_pose_mean = torch.tensor([0,0,0,0.8217494, -0.0019066, -0.00003673, -0.00000574,  0.697334, -0.00272899]).cuda()
            global_pose_std = torch.tensor([1,1,1,0.16815728, 0.27100316, 0.47223347, 0.27088866, 0.2769559, 0.60302496]).cuda()
    else:
        assert(False)

    model = ViTEss(args, global_pose_mean=global_pose_mean, global_pose_std=global_pose_std)

    model.to(thiscuda)
    model.train()

    # unused layers
    for param in model.resnet.layer4.parameters():
        param.requires_grad = False

    for param in model.resnet.layer3.parameters():
        param.requires_grad = False

    if not args.no_ddp:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    pct_warmup = args.warmup / args.steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=pct_warmup, div_factor=25, cycle_momentum=False)

    if args.ckpt is not None:
        print('loading separate checkpoint', args.ckpt)

        if args.no_ddp:
            existing_ckpt = torch.load(args.ckpt)
        else:
            existing_ckpt = torch.load(args.ckpt, map_location=map_location)

        model.load_state_dict(existing_ckpt['model'], strict=False)
        # optimizer.load_state_dict(existing_ckpt['optimizer'])

        del existing_ckpt
    elif args.existing_ckpt is not None:
        if args.no_ddp:
            existing_ckpt = torch.load(args.existing_ckpt)
            state_dict = OrderedDict([
                (k.replace("module.", ""), v) for (k, v) in existing_ckpt['model'].items()])
            model.load_state_dict(state_dict)
            del state_dict
            optimizer.load_state_dict(existing_ckpt['optimizer'])
            if 'scheduler' in existing_ckpt:
                scheduler.load_state_dict(existing_ckpt['scheduler'])
        else:
            existing_ckpt = torch.load(args.existing_ckpt, map_location=map_location)
            model.load_state_dict(existing_ckpt['model'])
            optimizer.load_state_dict(existing_ckpt['optimizer'])
            if 'scheduler' in existing_ckpt:
                scheduler.load_state_dict(existing_ckpt['scheduler'])
        print('loading existing checkpoint')
        del existing_ckpt

    logger = Logger(args.name, scheduler)
    should_keep_training = True

    subepoch = 0

    train_steps = 0
    epoch_count = 0
    while should_keep_training:
        is_training = True
        train_val = 'train'
        if subepoch == 10:
            """
            validate!
            """
            is_training = False
            train_val = 'val'
        
        db = dataset_factory([args.dataset], datapath=args.datapath, from_saved_preds=args.from_saved_preds, \
                subepoch=subepoch, use_loftr_gating=args.use_loftr_gating, \
                is_training=is_training, gpu=gpu, reshape_size=args.image_size, \
                streetlearn_interiornet_type=args.streetlearn_interiornet_type, use_mini_dataset=args.use_mini_dataset,
                solver=args.solver)
        if not args.no_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                db, shuffle=is_training, num_replicas=args.world_size, rank=gpu)
            train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(db, batch_size=args.batch, num_workers=0,shuffle=False)
        
        model.train()

        if not is_training:
            model.eval()

        with tqdm(train_loader, unit="batch") as tepoch:
            for i_batch, item in enumerate(tepoch):
                optimizer.zero_grad()

                images, poses, intrinsics, loftr_num_corr, loftr_preds = [x.to('cuda') for x in item]
                #Ps = SE3(poses)
                #Gs = SE3.IdentityLike(Ps)
                #Ps_out = SE3(Ps.data.clone())

                poses = poses[:,1]

                # turn pose[1] from quat to 6D
                rot_mtx = R.as_matrix(R.from_quat(poses.cpu().numpy()[:,3:]))
                gt_rot = compute_pose_from_rotation_matrix(T_pose, rot_mtx)

                metrics = {}

                torch.set_printoptions(precision=3, sci_mode=False)

                if not is_training:
                    with torch.no_grad():
                        tran_preds, rot_preds, rot_preds_mtx, rot_preds_6d = model(images, intrinsics=intrinsics, \
                                                                    loftr_num_corr=loftr_num_corr, \
                                                                    loftr_preds=loftr_preds)
                        #geo_loss_tr, geo_loss_rot, geo_metrics = geodesic_loss(Ps_out, poses_est, train_val=train_val)
                        # geo_loss_tr, geo_loss_rot, geo_metrics = geodesic_loss_6d(poses, poses_est, train_val=train_val)
                        if "losson6d" in args and args.losson6d:
                            # uses MDM loss. preds unchanged, GT mapped from rot mtx to 6D via MDM code
                            gt_rot_6d = matrix_to_rotation_6d(torch.from_numpy(rot_mtx).cuda())
                            
                            if args.use_normalized_6d:
                                # normalize to N(0,1) based on dataset
                                gt_rot_6d = (gt_rot_6d - global_pose_mean[3:]) / global_pose_std[3:]

                            geo_loss_rot = torch.pow(rot_preds_6d-gt_rot_6d,2).mean() # l2 on 6D
                        else:
                            # uses zhou et al loss. preds maps to rot mtx, then pose; GT mapped to pose. All via the zhou et al code
                            geo_loss_rot = torch.pow(rot_preds-gt_rot,2).mean() # l2 on pose, where pose is from rot mtx, which is projected from their 6D format
                            
                        # normalize to N(0,1) based on dataset
                        if args.use_normalized_6d:
                            gt_tran = (poses[:,:3] - global_pose_mean[:3]) / global_pose_std[:3]

                        geo_loss_tr = torch.pow(tran_preds-gt_tran,2).mean()
                        geo_metrics = {
                            train_val+'_geo_loss_tr': (geo_loss_tr).detach().item(),
                            train_val+'_geo_loss_rot': (geo_loss_rot).detach().item(),
                        }
                else:
                    tran_preds, rot_preds, rot_preds_mtx, rot_preds_6d = model(images, intrinsics=intrinsics, \
                                                                    loftr_num_corr=loftr_num_corr, \
                                                                    loftr_preds=loftr_preds)
                    #geo_loss_tr, geo_loss_rot, geo_metrics = geodesic_loss(Ps_out, poses_est, train_val=train_val)
                    # geo_loss_tr, geo_loss_rot, geo_metrics = geodesic_loss_6d(poses, poses_est, train_val=train_val)
                    if "losson6d" in args and args.losson6d:
                        # uses MDM loss. preds unchanged, GT mapped from rot mtx to 6D via MDM code
                        # import pdb; pdb.set_trace()
                        gt_rot_6d = matrix_to_rotation_6d(torch.from_numpy(rot_mtx).cuda())

                        # normalize to N(0,1) based on dataset
                        if args.use_normalized_6d:
                            gt_rot_6d = (gt_rot_6d - global_pose_mean[3:]) / global_pose_std[3:]

                        #torch.set_printoptions(sci_mode=False)
                        #torch.set_printoptions(precision=3)
                        #print(gt_rot_6d)
                        #print(gt_rot)
                        #print(compute_rotation_matrix_from_ortho6d(gt_rot_6d))
                        #print(rotation_6d_to_matrix(gt_rot_6d))
                        geo_loss_rot = torch.pow(rot_preds_6d-gt_rot_6d,2).mean() # l2 on 6D
                        
                        #import pdb; pdb.set_trace()
                        #print("loss", geo_loss_rot)
                        #print("preds", rot_preds_6d)
                        #print("gt", gt_rot_6d)
                        #import pdb; pdb.set_trace()
                    else:
                        # uses zhou et al loss. preds maps to rot mtx, then pose; GT mapped to pose. All via the zhou et al code
                        geo_loss_rot = torch.pow(rot_preds-gt_rot,2).mean() # l2 on pose, where pose is from rot mtx, which is projected from their 6D format
                    
                    # normalize to N(0,1) based on dataset
                    if args.use_normalized_6d:
                        gt_tran = (poses[:,:3] - global_pose_mean[:3]) / global_pose_std[:3]

                    geo_loss_tr = torch.pow(tran_preds-gt_tran,2).mean()
                    geo_metrics = {
                        train_val+'_geo_loss_tr': (geo_loss_tr).detach().item(),
                        train_val+'_geo_loss_rot': (geo_loss_rot).detach().item(),
                    }

                    loss = args.w_tr * geo_loss_tr + args.w_rot * geo_loss_rot

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    
                    scheduler.step() 
                    train_steps += 1
                
                metrics.update(geo_metrics)                                    

                if gpu == 0 or args.no_ddp:
                    logger.push(metrics)

                    if i_batch % 20 == 0:
                        poses_est = torch.cat([tran_preds.cpu(), torch.from_numpy(R.as_quat(R.from_matrix(rot_preds_mtx.cpu().detach().numpy())))], dim=1)
                        torch.set_printoptions(sci_mode=False, linewidth=150)
                        #for local_index in range(len(poses_est)):
                        local_index = 0
                        print('pred number:', local_index)
                        print('\n estimated pose')
                        print(poses_est[local_index].cpu().detach())
                        print('ground truth pose')
                        torch.set_printoptions(sci_mode=False)
                        torch.set_printoptions(precision=3)
                        print(poses.data[local_index].cpu().detach().double())
                        print('')
                    if (i_batch + 10) % 20 == 0:
                        print('\n metrics:', metrics, '\n')
                    if i_batch % 100 == 0:
                        print('epoch', str(epoch_count))
                        print('subepoch: ', str(subepoch))
                        print('using', train_val, 'set')

                if train_steps % args.ckpt_steps == 0 and (gpu == 0 or args.no_ddp) and is_training:
                    PATH = 'output/%s/checkpoints/%06d.pth' % (args.name, train_steps)
                    checkpoint = {"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()}
                    torch.save(checkpoint, PATH)

                if train_steps >= args.steps:
                    PATH = 'output/%s/checkpoints/%06d.pth' % (args.name, train_steps)
                    checkpoint = {"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()}
                    torch.save(checkpoint, PATH)
                    should_keep_training = False
                    break
       
        subepoch = (subepoch + 1)
        if subepoch == 11:# or (subepoch == 10 and (args.dataset == "interiornet" or args.dataset == "streetlearn")):
            # now, use the final 9% as val set.
            subepoch = 0
            epoch_count += 1

    print("finished training!")
    dist.destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('--w_tr', type=float, default=10.0)
    parser.add_argument('--w_rot', type=float, default=10.0)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--steps', type=int, default=120000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_ddp', action="store_true", default=False)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--name', default='bla', help='name your experiment')

    # data
    parser.add_argument("--datapath")
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--exp")
    parser.add_argument('--use_mini_dataset', action='store_true')
    parser.add_argument('--streetlearn_interiornet_type', default='', choices=('',"T"))
    parser.add_argument('--dataset', default='matterport', choices=("matterport", "interiornet", 'streetlearn'))

    # model
    parser.add_argument('--fusion_transformer', action="store_true", default=False)
    parser.add_argument('--fc_hidden_size', type=int, default=512)
    parser.add_argument('--pool_size', type=int, default=60)
    parser.add_argument('--transformer_depth', type=int, default=6)

    parser.add_argument('--losson6d', action='store_true')
    parser.add_argument('--use_normalized_6d', action='store_true')

    # loftr
    parser.add_argument('--use_loftr_gating', action='store_true', default=False)
    parser.add_argument("--from_saved_preds")
    parser.add_argument("--solver", type=str, default="ransac")
    parser.add_argument('--ckpt_steps', type=int, default=10000)

    args = parser.parse_args()
    
    print(args)

    PATHS = ['output/%s/checkpoints' % (args.name), 'output/%s/runs' % (args.name), 'output/%s/train_output/images' % (args.name)]
    args.existing_ckpt = None

    for PATH in PATHS:
        try:
            os.makedirs(PATH)
        except:
            if 'checkpoints' in PATH:
                ckpts = os.listdir(PATH)

                if len(ckpts) > 0:
                    if 'most_recent_ckpt.pth' in ckpts:
                        existing_ckpt = 'most_recent_ckpt.pth'
                    else:
                        ckpts = [int(i[:-4]) for i in ckpts]
                        ckpts.sort()
                        existing_ckpt = str(ckpts[-1]).zfill(6) +'.pth'
                
                    args.existing_ckpt = os.path.join(PATH, existing_ckpt)
                    print('existing',args.existing_ckpt)
            pass

    
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M")

    with open('output/%s/args_%s.txt' % (args.name, dt_string), 'w') as f:
        for k, v in vars(args).items():
            f.write(str(k) + '  '+ str(v) + '\n')
        
    if args.no_ddp:
        train(args.gpus, args)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        args.world_size = args.gpus
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    
