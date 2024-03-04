import torch
import torch.nn as nn
from src.losses.loftr_loss import compute_normalized_6d, rotation_6d_to_matrix, pose_mean_6d, pose_std_6d
import numpy as np

# taken from NeRF https://github.com/bmild/nerf MIT license
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class SimpleTransformer(nn.Module):
    def __init__(self, config):
        super(SimpleTransformer, self).__init__()

        correspondence_dim = 0

        if config['correspondence_transformer_use_pos_encoding']:
            print("\n\n using positional encoding! \n\n")

            multires = config['num_bands']

            correspondence_dim = 42
            feat_size = config['feat_size']
            i_embed = 0
            self.positional_encoding, _ = get_embedder(multires, i_embed)
        else:
            correspondence_dim = 2

        self.feat_size = feat_size
        num_heads = config['num_heads']
        num_layers = config['num_encoder_layers']
        scale_factor = 2
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.feat_size, nhead=num_heads), num_layers=num_layers)

        if config['correspondence_tf_use_global_avg_pooling']:
            self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.H = self.feat_size * scale_factor // 2
        self.H2 = 512
        
        pose_size_in = 9
        pose_size = 9

        if config['regress']['use_simple_moe']:
            self.moe_encoder = nn.Sequential(
                nn.Linear(self.H, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
            )

            if config['regress']['use_1wt']:
                local_pose_size = 1
            elif config['regress']['use_2wt']:
                local_pose_size = 2
            else:
                local_pose_size = pose_size

            self.moe_predictor = nn.Sequential(
                nn.Linear(self.H+2*pose_size+1, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
                nn.ReLU(),
                nn.Linear(self.H2, local_pose_size),
                nn.Sigmoid(),
            )
            print(f'predicting {local_pose_size} weights')
            self.pose_regressor_simple_moe = nn.Sequential(
                nn.Linear(self.H2, self.H2),
                nn.ReLU(),
                nn.Linear(self.H2, pose_size),
            )
        else:
            self.decoder = nn.Linear(self.H, pose_size)
        self.config = config

        if config['correspondence_tf_use_feats']:
            if self.config['ctf_cat_feats']:
                fsize = self.feat_size // 4
            else:
                fsize = self.feat_size
            
            self.depth_feat_encoder = nn.Sequential(
                nn.Linear(256 * scale_factor, fsize),
                nn.ReLU(),
                nn.Linear(fsize, fsize),
            )

        if self.config['ctf_cat_feats']:
            fsize = self.feat_size * 3 // 4
            self.positional_encoding_projection_small = nn.Linear(correspondence_dim * scale_factor, fsize)
        else:
            fsize = self.feat_size
            self.positional_encoding_projection = nn.Linear(correspondence_dim * scale_factor, fsize)

    def forward_rt_prediction(self, data):
        B, N, I, D = data['correspondence_details'].shape

        if N == 0:
            if self.config['regress']['save_gating_weights']:
                data.update({'gating_reg_weights': torch.zeros([2])})
            return
    
        positions = data['correspondence_details'][...,:2].permute([0,2,3,1]).reshape([B*N, I*2])
        if self.config['correspondence_transformer_use_pos_encoding']:
            position_encodings = self.positional_encoding(positions)
        else:
            position_encodings = positions

        if self.config['ctf_cat_feats']:
            position_encodings_proj = self.positional_encoding_projection_small(position_encodings)[...,:self.feat_size].reshape([B, N, -1])
        else:
            position_encodings_proj = self.positional_encoding_projection(position_encodings)[...,:self.feat_size].reshape([B, N, -1])

        if self.config['correspondence_tf_use_feats']:
            if self.config['correspondence_tf_use_feats'] and \
                self.config['part2']:
                # part 2 - 
                depth_feats = data['out_features'].reshape([B*N, I*(D-2)])
            else:
                depth_feats = data['correspondence_details'][...,2:].permute([0,2,3,1]).reshape([B*N, I*(D-2)])
            depth_feat_encodings = self.depth_feat_encoder(depth_feats).reshape([B, N, -1])
            if self.config['ctf_cat_feats']:
                src = torch.cat([position_encodings_proj, depth_feat_encodings], dim=2)
            else:
                src = position_encodings_proj + depth_feat_encodings
        else:
            src = position_encodings_proj

        out_features = self.encoder(src.permute([1,0,2])).permute([1,0,2])

        if self.config['correspondence_tf_use_global_avg_pooling']:
            output_features = self.pooling(out_features.permute([0,2,1])).reshape([B,-1])
        else:
            output_features = out_features[:-1]
    
        if 'loftr_rt' in data:
            loftr_preds = data['loftr_rt'].detach()
            if len(loftr_preds.shape) > 2:
                loftr_preds = loftr_preds.squeeze(0)

            # map to normalized translation + 6d coords
            loftr_preds_6d = compute_normalized_6d(loftr_preds.float()).unsqueeze(0)
            if self.config['regress']['regress_use_num_corres']:
                num_correspondences = data['num_correspondences'].detach().float().unsqueeze(0) / 500 # make similar scale to 6d
                loftr_preds_6d = torch.cat([loftr_preds_6d, num_correspondences],dim=-1)

        if self.config['regress']['use_simple_moe']:
            feats = self.moe_encoder(output_features)
            pred_reg_6d = self.pose_regressor_simple_moe(feats)
            feats_preds = torch.cat([output_features, pred_reg_6d, loftr_preds_6d],dim=-1)
            pred_RT_wt = self.moe_predictor(feats_preds)

            if self.config['regress']['use_2wt']:
                t_size = 3
                
                if self.config['regress']['scale_8pt']:
                    loftr_pred_t = loftr_preds_6d[...,:t_size] * torch.linalg.norm(pred_reg_6d[...,:t_size]) / torch.linalg.norm(loftr_preds_6d[...,:t_size])
                else:
                    loftr_pred_t = loftr_preds_6d[...,:t_size]
                    
                pred_T = pred_RT_wt[...,0] * pred_reg_6d[...,:t_size] + (1-pred_RT_wt[...,0]) * loftr_pred_t
                pred_R = pred_RT_wt[...,1] * pred_reg_6d[...,t_size:] + (1-pred_RT_wt[...,1]) * loftr_preds_6d[...,t_size:-1]
                pred_RT = torch.cat([pred_T, pred_R], dim=-1)
                
            else:
                pred_RT = pred_RT_wt * pred_reg_6d + (1-pred_RT_wt) * loftr_preds_6d
        else:
            pred_RT = self.decoder(output_features)

        data.update({'regressed_rt': pred_RT,
                        'expec_rt': pred_RT[0]},)
    
        if self.config['regress']['save_gating_weights']:
            data.update({'gating_reg_weights': pred_RT_wt[0]})

        if self.config['solver'] == 'prior_ransac':
            R = data['regressed_rt'][:,3:].detach().cpu() * pose_std_6d[3:] + pose_mean_6d[3:]
            t = data['regressed_rt'][0,:3].detach().cpu().numpy()* pose_std_6d[:3].numpy() + pose_mean_6d[:3].numpy()
            R = rotation_6d_to_matrix(R)[0].numpy()
            data.update({'priorRT': np.concatenate([R,np.expand_dims(t,1)],axis=-1)})
            
        return 
    
    def forward(self, data):
        return self.forward_rt_prediction(data)
    