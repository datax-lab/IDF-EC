import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    

class FeatureProjector(nn.Module):
    def __init__(self, input_dim, target_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
    def forward(self, x):
        return self.net(x)


class IDF_EC(nn.Module):
    def __init__(self, output_classes, cuda_available, device, num_classes, ecpick_feature, hitec_feature, clean_feature, proj_dim: int=256):
        super().__init__()
        self.output_classes = output_classes
        dim_ecpick = 384
        dim_hitec = 1024
        dim_clean = 256
        self.proj_dim = proj_dim
        
        self.proj_ecpick = FeatureProjector(dim_ecpick, self.proj_dim)
        self.proj_hitec = FeatureProjector(dim_hitec, self.proj_dim)
        self.proj_clean = FeatureProjector(dim_clean, self.proj_dim)

        self.index_emb = nn.Embedding(num_classes, 16)
        
        combined_input_dim = (10 * 3) + (160 * 3)
        self.ln_topk = nn.LayerNorm(10)

        self.unified_gate = nn.Sequential(
            nn.Linear(combined_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3) 
        )
        
        self.stacking_classifier = nn.Sequential(
            nn.Linear(proj_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        
        self.monitor_stats = {}
        nn.init.constant_(self.unified_gate[-1].bias[2], 1.0)
        nn.init.constant_(self.unified_gate[-1].bias[1], 0.5)
        nn.init.constant_(self.unified_gate[-1].bias[0], -0.5)
    
    def safe_inverse_sigmoid(self, p):
        p = torch.clamp(p, 0.0001, 0.9999)
        return torch.log(p / (1.0 - p))

    def _compute_stats(self, tensor, name):
        with torch.no_grad():
            t = tensor
            if not torch.is_floating_point(t):
                t = t.float()
            return {
                f"{name}_mean": t.mean().item(),
                f"{name}_std":  t.std().item(),
                f"{name}_min":  t.min().item(),
                f"{name}_max":  t.max().item()
            }
    

    def forward(self, ecpick_features, ecpick_outputs, hitec_features, hitec_outputs, clean_features, clean_outputs, gate=False):
        self.monitor_stats = {}
        fe = ecpick_features
        fh = hitec_features[3]
        fc = clean_features
        
        res_ec = self.safe_inverse_sigmoid(ecpick_outputs)
        res_hi = self.safe_inverse_sigmoid(hitec_outputs)
        res_cl = self.safe_inverse_sigmoid(clean_outputs)
        
        topk_ec, topk_idx_ec = torch.topk(res_ec, 10, dim=1)
        topk_hi, topk_idx_hi = torch.topk(res_hi, 10, dim=1)
        topk_cl, topk_idx_cl = torch.topk(res_cl, 10, dim=1)
        topk_ec_norm = self.ln_topk(topk_ec)
        topk_hi_norm = self.ln_topk(topk_hi)
        topk_cl_norm = self.ln_topk(topk_cl)
        
        f_ec = self.proj_ecpick(fe)
        f_hi = self.proj_hitec(fh)
        f_cl = self.proj_clean(fc)
        
        f_combined = torch.cat([f_ec, f_hi, f_cl], dim=1)
        mlp_logits = self.stacking_classifier(f_combined)
        final_logits = mlp_logits
        
        if gate == False:
            final_logits = mlp_logits
        else:
            emb_ec = self.index_emb(topk_idx_ec).view(topk_idx_ec.size(0), -1)
            emb_hi = self.index_emb(topk_idx_hi).view(topk_idx_hi.size(0), -1)
            emb_cl = self.index_emb(topk_idx_cl).view(topk_idx_cl.size(0), -1)
            
            gate_input = torch.cat([
                topk_ec_norm, topk_hi_norm, topk_cl_norm,
                emb_ec, emb_hi, emb_cl
            ], dim=1)
            
            gate_logits = self.unified_gate(gate_input)
            gate_weights = torch.softmax(gate_logits / 1.0, dim=1)
            
            w_ec = gate_weights[:, 0:1]
            w_hi = gate_weights[:, 1:2]
            w_cl = gate_weights[:, 2:3]
        
        
            # alpha = 0.15
            alpha = 1
            weighted_res = (w_ec * res_ec + w_hi * res_hi + w_cl * res_cl)
            final_logits = mlp_logits + alpha * weighted_res
            
            self.monitor_stats.update(self._compute_stats(w_ec.detach(), "w_ec"))
            self.monitor_stats.update(self._compute_stats(w_hi.detach(), "w_hi"))
            self.monitor_stats.update(self._compute_stats(w_cl.detach(), "w_cl"))
           
        # Stats Monitoring
        self.monitor_stats.update(self._compute_stats(final_logits.detach(), "final_logits"))
        
        
        return final_logits, self.monitor_stats, w_ec, w_hi, w_cl