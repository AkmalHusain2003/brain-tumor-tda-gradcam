"""
model.py - MRI Classifier with GradCAM++ and Differentiable TDA
================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    from torch_topological.nn import VietorisRipsComplex
    TOPO_AVAILABLE = True
except ImportError:
    TOPO_AVAILABLE = False
    print("ERROR: torch_topological not found. Install: pip install torch-topological")


class GradCAMpp:
    """
    GradCAM++ implementation following:
    https://github.com/vickyliin/gradcam_plus_plus-pytorch
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(self, x, class_idx=0):
        batch_size, channels, height, width = x.size()
        
        # Enable gradients for GradCAM computation
        with torch.enable_grad():
            # Ensure input requires grad
            if not x.requires_grad:
                x = x.requires_grad_(True)
            
            # Forward pass
            logits = self.model(x)
            
            # Backward pass - per sample
            self.model.zero_grad()
            score = logits[:, class_idx]
            score.backward(gradient=torch.ones_like(score), retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        b, k, u, v = gradients.size()
        
        # Alpha computation (GradCAM++ formula from paper)
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2) + \
                      (activations * gradients.pow(3)).sum(dim=(2, 3), keepdim=True)
        
        alpha_denom = torch.where(
            alpha_denom != 0,
            alpha_denom,
            torch.ones_like(alpha_denom)
        )
        alpha = alpha_num / (alpha_denom + 1e-7)
        
        # Weights computation with ReLU on score-weighted gradients
        positive_gradients = F.relu(score.exp().view(b, 1, 1, 1) * gradients)
        weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)
        
        # Saliency map
        saliency_map = (weights * activations).sum(dim=1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        
        # Upsample to input resolution
        saliency_map = F.interpolate(
            saliency_map,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize per sample
        saliency_map_min = saliency_map.view(b, -1).min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        saliency_map_max = saliency_map.view(b, -1).max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min + 1e-8)
        
        return saliency_map


class CNNBackbone(nn.Module):
    """VGG19 backbone with frozen layers."""
    
    def __init__(self, freeze_layers=15):
        super().__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        for i, param in enumerate(vgg19.features.parameters()):
            param.requires_grad = (i >= freeze_layers)
        
        self.features = vgg19.features
        self.target_layer = self.features[-1]
        
        print(f"VGG19: {freeze_layers} layers frozen")
    
    def forward(self, x):
        return self.features(x)


class TopologyBranch(nn.Module):
    """Differentiable topology extraction via TDA."""
    
    def __init__(self, output_dim=128, n_points=400, img_resolution=64, sigma=1.0):
        super().__init__()
        
        if not TOPO_AVAILABLE:
            raise RuntimeError("torch_topological required")
        
        self.n_points = n_points
        self.img_resolution = img_resolution
        self.sigma = sigma
        self.vr_complex = VietorisRipsComplex(dim=1)
        
        pi_size = img_resolution * img_resolution
        self.compressor = nn.Sequential(
            nn.Linear(pi_size * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        
        print(f"Topology: {n_points} pts, {img_resolution}x{img_resolution} PI")
    
    def attention_to_pointcloud(self, attention_maps):
        """Convert attention maps to point clouds."""
        batch_size, _, height, width = attention_maps.shape
        point_clouds = []
        
        for b in range(batch_size):
            att = attention_maps[b, 0]
            flat = att.flatten()
            k = min(self.n_points, flat.numel())
            
            _, indices = torch.topk(flat, k)
            
            y = (indices // width).float() / (height - 1 + 1e-8)
            x = (indices % width).float() / (width - 1 + 1e-8)
            
            coords = torch.stack([y, x], dim=1)
            point_clouds.append(coords)
        
        return point_clouds
    
    def persistence_image(self, persistence_info, dim=0, device=None):
        """Convert persistence diagram to persistence image."""
        if device is None:
            device = torch.device('cpu')
            
        try:
            if hasattr(persistence_info, 'diagram'):
                diagrams = persistence_info.diagram
            elif isinstance(persistence_info, (list, tuple)):
                if dim >= len(persistence_info):
                    return torch.zeros(self.img_resolution, self.img_resolution, device=device)
                diagrams = persistence_info[dim]
            else:
                diagrams = persistence_info
            
            if isinstance(diagrams, (list, tuple)):
                if dim >= len(diagrams):
                    return torch.zeros(self.img_resolution, self.img_resolution, device=device)
                pairs = diagrams[dim]
            else:
                pairs = diagrams
            
            if not isinstance(pairs, torch.Tensor):
                if hasattr(pairs, 'diagram'):
                    pairs = pairs.diagram
                elif hasattr(pairs, 'pairing'):
                    pairs = pairs.pairing
                else:
                    return torch.zeros(self.img_resolution, self.img_resolution, device=device)
            
            # Ensure pairs is on the correct device
            if pairs.device != device:
                pairs = pairs.to(device)
            
            if pairs.numel() == 0 or pairs.shape[0] == 0:
                return torch.zeros(self.img_resolution, self.img_resolution, device=device)
            
            if pairs.dim() == 1:
                pairs = pairs.unsqueeze(0)
            
            if pairs.shape[1] != 2:
                return torch.zeros(self.img_resolution, self.img_resolution, device=device)
            
            births = pairs[:, 0]
            deaths = pairs[:, 1]
            persistences = deaths - births
            
            valid = persistences > 1e-6
            if not valid.any():
                return torch.zeros(self.img_resolution, self.img_resolution, device=device)
            
            births = births[valid]
            persistences = persistences[valid]
            weights = persistences
            
            birth_min = births.min()
            birth_max = births.max()
            pers_min = persistences.min()
            pers_max = persistences.max()
            
            if birth_max - birth_min < 1e-6:
                birth_min = torch.tensor(0.0, device=device)
                birth_max = torch.tensor(1.0, device=device)
            if pers_max - pers_min < 1e-6:
                pers_min = torch.tensor(0.0, device=device)
                pers_max = torch.tensor(1.0, device=device)
            
            births_norm = (births - birth_min) / (birth_max - birth_min + 1e-8)
            pers_norm = (persistences - pers_min) / (pers_max - pers_min + 1e-8)
            
            b_grid = torch.linspace(0.0, 1.0, self.img_resolution, device=device)
            p_grid = torch.linspace(0.0, 1.0, self.img_resolution, device=device)
            
            img = torch.zeros(self.img_resolution, self.img_resolution, device=device)
            
            for i in range(births_norm.shape[0]):
                b = births_norm[i]
                p = pers_norm[i]
                w = weights[i]
                b_diff = (b_grid - b).pow(2)
                p_diff = (p_grid - p).pow(2)
                kernel = torch.exp(-(b_diff.unsqueeze(1) + p_diff.unsqueeze(0)) / (2 * self.sigma ** 2))
                img += w * kernel
            
            return img
            
        except Exception:
            return torch.zeros(self.img_resolution, self.img_resolution, device=device)
    
    def forward(self, attention_maps):
        """Extract topological features from attention maps."""
        device = attention_maps.device
        point_clouds = self.attention_to_pointcloud(attention_maps)
        
        pi_h0_list = []
        pi_h1_list = []
        
        for pc in point_clouds:
            pc_batch = pc.unsqueeze(0)
            persistence_list = self.vr_complex(pc_batch)
            persistence_info = persistence_list[0] if isinstance(persistence_list, list) else persistence_list
            
            pi_h0 = self.persistence_image(persistence_info, dim=0, device=device)
            pi_h1 = self.persistence_image(persistence_info, dim=1, device=device)
            
            pi_h0_list.append(pi_h0.flatten())
            pi_h1_list.append(pi_h1.flatten())
        
        pi_h0_batch = torch.stack(pi_h0_list, dim=0)
        pi_h1_batch = torch.stack(pi_h1_list, dim=0)
        
        topo_features = torch.cat([pi_h0_batch, pi_h1_batch], dim=1)
        output = self.compressor(topo_features)
        
        return output


class CNNWrapper(nn.Module):
    """Wrapper for GradCAM++ computation."""
    
    def __init__(self, parent_model):
        super().__init__()
        self.parent = parent_model
    
    def forward(self, x):
        features = self.parent.backbone(x)
        features = features.reshape(x.size(0), -1)
        x = F.relu(self.parent.cnn_bn1(self.parent.cnn_fc1(features)))
        x = F.relu(self.parent.cnn_fc2(x))
        logits = self.parent.cnn_output(x)
        return logits


class MRIClassifier(nn.Module):
    """Two-stage MRI classifier with GradCAM++ and TDA."""
    
    def __init__(self, freeze_layers=15):
        super().__init__()
        self.backbone = CNNBackbone(freeze_layers=freeze_layers)
        
        self.cnn_fc1 = nn.Linear(512 * 7 * 7, 512)
        self.cnn_bn1 = nn.BatchNorm1d(512)
        self.cnn_fc2 = nn.Linear(512, 128)
        self.cnn_output = nn.Linear(128, 1)
        
        self.topology_branch = TopologyBranch(output_dim=128)
        
        self.fusion_fc1 = nn.Linear(256, 256)
        self.fusion_bn1 = nn.BatchNorm1d(256)
        self.fusion_fc2 = nn.Linear(256, 128)
        self.fusion_output = nn.Linear(128, 1)
        
        self.gradcam = None
        
        print("MRIClassifier: GradCAM++ + TDA initialized")
    
    def _init_gradcam(self):
        """Initialize GradCAM++ wrapper on first use."""
        if self.gradcam is None:
            wrapper = CNNWrapper(self)
            self.gradcam = GradCAMpp(wrapper, self.backbone.target_layer)
    
    def forward_cnn_only(self, images):
        """Stage 1: CNN-only forward pass."""
        features = self.backbone(images)
        features = features.reshape(images.size(0), -1)
        
        x = F.relu(self.cnn_bn1(self.cnn_fc1(features)))
        x = F.relu(self.cnn_fc2(x))
        logits = self.cnn_output(x)
        
        return torch.sigmoid(logits)
    
    def forward_fusion(self, images):
        """Stage 2: CNN + GradCAM++ + TDA fusion."""
        self._init_gradcam()
        
        # Compute GradCAM++ with detached input to avoid backprop to original input
        images_detached = images.detach()
        attention_maps = self.gradcam(images_detached)
        
        # Detach attention to prevent backprop through GradCAM during main optimization
        attention_maps = attention_maps.detach()
        
        # Continue with main forward pass
        cnn_features = self.backbone(images)
        cnn_features_flat = cnn_features.reshape(images.size(0), -1)
        cnn_x = F.relu(self.cnn_bn1(self.cnn_fc1(cnn_features_flat)))
        cnn_x = F.relu(self.cnn_fc2(cnn_x))
        
        topo_x = self.topology_branch(attention_maps)
        
        combined = torch.cat([cnn_x, topo_x], dim=1)
        x = F.relu(self.fusion_bn1(self.fusion_fc1(combined)))
        x = F.relu(self.fusion_fc2(x))
        logits = self.fusion_output(x)
        
        return torch.sigmoid(logits)
    
    def forward(self, images, stage=1):
        """Forward pass with stage selection."""
        if stage == 1:
            return self.forward_cnn_only(images)
        else:
            return self.forward_fusion(images)
    
    def extract_attention_numpy(self, images, device):
        """Extract attention maps for visualization."""
        self.eval()
        images_tensor = torch.from_numpy(images).float().to(device)
        
        self._init_gradcam()
        
        attention_maps = self.gradcam(images_tensor)
        
        return attention_maps.squeeze(1).detach().cpu().numpy()