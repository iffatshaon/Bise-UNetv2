import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# Add CD-PolypNet-main/train to sys.path to resolve internal imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CD_PATH = os.path.join(PROJECT_ROOT, "CD-PolypNet-main", "train")
if CD_PATH not in sys.path:
    sys.path.append(CD_PATH)

try:
    from segment_anything_training import sam_model_registry
    from efb_net.efbanch import EFBranch
    from train.SSFD import SSFDLoss
    from canny import Net as CannyNet
except ImportError as e:
    print(f"Warning: Could not import CD-PolypNet components. Ensure CD-PolypNet-main is in the root. Error: {e}")

class CDPolypNet(nn.Module):
    """
    Wrapper for CD-PolypNet that supports training from scratch.
    It combines the SAM encoder and the HQ Decoder with Edge Feedback.
    """
    def __init__(self, model_type="vit_l", out_ch=1, use_pretrained=False, checkpoint_path=None):
        super().__init__()
        self.model_type = model_type
        
        # 1. Initialize SAM Model
        # If use_pretrained is False, we pass checkpoint=None to build it with random weights
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path if use_pretrained else None)
        
        # 2. Canny Edge Detector (Trainable or Fixed)
        self.canny = CannyNet(threshold=3.0, use_cuda=torch.cuda.is_available())
        
        # 3. HQ Decoder & EFBranch (Edge Feedback)
        # We manually initialize these to avoid the internal _load_model_weights call if training from scratch
        from train.train import MaskDecoderHQ
        
        # Subclass or modify MaskDecoderHQ to avoid loading weights in __init__
        class MaskDecoderHQScratch(MaskDecoderHQ):
            def _load_model_weights(self, model_type: str):
                if use_pretrained:
                    super()._load_model_weights(model_type)
                else:
                    print(f"Initializing {model_type} HQ Decoder from scratch (random weights).")

        self.mask_decoder = MaskDecoderHQScratch(model_type)
        
        # Final projection to match out_ch if needed (CD-PolypNet usually outputs 1 or 2 channels)
        self.out_ch = out_ch

    def forward(self, x):
        # x: [B, 3, H, W]
        bs, _, h, w = x.shape
        device = x.device
        
        # CD-PolypNet expects input in a specific 'batched_input' format for SAM
        # We must also handle the Canny edge extraction
        
        # A. Canny Edge Extraction
        # The authors' canny returns (blurred, grad_mag, orientation, thin, thresholded, early)
        _, _, _, _, edge, _ = self.canny(x) 
        
        # B. Prepare SAM Inputs
        # SAM ViT expects 1024x1024. We must resize or pad.
        # For simplicity in this wrapper, we assume x is already resized or we resize here.
        if h != 1024 or w != 1024:
            x_sam = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
            edge_sam = F.interpolate(edge, size=(1024, 1024), mode='bilinear', align_corners=False)
        else:
            x_sam = x
            edge_sam = edge
            
        # Transform for SAM: [0, 1] -> [0, 255] uint8 (internal to their logic usually)
        # But here we use the tensors directly where possible.
        
        batched_input = []
        for i in range(bs):
            batched_input.append({
                'image': (x_sam[i] * 255).byte(),
                'original_size': (h, w)
            })

        # C. SAM Encoder Pass
        # Returns batched_output, interm_embeddings, encoder_list, encoder_list_gscnn
        with torch.set_grad_enabled(self.training): # Allow gradients if training from scratch
            batched_output, interm_embeddings, encoder_list, encoder_list_gscnn = self.sam(batched_input, multimask_output=False)
        
        # D. HQ Decoder Pass
        # Extract necessary embeddings
        encoder_embedding = torch.cat([batched_output[i]['encoder_embedding'] for i in range(bs)], dim=0)
        image_pe = [batched_output[i]['image_pe'] for i in range(bs)]
        sparse_embeddings = [batched_output[i]['sparse_embeddings'] for i in range(bs)]
        dense_embeddings = [batched_output[i]['dense_embeddings'] for i in range(bs)]
        
        # Run MaskDecoderHQ
        # masks_sam, masks_hq, hq_features = net(...)
        _, masks_hq, _ = self.mask_decoder(
            edge=edge_sam,
            batched_input=batched_input,
            image_embeddings=encoder_embedding,
            image_pe=image_pe,
            encoder_list_gscnn=encoder_list_gscnn,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            hq_token_only=False,
            interm_embeddings=interm_embeddings
        )
        
        # E. Post-process
        # Resize back to original input size if needed
        if masks_hq.shape[-2:] != (h, w):
            logits = F.interpolate(masks_hq, size=(h, w), mode='bilinear', align_corners=False)
        else:
            logits = masks_hq
            
        return logits

def get_cd_polypnet(use_pretrained=False, **kwargs):
    model_type = kwargs.get("model_type", "vit_l")
    out_ch = kwargs.get("out_ch", 1)
    # If user has weights but wants to start "from scratch" logic-wise, they can pass them, 
    # but here we follow the request to NOT load pretrained.
    return CDPolypNet(model_type=model_type, out_ch=out_ch, use_pretrained=use_pretrained)
