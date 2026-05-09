from .unet_model import UNet
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .biseunetv2 import BiseUNetV2
from .ducknet_model import DuckNet
from .hardnet_model import HardNetMSEG
from .rfdetr_seg import RFDetrSeg
from .eomt_seg import EoMTSeg
from .cd_polypnet import get_cd_polypnet
from .pranet_model import PraNet

def get_model(name, **kwargs):
    name = name.lower()
    in_ch = kwargs.get('in_ch', 3)
    out_ch = kwargs.get('out_ch', 1)
    base_ch = kwargs.get('base_ch', 32)
    
    if name == 'unet':
        return UNet(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch)
    
    elif name == 'bisenet' or name == 'bisenetv1':
        return BiSeNetV1(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch)

    elif name == 'bisenetv2':
        return BiSeNetV2(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch)
        
    elif name == 'biseunet' or name == 'biseunetv2':
        return BiseUNetV2(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch, 
                          use_sp=kwargs.get('use_sp', True),
                          use_ffm_light=kwargs.get('use_ffm_light', True))
                          
    elif name == 'ducknet':
        return DuckNet(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch)
        
    elif name == 'hardnet':
        return HardNetMSEG(n_classes=out_ch)
        
    elif name == 'rfdetr':
        return RFDetrSeg(
            in_ch=in_ch,
            out_ch=out_ch,
            d_model=kwargs.get('d_model', 128),
            num_enc_layers=kwargs.get('num_enc_layers', 4),
        )

    elif name == 'eomt':
        return EoMTSeg(
            in_ch=in_ch,
            out_ch=out_ch,
            d_model=kwargs.get('d_model', 128),
            n_tokens=kwargs.get('n_tokens', 4),
        )

    elif name == 'cdpolypnet' or name == 'cd_polypnet':
        return get_cd_polypnet(use_pretrained=False, out_ch=out_ch)
        
    elif name == 'pranet':
        return PraNet(n_channels=in_ch, n_classes=out_ch)

    else:
        raise ValueError(f"Unknown model architecture: {name}")
