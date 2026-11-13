import torch
import torch.nn as nn
import timm
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
import random


class AttentionPooling(nn.Module):
    def __init__(self, embedding_size, hid=None):
        super().__init__()

        hid = embedding_size if hid is None else hid

        self.attn = nn.Sequential(
            nn.Linear(embedding_size, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Linear(hid, 1)
        )

    def forward(self, x, bs, mode='early', missing=None):
        # x: [B*T, D, H, W]
        # mask: [B, T]
        _, d, h, w = x.size()
        x = x.view(bs, -1, d, h, w)
        x = x.permute(0, 1, 3, 4, 2)  # [bs, t, h, w, d]

        # x:    [B, T, *, D]
        if mode == 'early':
            attn_logits = self.attn(x)  # [B, T, *, 1]
        else:
            attn_logits=torch.zeros((bs, x.size(1), x.size(2), x.size(3), 1), device=x.device)
            for b in range(bs):
                available_indices = torch.where(~missing[b])[0]
                chosen_month = random.choice(available_indices.tolist())
                missing[b, :] = True
                missing[b, chosen_month] = False
            
            attn_logits[missing]=-torch.inf
            attn_logits[~missing]=0

        attn_weights = attn_logits.softmax(dim=1)  # [B, T, *, 1]

        x = attn_weights * x  # [B, T, *, D]
        x = x.sum(dim=1)   # [B, *, D]

        x = x.permute(0, 3, 1, 2)     # [bs, d, h, w]

        return x, attn_weights


class TimmEncoder(nn.Module):
    def __init__(self, cfg, output_stride=32):
        super().__init__()

        depth = len(cfg.out_indices)
        if cfg.backbone.startswith("convnext"):
            cfg.backbone='convnextv2_base.fcmae_ft_in22k_in1k'
        elif cfg.backbone.startswith("mobilenet"):
            cfg.backbone='mobilenetv4_conv_medium.e500_r224_in1k'
        elif cfg.backbone=="efficientnet":
            cfg.backbone='tf_efficientnet_b5.ns_jft_in1k'
        elif cfg.backbone=="efficientnetv2":
            cfg.backbone='tf_efficientnetv2_m'
        
        
        self.model = timm.create_model(
            cfg.backbone,
            in_chans=cfg.in_channels,
            pretrained=cfg.pretrained,
            num_classes=0,
            features_only=True,
            output_stride=output_stride if output_stride != 32 else None,
            out_indices=cfg.out_indices,
        )
        self._in_channels = cfg.in_channels
        self._out_channels = [
            cfg.in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride  # 32

    def forward(self, x):
        features = self.model(x)

        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)

def late_fusion(x, bs, missing):
    x=x.view(bs, -1, x.size(1), x.size(2), x.size(3))  
    attn_logits=torch.zeros((bs, x.size(1), x.size(2), x.size(3), 1), device=x.device)

    attn_logits[missing]=-torch.inf
    attn_logits[~missing]=0

    attn_weights = attn_logits.softmax(dim=1)  # [B, T, *, 1]

    x = attn_weights * x  # [B, T, *, D]
    x = x.sum(dim=1)   # [B, *, D]

    return x

class UnetVFLOW(nn.Module):
    def __init__(self, args, decoder_use_batchnorm: bool = True, mode='early'):
        super().__init__()

        encoder_name = args.backbone

        self.encoder = TimmEncoder(args)

        encoder_depth = len(self.encoder.out_channels) - 1
        self.mode = mode

        self.attn = nn.ModuleList(
            [
                AttentionPooling(i)
                for i in self.encoder.out_channels[1:]
            ]
        )

        decoder_channels = args.dec_channels[:encoder_depth]
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            add_center_block=True if encoder_name.startswith("vgg") else False,
            attention_type=args.dec_attn_type,
        )
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], args.n_classes, kernel_size=3, padding=1)

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x, missing):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # Input x shape: torch.Size([4, 12, 15, 256, 256])
        # Input missing shape: torch.Size([4, 12])


        bs, _, d, h, w = x.size() 
        x = x.view(-1, d, h, w) #cada mes lo considera como frame independiente
        #Dimension de x torch.Size([48, 15, 256, 256])

        x = x.to(memory_format=torch.channels_last)
        features = self.encoder(x)
        #el encoder trata cada mes como frame independiente

        # Feature 0 before attention shape: torch.Size([48, 24, 128, 128])
        # Feature 1 before attention shape: torch.Size([48, 40, 64, 64])
        # Feature 2 before attention shape: torch.Size([48, 64, 32, 32])
        # Feature 3 before attention shape: torch.Size([48, 176, 16, 16])
        # Feature 4 before attention shape: torch.Size([48, 512, 8, 8])

        if self.training or self.mode=='early':
            features = [x] + [
                attn(f, bs, self.mode, missing)[0]
                for f, attn in zip(features, self.attn) #AQUI SE HACE LA FUSION
            ]#La x se incluye UNICAMENTE para saber reconstruir las dimensiones originales pero no se procesa,
            #por eso no importa que la dimension de batch sea 48 y no 4

            #Se obtiene el mapa de frames totales y los mapas de caracteristicas fusionados
            # Feature 0 shape: torch.Size([48, 15, 256, 256])
            # Feature 1 shape: torch.Size([4, 24, 128, 128])
            # Feature 2 shape: torch.Size([4, 40, 64, 64])
            # Feature 3 shape: torch.Size([4, 64, 32, 32])
            # Feature 4 shape: torch.Size([4, 176, 16, 16])
            # Feature 5 shape: torch.Size([4, 512, 8, 8])

        else:
            if self.mode=='late':
                features = [x] + features

        decoder_output = self.decoder(features)

        if self.mode=='late' and not self.training:
            decoder_output = late_fusion(decoder_output, bs, missing)
        
        masks = self.segmentation_head(decoder_output)

        return masks