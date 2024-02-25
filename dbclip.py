from typing import Any, Mapping
import yaml, torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
import open_clip_m as open_clip
import time
from torch.cuda.amp import autocast
from Visual_Prompt import visual_prompt
from dover import DOVER
import open_clip_m
from efficientvideorecognition import model as eff_model


Device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model = open_clip.load_openai_model('ViT-B-32')



def read_video(video_path: str, yml_path: str = r"maxvqa.yml", data_set = "universal"):
    with open(yml_path, "r") as f:
        opt = yaml.safe_load(f)

        data_option = opt["data"][data_set]["args"]

        temporal_samplers = {}
        for sample_type, sample_option in data_option["sample_types"].items():
            if "t_frag" not in sample_option:
                # resized temporal sampling for TQE in DOVER
                temporal_samplers[sample_type] = UnifiedFrameSampler(
                    sample_option["clip_len"], sample_option["num_clips"], sample_option["frame_interval"]
                )
            else:
                # temporal sampling for AQE in DOVER
                temporal_samplers[sample_type] = UnifiedFrameSampler(
                    sample_option["clip_len"] // sample_option["t_frag"],
                    sample_option["t_frag"],
                    sample_option["frame_interval"],
                    sample_option["num_clips"],
                )
    mean = torch.FloatTensor([123.675, 116.28, 103.53]).reshape(-1,1,1,1)
    std = torch.FloatTensor([58.395, 57.12, 57.375]).reshape(-1,1,1,1)
    video_data, _ = spatial_temporal_view_decomposition(
        video_path, data_option["sample_types"], temporal_samplers,
    )
    video_data = {"aesthetic": (video_data["aesthetic"] - mean) / std,
                  "technical": (video_data["technical"] - mean) / std}
    return video_data

def encode_text_prompts(tokenizer, prompts,device="cuda"):
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(text_tokens)
            text_features = clip_model.encode_text(text_tokens).float()
        return text_tokens, embedding, text_features


def trunc_normal_(x, mean=0., std=1.):
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.transformer.get_cast_dtype()
        self.attn_mask = None

    def forward(self, prompts, tokenized_prompts):
        # assert not torch.any(torch.isnan(self.text_projection))
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2).to(self.dtype)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.out_conv = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            module.weight.data.normal_(std=0.5)

    def forward(self, x:torch.tensor):
        shape = x.shape
        if x.ndim > 3:
            x = x.reshape(-1, *shape[-2:])
        x = x.transpose(-1, -2)  # Reshape input to (batch_size, in_channels, sequence_length)
        x = self.in_conv(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.out_conv(x)
        return x.transpose(-1, -2).reshape(*shape[:-1], self.out_conv.out_channels)  # Reshape output back to (batch_size, sequence_length, out_channels)


class encodeMultiVideo(nn.Module):
    def __init__(self, video_len, num_clips=1):
        super().__init__()
        self.num_clips = num_clips
        T = video_len // num_clips
        self.scale = clip_model.logit_scale

        self.visual_projection = torch.load('pretrain/visual.proj.pt').cuda()
        
        self.temp_fusion = visual_prompt('Transf', clip_model.state_dict(), T)
        self.temp_fusion_clips = visual_prompt('Transf', clip_model.state_dict(),
                                               num_clips) if num_clips != 1 else nn.Identity()

    def forward(self, video):
        b, t, c, h, w = video.shape
        assert t % self.num_clips == 0
        images = video.reshape(-1, c, h, w)
        with torch.no_grad():
            _, positional_features = clip_model.encode_image(images)
            image_features = positional_features.mean(1, keepdim=False)
            _, width = image_features.shape
            image_features = image_features.reshape(b, t, width)      # b, t, 768
            image_features = image_features @ self.visual_projection  # b, t, 512
        image_feature = self.temp_fusion(image_features)
        image_feature = self.temp_fusion_clips(image_feature.reshape(b, self.num_clips, -1))
        image_feature = nn.functional.layer_norm(image_feature, [512])
        return image_feature.reshape(b, -1)

text_encoder = TextEncoder()

class myModel(nn.Module):
    def __init__(self, text_tokens, embedding, n_ctx=1, share_ctx=False, n_frames=64):
        super().__init__()
        self.context_length, width = 77, 768
        self.share_ctx = share_ctx
        video_lengths = (64, 128)
        self.tokenized_prompts = text_tokens
        self.scale = clip_model.logit_scale
        self.maxw = False
        self.video_encoder_aesthetic = encodeMultiVideo(video_lengths[0], num_clips=1)
        # self.video_encoder_technical = encodeMultiVideo(video_lengths[1], num_clips=4)
        self.video_encoder_technical = eff_model.EVLTransformer(num_frames=n_frames)
        self.mlp = MLP(1024, 64, 512)
        self.dropout = nn.Dropout()
        # self.bn = nn.BatchNorm1d(512)
        self.ln = nn.LayerNorm([512])
        if n_ctx > 0:
            if not share_ctx:
                self.ctx = nn.Parameter(embedding[:, 1:1+n_ctx].clone())
            else:
                self.ctx = nn.Parameter(embedding[0:1, 1:1+n_ctx].clone())
        else:
            self.register_buffer("ctx", embedding[:, 1:1, :])
            print("Disabled Context Prompt")
        self.register_buffer("prefix", embedding[:, :1, :].clone())  # SOS
        self.register_buffer("suffix", embedding[:, 1 + n_ctx:, :].clone())# CLS, EOS
        
        self.prefix.requires_grad = False
        self.suffix.requires_grad = False
        
        n_prompts = self.get_text_prompts()
        self.text_feats = text_encoder(n_prompts.cuda(), self.tokenized_prompts)

    def initialize_inference(self, text_encoder):
        n_prompts = self.get_text_prompts()
        text_feats = text_encoder(n_prompts, self.tokenized_prompts)
        self.text_feats = text_feats 


    def get_text_prompts(self):
        if self.share_ctx:
            return torch.cat(
                [
                    self.prefix,  # (n_cls, 1, dim)
                    self.ctx.repeat(self.prefix.shape[0],1,1),  # (n_cls, n_ctx, dim)
                    self.suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            ).half()
        return torch.cat(
                [
                    self.prefix,  # (n_cls, 1, dim)
                    self.ctx,     # (n_cls, n_ctx, dim)
                    self.suffix,  # (n_cls, *, dim)
                ],
                dim=1,
        ).half()



    def forward(self, aesthetic_video, technical_video):
        n_prompts = self.get_text_prompts()
        if self.training:
            text_feats = text_encoder(n_prompts, self.tokenized_prompts)
            self.text_feats = text_feats 
        else:
            text_feats = self.text_feats
        text_feats = text_feats.reshape(2, -1, 512)
        if text_feats.shape[1] > 1:
                self.maxw = True
        aesthetic_feature = self.video_encoder_aesthetic(aesthetic_video)
        technical_feature = self.video_encoder_technical(technical_video)
        overall_feat = self.mlp(torch.cat((aesthetic_feature, technical_feature), dim=-1))
        # assert not torch.any(torch.isnan(overall_feat))
        overall_feat = self.ln(overall_feat)
        # assert not torch.any(torch.isnan(overall_feat))
        if self.maxw: # i:2, j:不同指标(16), k:512, l:batchsize
            logitAes =  torch.einsum('ijk,lk -> lji', text_feats[:,1:7,:], 
                                     aesthetic_feature).softmax(-1)[...,0]
            logitTec =  torch.einsum('ijk,lk -> lji', text_feats[:,7:,:], 
                                     technical_feature).softmax(-1)[...,0]
            logit = torch.einsum('ik,lk -> li', text_feats[:,0,:], 
                                 overall_feat).softmax(-1)[...,0]
            res = torch.cat((logit.unsqueeze(-1), logitAes, logitTec), dim=-1)
        else:
            logit = torch.einsum('ijk,lk -> lji', text_feats, overall_feat)
            res = logit.softmax(-1)[..., 0]
        return res


class newModel(nn.Module):
    '''
    计算匹配度
    '''
    def __init__(self, n_frames=16):
        super().__init__()

        self.video_encoder_aesthetic = encodeMultiVideo(n_frames//2, num_clips=1)
        self.video_encoder_technical = eff_model.EVLTransformer(num_frames=n_frames)
        self.mlp = MLP(1024, 64, 512)
        self.dropout = nn.Dropout()
        self.ln = nn.LayerNorm([512])


    def forward(self, aesthetic_video, technical_video, tokens):

        text_feats = clip_model.encode_text(tokens).float()

        aesthetic_feature = self.video_encoder_aesthetic(aesthetic_video)
        technical_feature = self.video_encoder_technical(technical_video)
        overall_feat = self.mlp(torch.cat((aesthetic_feature, technical_feature), dim=-1))
        overall_feat = self.ln(overall_feat)

        res = torch.einsum('jk,jk -> j', text_feats, overall_feat).unsqueeze(-1)

        # text_feats = text_feats.reshape(2, -1, 512)
        # logit = torch.einsum('ijk,jk -> ji', text_feats, overall_feat)
        # res = logit.softmax(-1)[..., 0].unsqueeze(-1)

        return res


class EnhancedVisualEncoder(nn.Module):
    def __init__(self, clip_model, fast_vqa_encoder):
        super().__init__()
        self.clip_visual = clip_model.visual
        self.fast_vqa_encoder = fast_vqa_encoder.technical_backbone
        
    def forward(self, x_aes, x_tech):
        
        # frame-wise
        x_aes = x_aes.transpose(1,2).reshape(-1,3,224,224)
        clip_feats = self.clip_visual(x_aes)
        clip_feats = clip_feats[1:].reshape(7,7,-1,1024).permute(3,2,0,1)
        clip_feats = clip_feats.reshape(1024, -1, 64, 49).permute(1,2,3,0)

        # chunk-wise
        x_tech = x_tech.reshape(-1,3,4,32,224,224).permute(0,2,1,3,4,5).reshape(-1,3,32,224,224)
        fast_feats = self.fast_vqa_encoder(x_tech).reshape(-1,4,768,16,7,7).permute(0,1,3,4,5,2)
        fast_feats = fast_feats.reshape(-1,64,49,768)
        return torch.cat((clip_feats, fast_feats), -1)
    
    
class MaxVQA(nn.Module):
    """
        Modified CLIP, which combined prompt tuning and feature adaptation.
        The spatial and temporal naturalnesses are fed as final features.
        Implcit features is also optional fed into the model.
    """
    def __init__(self, text_tokens, embedding, n_ctx=1, share_ctx=False):
        
        super().__init__()
        with open("maxvqa.yml", "r") as f:
            opt = yaml.safe_load(f)   
        self.device = "cuda" 
        self.implicit_mlp = MLP(1792,64,1025)
        self.tokenized_prompts = text_tokens
        #self.text_encoder = TextEncoder(clip_model)
        fast_vqa_encoder = DOVER(**opt["model"]["args"]).to(Device)
        doverpath = r'C:\Users\Administrator\Desktop\maxvqa\DOVER\pretrained_weights\DOVER.pth'
        model, _, _ = open_clip_m.create_model_and_transforms("RN50",pretrained="openai")
        model = model.to(Device)
        fast_vqa_encoder.load_state_dict(torch.load(doverpath),strict=False)
        self.vis_encoder = EnhancedVisualEncoder(model,fast_vqa_encoder)
        self.share_ctx = share_ctx
        
        if n_ctx > 0:
            if not share_ctx:
                self.ctx = nn.Parameter(embedding[:, 1:1+n_ctx].clone())
            else:
                self.ctx = nn.Parameter(embedding[0:1, 1:1+n_ctx].clone())
        else:
            self.register_buffer("ctx", embedding[:, 1:1, :])
            print("Disabled Context Prompt")
        self.register_buffer("prefix", embedding[:, :1, :].clone())  # SOSzhe
        self.register_buffer("suffix", embedding[:, 1 + n_ctx:, :].clone())# CLS, EOS
        
        self.prefix.requires_grad = False
        self.suffix.requires_grad = False
        self.dropout = nn.Dropout(0.5)

        
        n_prompts = self.get_text_prompts()
        self.text_feats = text_encoder(n_prompts.cuda(), self.tokenized_prompts)
        
    def get_text_prompts(self):
        if self.share_ctx:
            return torch.cat(
                [
                    self.prefix,  # (n_cls, 1, dim)
                    self.ctx.repeat(self.prefix.shape[0],1,1),     # (n_cls, n_ctx, dim)
                    self.suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        return torch.cat(
                [
                    self.prefix,  # (n_cls, 1, dim)
                    self.ctx,     # (n_cls, n_ctx, dim)
                    self.suffix,  # (n_cls, *, dim)
                ],
                dim=1,
        )

        
    def initialize_inference(self):
        n_prompts = self.get_text_prompts()
        text_feats = text_encoder(n_prompts, self.tokenized_prompts)
        self.text_feats = text_feats 
            
    def forward(self, x_aes, x_tech, train=True, local=False):
        n_prompts = self.get_text_prompts()
        if train:
            text_feats = text_encoder(n_prompts, self.tokenized_prompts)
            self.text_feats = text_feats 
        else:
            text_feats = self.text_feats
        with torch.no_grad():
            vis_feat = self.vis_encoder(x_aes, x_tech)
            
        vis_feats = vis_feat.float()#.to(self.device)
        tmp_res = self.implicit_mlp(vis_feats)
        
        vis_feats = tmp_res[...,:1024]  + vis_feats[...,:1024]

        self.vis_feats = vis_feats 
        logits = 2 * self.dropout(self.vis_feats) @ text_feats.T
        
    
        res = logits.float().reshape(*logits.shape[:-1], 2, -1).transpose(-2,-1).softmax(-1)[...,0]
             
        if local:
            return res
        else:
            return res.mean((-3,-2))



if __name__ == '__main__':
    def test_all_model():
        video = r"coverr-a-man-prepares-to-launch-a-kite-2730-1080p.mp4"
        video_data = read_video(video)
        video_aes = video_data["aesthetic"].permute(1, 0, 2, 3).unsqueeze(0).to(device=Device, dtype=torch.float16)
        video_tec = video_data["technical"].permute(1, 0, 2, 3).unsqueeze(0).to(device=Device, dtype=torch.float16)
        prompts = [f'a X high quality picture', f'a X low quality picture']
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        text_tokens, embedding, text_feats = encode_text_prompts(tokenizer, prompts, device=Device)
        model = myModel(text_tokens, embedding, share_ctx=True).to(Device)
        model.train()
        with torch.no_grad() and autocast():
            for name, param in model.state_dict(keep_vars=True).items():
                if param.requires_grad == False:
                    print(name)
            begin_time = time.time()
            
            RES = model(video_aes, video_tec)
        print('res: ', RES)
        end_time = time.time()
        print(end_time - begin_time)


    # def train_mlp():
    #     mlp = MLP(1024, 64, 512)
    #     optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.0001)
    #     loss_func = nn.MSELoss()
    #     input = torch.randn(5, 1024)
    #     label = torch.ones(5)
    #     for i in range(3):
    #         with torch.autocast('cuda'):
    #             optimizer.zero_grad()
    #             output = mlp(input)
    #             res = nn.functional.sigmoid(output)[..., 0]
    #             loss = loss_func(res, label)
    #             loss.backward()
    #             optimizer.step()
    #             for name, parms in mlp.named_parameters():
    #                 print('------->name:', name)
    #                 print('------->para:', parms)
    #                 print('------->grad_value:', parms.grad)


    test_all_model()
