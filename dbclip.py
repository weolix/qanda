import os
import yaml, torch
import torch.nn as nn

from Visual_Prompt import visual_prompt
import open_clip
from efficientvideorecognition import model as eff_model

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
Device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model = open_clip.load_openai_model('ViT-B-32')

    

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


class video_encoder(nn.Module):
    def __init__(self, video_len = 16):
        super().__init__()
        self.temp_fusion = visual_prompt(clip_model.state_dict(), video_len)
        self.vis_proj = nn.Linear(768, 512, bias=False)

    def forward(self, video):
        b, t, c, h, w = video.shape

        images = video.reshape(-1, c, h, w)
        with torch.no_grad():
            _, image_features = clip_model.encode_image(images)
            *_, width = image_features.shape
            image_features = image_features.mean(1, keepdim=False)
            image_features = image_features.reshape(b, t, width)

        image_features = self.vis_proj(image_features)
        image_feature = self.temp_fusion(image_features)
        image_feature = nn.functional.layer_norm(image_feature, [image_feature.shape[-1]])
        return image_feature


class newModel(nn.Module):
    '''
    计算图文匹配度和视频质量
    '''
    def __init__(self, n_frames=16):
        super().__init__()

        self.video_encoder_aesthetic = video_encoder(n_frames//2)
        self.video_encoder_technical = eff_model.EVLTransformer(num_frames=n_frames)
        self.mlp_visual = MLP(512+512, 64, 512)
        # self.dropout = nn.Dropout()
        self.ln = nn.LayerNorm([512])
        # self.mlp_tech = MLP(768, 64, 8)
        # self.vqa_head = nn.Linear(16, 1)
        # quality_prompt = ['a high quality picture']
        # with torch.no_grad():
        #     self.q_token = clip_model.encode_text(open_clip.tokenize(quality_prompt).to(Device)).float()

        # with open(r"F:/NTIREdataset/aigcc.yml", "r") as f:
        #     opt = yaml.safe_load(f)
        # fast_vqa_encoder = DOVER(**opt["model"]["args"])
        # fast_vqa_encoder.load_state_dict(torch.load(r"C:\Users\Administrator\Desktop\maxvqa\DOVER\pretrained_weights\DOVER.pth"),strict=False)
        # self.vqa = fast_vqa_encoder.technical_backbone

    def forward(self, aesthetic_video, technical_video, tokens, epoch=0):

        # 拼接文本特征
        # batch_size = aesthetic_video.shape[0]
        # self.q_batch_token = torch.cat([self.q_token] * batch_size, dim=0)
        text_feats = clip_model.encode_text(tokens).float()

        # # 生成query，key和value, 融合文本特征
        # query = self.query_proj(self.q_batch_token.unsqueeze(1))
        # key = self.key_proj(text_feats.unsqueeze(1))
        # value = self.value_proj(text_feats.unsqueeze(1))
        # fused_txt = self.attn(query, key, value)[0].squeeze(1)
            
        # fused_txt = self.txt_mlp(torch.cat((self.q_batch_token, text_feats), dim=-1))
        # fused_txt = fused_txt + text_feats

        # with torch.no_grad():
        #     fastvqa_feature = self.vqa(technical_video.permute(0, 2, 1, 3, 4)).mean((-1,-2,-3), keepdim=False)
        if epoch < 20:
            semantic_feature = self.video_encoder_aesthetic(aesthetic_video)
            technical_feature = self.video_encoder_technical(technical_video)
        elif epoch < 40:
            with torch.no_grad():
                semantic_feature = self.video_encoder_aesthetic(aesthetic_video)
            technical_feature = self.video_encoder_technical(technical_video)
        else:
            semantic_feature = self.video_encoder_aesthetic(aesthetic_video)
            with torch.no_grad():
                technical_feature = self.video_encoder_technical(technical_video)
            
        vis_feat = self.mlp_visual(torch.cat((semantic_feature, technical_feature), dim=-1))
        vis_feat = self.ln(vis_feat)
        # vis_feat = self.dropout(vis_feat)
        match = torch.einsum('bf,bf -> b', text_feats, vis_feat)
        
        # tech_feat = self.mlp_tech(fastvqa_feature)
        # res = self.vqa_head(torch.cat((match, tech_feat), dim=-1))

        return match
