import glob
import os
import torch
from rich.progress import track
import numpy as np
import wandb
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import open_clip
import dbclip

Device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = open_clip.get_tokenizer("ViT-B-32")

wandb.init(
    project="myproj",
    config={
        "learning_rate": 0.001,
        "architecture": "ViT-B-32",
        "dataset": "konvid-1k",
        "epochs": 20,
}
)


class KoNViDdataset(Dataset):
    def __init__(self, path_to_labelfile: str):
        super().__init__()
        with open(path_to_labelfile, 'r') as f:
            content = f.read()
            lines = content.splitlines()
        informations = np.array([line.split(', ') for line in lines])
        self.video_paths = informations[:, 0]
        self.labels = torch.from_numpy(informations[:, 3].astype(np.float32))

    def __getitem__(self, index):
        feat_path = 'E:/features/k/' + self.video_paths[index][-14:-4] + '.pt'
        if glob.glob(feat_path):
            video_data = torch.load(feat_path)
        else:
            video_data = dbclip.read_video(self.video_paths[index])
            torch.save(video_data, feat_path)
        video_tec = video_data["technical"].permute(1, 0, 2, 3).to(device=Device)
        video_aes = video_data["aesthetic"].permute(1, 0, 2, 3).to(device=Device)
        label = self.labels[index].to(Device).unsqueeze(0)
        return video_aes, video_tec, label

    def __len__(self):
        return len(self.labels)


class MaxWelldataset(Dataset):
    def __init__(self, path_labelfile: str, path_video_dir, video_names):
        super().__init__()
        with open(video_names, 'r') as f:
            content = f.read()
            lines = content.splitlines()
        information = np.array([line.split(', ') for line in lines])
        self.video_names = information[:, 0]
        self.labels = torch.from_numpy(information[:, 3].astype(np.float32))
        csv = pd.read_csv(path_labelfile)
        self.video_dir = path_video_dir
        self.gts = [csv.iloc[ind].values for ind in range(len(csv))]

    def __getitem__(self, index):

        video_name = self.video_names[index]
        feat_path = 'E:/features/m/' + video_name + '.pt'
        if glob.glob(feat_path):
            video_data = torch.load(feat_path)
        else:
            video_data = dbclip.read_video(os.path.join(self.video_dir, video_name))
            torch.save(video_data, feat_path)

        video_tec = video_data["technical"].permute(1, 0, 2, 3).to(device=Device)
        video_aes = video_data["aesthetic"].permute(1, 0, 2, 3).to(device=Device)
        label = torch.tensor(self.gts[index], device=Device, dtype=torch.float32)
        return video_aes, video_tec, label

    def __len__(self):
        return len(self.gts)

class MaxWellFeatset(Dataset):
    def __init__(self, path_labelfile: str, feat_dir, video_names):
        super().__init__()
        with open(video_names, 'r') as f:
            content = f.read()
            lines = content.splitlines()
        information = np.array([line.split(', ') for line in lines])
        self.video_names = information[:, 0]
        self.labels = torch.from_numpy(information[:, 3].astype(np.float32))
        csv = pd.read_csv(path_labelfile)
        self.video_dir = feat_dir
        self.gts = [csv.iloc[ind].values for ind in range(len(csv))]

    def __getitem__(self, index):

        video_name = self.video_names[index]
        feat_path = 'E:/features/m/' + video_name + '.pt'
        if glob.glob(feat_path):
            video_data = torch.load(feat_path)
        else:
            video_data = dbclip.read_video(os.path.join(self.video_dir, video_name))
            torch.save(video_data, feat_path)

        video_tec = video_data["technical"].permute(1, 0, 2, 3).to(device=Device)
        video_aes = video_data["aesthetic"].permute(1, 0, 2, 3).to(device=Device)
        label = torch.tensor(self.gts[index], device=Device, dtype=torch.float32)
        return video_aes, video_tec, label

    def __len__(self):
        return len(self.gts)


def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
            torch.sum(ranking_loss) / y_pred.shape[0] / scale
    ).float()


def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()



def train(model, train_set, val_set):
    train_loader, val_loader = DataLoader(train_set, 4, shuffle=True), DataLoader(val_set, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    
    best_plcc = 0.8
    model.train()
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(20):
        print('epoch:', epoch+1, '\t', 'lr:', scheduler.get_last_lr()[0])
        wandb.log({'epoch': epoch ,'lr': scheduler.get_last_lr()[0]})
        
        # train
        epoch_loss = 0.
        for a, t, gt in track(train_loader,description=str()+' train'):
            optimizer.zero_grad()
            with autocast():
                res = model(a, t)
            loss, aux_loss = 0, 0
            if res.shape[-1] > 1:
                all_loss = 0.
                for i in range(16):
                    loss = plcc_loss(res[:, i], gt[:, i])
                    all_loss += loss
            else:
                all_loss = plcc_loss(res[:, 0], gt[:, 0])
            wandb.log({'loss': all_loss})

            # wandb.log({"loss": loss.item(), "aux_loss": aux_loss.item()})
            # loss += aux_loss
            # with torch.autograd.detect_anomaly():
            all_loss.backward()
            optimizer.step()

            epoch_loss += all_loss

            torch.save(model.state_dict(), 'pretrain/now.pt')

        wandb.log({'epoch loss': epoch_loss, 'epoch': epoch})

        # validate
        model.eval()
        val_prs, val_gts = [], []
        for a, t, gt in track(val_loader,description=str(i)+' val'):
            with torch.no_grad() and autocast():
                res = model(a, t)
                val_prs.extend(list(res.detach().cpu().numpy()))
                val_gts.extend(list(gt.detach().cpu().numpy()))
        val_prs = np.stack(val_prs, 0).squeeze()
        val_gts = np.stack(val_gts, 0).squeeze()
        all_plcc, plcc, srcc = 0, 0, 0
        print('\tsrcc\t\t\tplcc')
        if val_prs.ndim > 1:
            for i in range(16):
                srcc = spearmanr(val_prs[:, i], val_gts[:, i])[0]
                plcc = pearsonr(val_prs[:, i].squeeze(), val_gts[:, i].squeeze())[0]
                print(srcc, '\t', plcc)
                all_plcc = max(all_plcc, plcc)
                wandb.log({f'plcc_{i}': plcc, f'srcc_{i}': srcc, 'epoch': epoch})
        else:
            srcc, plcc = spearmanr(val_prs, val_gts)[0], pearsonr(val_prs, val_gts)[0]
            wandb.log({'plcc': plcc, 'srcc': srcc, 'epoch': epoch})
            print(srcc, '\t', plcc)
            all_plcc = plcc
        if all_plcc > best_plcc:
            torch.save(model.state_dict(), "pretrain/best.pt")
            print('saved!')
            best_plcc = all_plcc

        # val_loss = 0
        # for a, t, label in val_loader:
        #     with torch.no_grad():
        #         score_predict = model(a,t,prmpt)
        #         val_loss += torch.nn.functional.mse_loss(score_predict, label)
        # wandb.log({"val_loss": val_loss.item()})
        # if val_loss < val_loss_min:
        #     torch.save(model.state_dict(), 'best.pt')
        scheduler.step()


# def train_maxvqa():
#     tokenizer = open_clip.get_tokenizer("RN50")

#     text_tokens, embedding, text_feats = encode_text_prompts(pos_prompts + neg_prompts, device=device)

#     ## Load model
#     text_encoder = TextEncoder(model).to(device)
#     visual_encoder = EnhancedVisualEncoder(model, fast_vqa_encoder).to(device)

#     maxvqa = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)






















if __name__ == '__main__':
    generator42 = torch.Generator().manual_seed(42)
    Kset = KoNViDdataset('datasets\KoNViD.txt')
    K_train_set, K_val_set = random_split(Kset, (0.8, 0.2), generator42)
    max_files = r'C:\Users\Administrator\Desktop\maxvqa'
    maxdir = max_files + r'\datasets\MaxWell'
    max_train_label = max_files + r'\ExplainableVQA\MaxWell_train.csv'
    max_val_label   = max_files + r'\ExplainableVQA\MaxWell_val.csv'
    max_test_video  = max_files + r"\ExplainableVQA\examplar_data_labels\MaxWell\test_labels.txt"
    max_train_video = max_files + r"\ExplainableVQA\examplar_data_labels\MaxWell\train_labels.txt"
    M_train_set = MaxWelldataset(max_train_label, maxdir, max_train_video)
    M_val_set = MaxWelldataset(max_val_label, maxdir, max_test_video)

    positive_descs = ["high quality", "good content", "organized composition", "vibrant color",
                      "contrastive lighting", "consistent trajectory",
                      "good aesthetics",
                      "sharp", "in-focus", "noiseless", "clear-motion", "stable", "well-exposed",
                      "original", "fluent", "clear",
                      ]

    negative_descs = ["low quality", "bad content", "chaotic composition", "faded color", "gloomy lighting",
                      "incoherent trajectory",
                      "bad aesthetics",
                      "fuzzy", "out-of-focus", "noisy", "blurry-motion", "shaky", "poorly-exposed",
                      "compressed", "choppy", "severely degraded",
                      ]
    context = "X"
    pos_prompts = [f"a {context} {desc} picture" for desc in positive_descs]
    neg_prompts = [f"a {context} {desc} picture" for desc in negative_descs]
    Mprompts = pos_prompts + neg_prompts
    prompts = [f'a {context} high quality picture', f'a {context} low quality picture']
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text_tokens, embedding, text_feats = dbclip.encode_text_prompts(prompts, device=Device)

    model = dbclip.myModel(text_tokens, embedding, share_ctx=True).to(Device)
    state = torch.load(r'best.pt')
    model.load_state_dict(state, strict=True)
    model.initialize_inference()
    # train(model, M_train_set, M_val_set,)
    train(model, K_train_set, K_val_set)
