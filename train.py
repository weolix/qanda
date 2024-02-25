import glob
import os
import torch
from rich.progress import track
import numpy as np
import wandb
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.stats import pearsonr, spearmanr
from copy import deepcopy
import pandas as pd
import open_clip
import dbclip
import time
Device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


class MaxFeatDataset(Dataset):
    def __init__(self, path_labelfile: str, feat_dir, video_names, mode):
        '''
        mode: 'train' or 'val'
        '''
        super().__init__()
        with open(video_names, 'r') as f:
            content = f.read()
            lines = content.splitlines()
        information = np.array([line.split(', ') for line in lines])
        self.video_names = information[:, 0]
        self.labels = torch.from_numpy(information[:, 3].astype(np.float32))
        csv = pd.read_csv(path_labelfile)
        self.gts = [csv.iloc[ind].values for ind in range(len(csv))]
        print('loading..')
        self.aes_feat = torch.load(feat_dir + '/maxvqa_aes_' + mode + '-maxwell.pkl')
        self.tec_feat = torch.load(feat_dir + '/maxvqa_tech_' + mode + '-maxwell.pkl')
        print('loaded!')

    def __getitem__(self, index):
        tec_feat = self.tec_feat[index].to(device=Device)
        aes_feat = self.aes_feat[index].to(device=Device)
        label = torch.tensor(self.gts[index], device=Device, dtype=torch.float32)
        return aes_feat, tec_feat, label
    
    def __len__(self):
        return len(self.gts)
    

class NTIRE_Dataset(Dataset):
    def __init__(self, data_dir, mode='training'):
        '''
        args:
            data_dir: 数据集根目录
            mode: 'training', 'val', 'test'
        '''
        super(NTIRE_Dataset, self).__init__()
        self.video_dir = os.path.join(data_dir, mode)
        self.data = []
        self.data_dir = data_dir
        self.mode = mode
        label_file = data_dir + '/' + mode + '.txt'
        if mode == 'training':
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    video_path, description, label = line.strip().split('|')
                    self.data.append((video_path, description, float(label)))
        else:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    video_path, description = line.strip().split('|')
                    self.data.append((video_path, description, 0.0))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        video, description, label = self.data[index]
        video_path = os.path.join(self.video_dir, video)
        video_data = dbclip.read_video(video_path, yml_path=r"C:\Users\Administrator\Desktop\aigcc.yml")

        a = video_data["aesthetic"].permute(1, 0, 2, 3).to(device=Device)
        t = video_data["technical"].permute(1, 0, 2, 3).to(device=Device)

        if self.mode == 'training':
            label = torch.tensor(label, device=Device, dtype=torch.float32).unsqueeze(0)
            return a, t, description, label
        else:
            return a, t, description, video

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



def train(model, train_set, val_set = None):
    train_loader, val_loader = DataLoader(train_set, 24, shuffle=True), DataLoader(val_set, 24)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
    
    wandb.init(
    project="aicg-vqa",
    config={
        
        "architecture": "newModel",
        "dataset": "aicg-vqa",
        "epochs": 30,
    }
    )
    best_mcc = 0.8
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(30):
        
        print('epoch:', epoch+1, '\t', 'lr:', scheduler.get_last_lr()[0])
        wandb.log({'epoch': epoch ,'lr': scheduler.get_last_lr()[0]})
        
        train
        model.train(True)
        
        epoch_loss = 0.
        for a, t, desc, gt in track(train_loader, description='train'):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            prompts = desc
            # prompts = [prompt + des for prompt in base_prompts for des in desc]
            tokens = open_clip.tokenize(prompts).to(Device)
            with autocast():
                res = model(a, t, tokens)
                
                all_loss = 0.
                if res.shape[-1] > 1:
                    for i in range(16):
                        loss = plcc_loss(res[:, i], gt[:, i])
                        all_loss += loss
                else:
                    all_loss = plcc_loss(res[:, 0], gt[:, 0])
                wandb.log({'loss': all_loss})

            # with torch.autograd.detect_anomaly():
            all_loss.backward()
            optimizer.step()

            epoch_loss += all_loss.detach().cpu()
            
        torch.save(model.state_dict(), 'pretrain/now.pt')
        wandb.log({'epoch loss': epoch_loss, 'epoch': epoch})

        if val_set is None:
            torch.save(model.state_dict(), f'pretrain/now_{epoch+30}.pt')

        else:
            # validate
            model.eval() 
            val_prs, val_gts = [], []
            for a, t, desc, gt in track(val_loader,description=' val '):
                # prompts = [prompt + des for prompt in base_prompts for des in desc]
                prompts = desc
                tokens = open_clip.tokenize(prompts).to(Device)
                with torch.no_grad() and autocast():
                    res = model(a, t, tokens)
                    
                val_prs.extend(list(res.detach().cpu().numpy()))
                val_gts.extend(list(gt.detach().cpu().numpy()))
                torch.cuda.empty_cache()
            val_prs = np.stack(val_prs, 0).squeeze()
            val_gts = np.stack(val_gts, 0).squeeze()
            mean_cc, plcc, srcc = 0, 0, 0
            print('\tsrcc\t\t\tplcc')
            if val_prs.ndim > 1:
                for i in range(16):
                    srcc = spearmanr(val_prs[:, i], val_gts[:, i])[0]
                    plcc = pearsonr(val_prs[:, i].squeeze(), val_gts[:, i].squeeze())[0]
                    print(srcc, '\t', plcc)
                    mean_cc = max(mean_cc, (plcc+srcc)/2)
                    wandb.log({f'plcc_{i}': plcc, f'srcc_{i}': srcc, 'epoch': epoch})
            else:
                srcc, plcc = spearmanr(val_prs, val_gts)[0], pearsonr(val_prs, val_gts)[0]
                wandb.log({'plcc': plcc, 'srcc': srcc, 'epoch': epoch})
                print(srcc, '\t', plcc)
                mean_cc = (plcc+srcc)/2
            if mean_cc > best_mcc:
                torch.save(model.state_dict(), "pretrain/best.pt")
                print('saved!')
                best_mcc = mean_cc
        
        scheduler.step()
        

def train_q(model, train_set, val_set):
    train_loader, val_loader = DataLoader(train_set, 16, shuffle=True), DataLoader(val_set, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
    
    wandb.init(
    project="aicg-vqa",
    config={
        
        "architecture": "myModel",
        "dataset": "aicg-vqa",
        "epochs": 30,
    }
    )
    best_mcc = 0.76
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(30):
        
        print('epoch:', epoch+1, '\t', 'lr:', scheduler.get_last_lr()[0])
        wandb.log({'epoch': epoch ,'lr': scheduler.get_last_lr()[0]})
        
        # train
        model.train(True)
        
        epoch_loss = 0.
        for a, t, desc, gt in track(train_loader, description='train'):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            
            with autocast():
                res = model(a, t)
                
                all_loss = 0.
                if res.shape[-1] > 1:
                    for i in range(16):
                        loss = plcc_loss(res[:, i], gt[:, i])
                        all_loss += loss
                else:
                    all_loss = plcc_loss(res[:, 0], gt[:, 0])
                wandb.log({'loss': all_loss})

            # with torch.autograd.detect_anomaly():
            all_loss.backward()
            optimizer.step()


            epoch_loss += all_loss.detach().cpu()
            

        torch.save(model.state_dict(), 'pretrain/now.pt')
        wandb.log({'epoch loss': epoch_loss, 'epoch': epoch})


        # validate
        model.eval() 
        val_prs, val_gts = [], []
        for a, t, desc, gt in track(val_loader,description=' val '):
            # prompts = [prompt + des for prompt in base_prompts for des in desc]
            
            
            with torch.no_grad() and autocast():
                res = model(a, t)
                
            val_prs.extend(list(res.detach().cpu().numpy()))
            val_gts.extend(list(gt.detach().cpu().numpy()))
            torch.cuda.empty_cache()
        val_prs = np.stack(val_prs, 0).squeeze()
        val_gts = np.stack(val_gts, 0).squeeze()
        mean_cc, plcc, srcc = 0, 0, 0
        print('\tsrcc\t\t\tplcc')
        if val_prs.ndim > 1:
            for i in range(16):
                srcc = spearmanr(val_prs[:, i], val_gts[:, i])[0]
                plcc = pearsonr(val_prs[:, i].squeeze(), val_gts[:, i].squeeze())[0]
                print(srcc, '\t', plcc)
                mean_cc = max(mean_cc, (plcc+srcc)/2)
                wandb.log({f'plcc_{i}': plcc, f'srcc_{i}': srcc, 'epoch': epoch})
        else:
            srcc, plcc = spearmanr(val_prs, val_gts)[0], pearsonr(val_prs, val_gts)[0]
            wandb.log({'plcc': plcc, 'srcc': srcc, 'epoch': epoch})
            print(srcc, '\t', plcc)
            mean_cc = (plcc+srcc)/2
        if mean_cc > best_mcc:
            torch.save(model.state_dict(), "pretrain/best.pt")
            print('saved!')
            best_mcc = mean_cc
        scheduler.step()



def test(model, data):
    # validate
    model.eval() 
    loader = DataLoader(data, 4)
    val_prs, val_gts = [], []
    for a, t, gt in track(loader,description=' val '):
        with torch.no_grad() and autocast():
            res = model(a, t)
            print(res)
        val_prs.extend(list(res.detach().cpu().numpy()))
        val_gts.extend(list(gt.detach().cpu().numpy()))
        torch.cuda.empty_cache()
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
    else:
        srcc, plcc = spearmanr(val_prs, val_gts)[0], pearsonr(val_prs, val_gts)[0]
        print(srcc, '\t', plcc)
        all_plcc = plcc


def generate_result(model, test_data):
    import csv
    model.eval()
    val_prs, vid_names = [], []
    val_loader = DataLoader(test_data, 16)
    
    Runtime = 0
    for a, t, desc, name in track(val_loader,description='test '):
        
        prompts = desc
        tokens = open_clip.tokenize(prompts).to(Device)
        with torch.no_grad() and autocast():
            t_begin = time.time()
            res = model(a, t, tokens)
            t_dur = time.time() - t_begin
            Runtime += t_dur
        val_prs.extend(res.squeeze(-1).detach().cpu().tolist())
        vid_names.extend(list(name))
        torch.cuda.empty_cache()

    with open(r"F:\NTIREdataset\output.txt", 'w', newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        for pr, name in zip(val_prs, vid_names):
            writer.writerow([name, pr])
    
    with open(r"F:\NTIREdataset\readme.txt", 'w', encoding='utf-8') as file:
        file.write(f"Runtime per video [s] : " + str(Runtime/len(vid_names)) + '\n'
                    "CPU[1] / GPU[0] : 0" + '\n' + 
                    "Extra Data [1] / No Extra Data [0] : 0" + '\n' +
                    "LLM [1] / No LLM [0] : 0" + '\n' + 
                    'Other description : ' + 'based on open_clip.' + '\n')



if __name__ == '__main__':
    def main():
        generator42 = torch.Generator().manual_seed(42)
        # max_files = r'C:\Users\Administrator\Desktop\maxvqa'
        # maxdir = max_files + r'\datasets\MaxWell'
        # max_train_label = max_files + r'\ExplainableVQA\MaxWell_train.csv'
        # max_val_label   = max_files + r'\ExplainableVQA\MaxWell_val.csv'
        # max_test_video  = max_files + r"\ExplainableVQA\examplar_data_labels\MaxWell\test_labels.txt"
        # max_train_video = max_files + r"\ExplainableVQA\examplar_data_labels\MaxWell\train_labels.txt"
        # Kset = KoNViDdataset('datasets\KoNViD.txt')
        # K_train_set, K_val_set = random_split(Kset, (0.8, 0.2), generator42)
        # M_train_set = MaxWelldataset(max_train_label, maxdir, max_train_video)
        # M_val_set = MaxWelldataset(max_val_label, maxdir, max_test_video)

        aigc_trainset = NTIRE_Dataset(r'F:\NTIREdataset')
        aigc_valset = NTIRE_Dataset(r'F:\NTIREdataset', mode='val')

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
        base_prompts = [f"a high quality picture of ", f"a low quality picture of "]
        prompts = [f'a {context} high quality picture', f'a {context} low quality picture']
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        text_tokens, embedding, text_feats = dbclip.encode_text_prompts(tokenizer, prompts, device=Device)
        # model_q = dbclip.myModel(text_tokens, embedding, n_frames=16).to(Device)
        model = dbclip.newModel(n_frames=16).to(Device)
        
        # model_ema = dbclip.myModel(text_tokens, embedding).to(Device)

        # test(model, K_val_set)
        # text_encoder = dbclip.TextEncoder()
        # model.initialize_inference()
        # Mtrain_set = MaxFeatDataset(max_train_label, 'features', max_train_video, mode='train')
        # Mval_set =  MaxFeatDataset(max_train_label, 'features', max_test_video, mode='val')
        # train(model, Mtrain_set, Mval_set,)
        # train(model, M_train_set, M_val_set,)
        # train(model, K_train_set, K_val_set)
        # state = torch.load(r'pretrain/now_47.pt')
        # model.load_state_dict(state, strict=True)
        # train(model, aigc_trainset)
        # train_q(model_q, ts, vs)

        # state = torch.load(r'pretrain/now_28.pt')
        # model.load_state_dict(state, strict=True)
        # generate_result(model, aigc_valset)

    main()