import os
import random
import torch
from rich.progress import track
import numpy as np
import wandb
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.stats import pearsonr, spearmanr
import open_clip
from datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
import dbclip
import time
import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

Device = 'cuda' if torch.cuda.is_available() else 'cpu'


data_option = {
    "technical": {
        "fragments_h": 7,
        "fragments_w": 7,
        "fsize_h": 32,
        "fsize_w": 32,
        "aligned": 16,
        "clip_len": 16,
        "frame_interval": 1,
        "num_clips": 1
    },
    "aesthetic": {
        "size_h": 224,
        "size_w": 224,
        "clip_len": 8,
        "frame_interval": 2,
        "t_frag": 8,
        "num_clips": 1
    }
}


def read_video(video_path: str):
    
    temporal_samplers = {}
    for sample_type, sample_option in data_option.items():
        if "t_frag" not in sample_option:
            # resized temporal sampling for TQE in DOVER
            temporal_samplers[sample_type] = UnifiedFrameSampler(16,1,1)
        else:
            # temporal sampling for AQE in DOVER
            temporal_samplers[sample_type] = UnifiedFrameSampler(8,1,2)
    mean = torch.FloatTensor([123.675, 116.28, 103.53]).reshape(-1,1,1,1)
    std = torch.FloatTensor([58.395, 57.12, 57.375]).reshape(-1,1,1,1)
    video_data, _ = spatial_temporal_view_decomposition(
        video_path, data_option, temporal_samplers,
    )
    va = (video_data["aesthetic"] - mean) / std
    vt = (video_data["technical"] - mean) / std
    return va, vt


class NTIRE_Dataset(Dataset):
    def __init__(self, data_dir, mode='training'):
        '''
        args:
            data_dir: 数据集根目录
            mode: 'train-all', 'train-train', 'train-val' 'val-all', 'test'
        '''
        super(NTIRE_Dataset, self).__init__()
        self.data = []
        self.data_dir = data_dir
        self.mode = mode
        label_file = data_dir + '/' + mode + '.txt'

        vdir = mode.split('-')[0]

        self.video_dir = os.path.join(data_dir, vdir)
        if 'train' in mode:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    video_path, description, label = line.strip().split('|')
                    self.data.append((video_path, description, float(label)))
        else: # test
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    video_path, description = line.strip().split('|')
                    self.data.append((video_path, description, 0.0))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        video_name, description, label = self.data[index]
        video_path = os.path.join(self.video_dir, video_name)
        a, t = read_video(video_path)

        a = a.permute(1, 0, 2, 3)
        t = t.permute(1, 0, 2, 3)

        if 'train' in self.mode:
            # if(random.random() > 0.5):
            #     a = torch.flip(a, [3])
            #     t = torch.flip(t, [3])
            # if(random.random() > 0.5):
            #     a = torch.flip(a, [2])
            #     t = torch.flip(t, [2])
            # label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
            return a, t, description, label
        else:
            return a, t, description, video_name

def rank_loss(y_pred, y):
    if(y.ndim == 1):
        y = y.unsqueeze(0)
    if(y_pred.ndim == 1):
        y_pred = y_pred.unsqueeze(0)
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
    loss0 = torch.nn.functional.l1_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.l1_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()



def train(model, train_set, val_set = None, epochs=80, batch_size=24, lr=0.0002):
    train_loader  = DataLoader(train_set, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader    = DataLoader(val_set, 16, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)

    wandb.init(
    project="aicg-vqa",
    config={
        
        "architecture": "newModel continue train with vqa",
        "dataset": "aicg-vqa",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
    }
    )
    best_mcc = 0.77
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        
        print('epoch:', epoch+1, '\t', 'lr:', optimizer.param_groups[0]['lr'])
        wandb.log({'epoch': epoch ,'lr': optimizer.param_groups[0]['lr']})
        
        # train
        model.train(True)
        
        epoch_loss = 0.
        for a, t, desc, gt in track(train_loader, description='train'):
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            # prompts = desc
            # quality_prompt = "a high quality picture of "
            # prompts = [quality_prompt + des  for des in desc]

            tokens = open_clip.tokenize(desc).to(Device)
            a = a.to(Device)
            t = t.to(Device)
            gt=gt.to(Device)

            with autocast():
                res = model(a, t, tokens, epoch)
                
                all_loss = 0.
                all_loss = plcc_loss(res, gt)
            wandb.log({'loss': all_loss})

            # with torch.autograd.detect_anomaly():
            all_loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += all_loss.detach().cpu()
            
        torch.save(model, 'pretrain/now.pth')
        print('epoch loss:', epoch_loss)
        wandb.log({'epoch loss': epoch_loss, 'epoch': epoch})

        if val_set is None:
             if (epoch+1) % 5 == 0:
                torch.save(model, f'pretrain/now_{epoch+1}.pth')
        

        else:
            # validate
            model.eval() 
            val_prs, val_gts = [], []
            for a, t, desc, gt in track(val_loader,description=' val '):
                # prompts = [prompt + des for prompt in base_prompts for des in desc]
                prompts = desc
                tokens = open_clip.tokenize(prompts).to(Device)
                a = a.to(Device)
                t = t.to(Device)
                gt=gt.to(Device)
                with torch.no_grad() and autocast():
                    res = model(a, t, tokens)
                    
                val_prs.extend(list(res.detach().cpu().numpy()))
                val_gts.extend(list(gt.detach().cpu().numpy()))
                torch.cuda.empty_cache()
            val_prs = np.stack(val_prs, 0).squeeze()
            val_gts = np.stack(val_gts, 0).squeeze()
            mean_cc, plcc, srcc = 0, 0, 0
            print('\tsrcc\t\t\tplcc')
            
            srcc, plcc = spearmanr(val_prs, val_gts)[0], pearsonr(val_prs, val_gts)[0]
            print(srcc, '\t', plcc)
            wandb.log({'plcc': plcc, 'srcc': srcc, 'epoch': epoch})
            mean_cc = (plcc+srcc)/2
            if mean_cc > best_mcc:
                torch.save(model, "pretrain/best.pth")
                print('saved!')
                best_mcc = mean_cc
        
        scheduler.step()
    
    wandb.finish()


def validate(model, val_set, norm='manmin'):
    '''
    args:
        model: 模型
        val_set: 验证集
        norm: 归一化方式，'manmin'或'zscore'
    '''
    val_loader = DataLoader(val_set, 16, num_workers=2, pin_memory=True)

    if issubclass(type(model), torch.nn.Module):
        model.eval()
        prs, gts = [], []
        for a, t, desc, gt in track(val_loader,description=' val '):
            tokens = open_clip.tokenize(desc).to(Device)
            a = a.to(Device)
            t = t.to(Device)
            gt=gt.to(Device)
            
            with torch.no_grad() and autocast():
                res = model(a, t, tokens)
            prs.extend(list(res.detach().cpu().numpy()))
            gts.extend(list(gt.detach().cpu().numpy()))
            torch.cuda.empty_cache()
        prs = np.stack(prs, 0).squeeze()


    elif isinstance(model, list):
        for m in model:
            m.eval()
        prs1, prs2, gts = [], [], []
        for a, t, desc, gt in track(val_loader,description=' val '):
            tokens = open_clip.tokenize(desc).to(Device)
            a = a.to(Device)
            t = t.to(Device)
            gt=gt.to(Device)
            with torch.no_grad():
                res1 = model[0](a, t, tokens)
            with torch.no_grad() and autocast():
                res2 = model[1](a, t, tokens)
            prs1.extend(list(res1.detach().cpu().numpy()))
            prs2.extend(list(res2.detach().cpu().numpy()))
            gts.extend(list(gt.detach().cpu().numpy()))
        prs1 = np.stack(prs1, 0).squeeze()
        prs2 = np.stack(prs2, 0).squeeze()
        if norm == 'manmin':
            pr1 = (prs1 - np.min(prs1)) / (np.max(prs1) - np.min(prs1))
            pr2 = (prs2 - np.min(prs2)) / (np.max(prs2) - np.min(prs2))
            prs = pr1 + pr2
        elif norm == 'zscore':
            pr1 = (prs1 - np.mean(prs1)) / np.std(prs1)
            pr2 = (prs2 - np.mean(prs2)) / np.std(prs2)
            prs = pr1 + pr2

    gts = np.stack(gts, 0).squeeze()

    srcc, plcc = spearmanr(prs, gts)[0], pearsonr(prs, gts)[0]
    print('\tsrcc\t\t\tplcc')
    print(srcc, '\t', plcc)


def generate_result(model, test_data):
    import csv
    import zipfile
    val_prs, vid_names = [], []
    val_loader = DataLoader(test_data, 16, num_workers=2, pin_memory=True)
    
    Runtime = 0
    if issubclass(type(model), torch.nn.Module):
        model.eval()
        for a, t, desc, name in track(val_loader,description='test '):
            
            tokens = open_clip.tokenize(desc).to(Device)
            a = a.to(Device)
            t = t.to(Device)

            with torch.no_grad() and autocast():
                t_begin = time.time()
                res = model(a, t, tokens)
                t_dur = time.time() - t_begin
                Runtime += t_dur
            
            val_prs.extend(res.detach().cpu().tolist())
            vid_names.extend(list(name))
            torch.cuda.empty_cache()

    elif isinstance(model, list):
        prs1, prs2 = [], []
        for m in model:
            m.eval()
        for a, t, desc, name in track(val_loader,description='test '):
            tokens = open_clip.tokenize(desc).to(Device)
            a = a.to(Device)
            t = t.to(Device)
            with torch.no_grad():
                res1 = model[0](a, t, tokens)
            with torch.no_grad() and autocast():
                res2 = model[1](a, t, tokens)
            prs1.extend(list(res1.detach().cpu().numpy()))
            prs2.extend(list(res2.detach().cpu().numpy()))
            vid_names.extend(list(name))
        prs1 = np.stack(prs1, 0).squeeze()
        prs2 = np.stack(prs2, 0).squeeze()

        pr1 = (prs1 - np.min(prs1)) / (np.max(prs1) - np.min(prs1))
        pr2 = (prs2 - np.min(prs2)) / (np.max(prs2) - np.min(prs2))
        val_prs = pr1 + pr2


    with open(r"../NTIREdataset/output.txt", 'w', newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        for pr, name in zip(val_prs, vid_names):
            writer.writerow([name, pr])
    
    with open(r"../NTIREdataset/readme.txt", 'w', encoding='utf-8') as file:
        file.write(f"Runtime per video [s] : " + str(Runtime/len(vid_names)) + '\n'
                    "CPU[1] / GPU[0] : 0" + '\n' + 
                    "Extra Data [1] / No Extra Data [0] : 0" + '\n' +
                    "LLM [1] / No LLM [0] : 0" + '\n' + 
                    'Other description : ' + 'based on open_clip.' + '\n')
        
    zp = zipfile.ZipFile('../NTIREdataset/submissions.zip', 'w')
    zp.write("../NTIREdataset/output.txt", "output.txt")
    zp.write("../NTIREdataset/output.txt", "readme.txt")
    zp.close()


if __name__ == '__main__':
    def main():
        generator = torch.Generator().manual_seed(37)

        train_trainset = NTIRE_Dataset(r'../NTIREdataset', mode='train-all')
        # train_trainset = NTIRE_Dataset(r'../NTIREdataset', mode='train-train')
        # train_valset = NTIRE_Dataset(r'../NTIREdataset', mode='train-val')
        # val_allset = NTIRE_Dataset(r'../NTIREdataset', mode='val-all')
        # testset = NTIRE_Dataset(r'../NTIREdataset', mode='test')

        model = [dbclip.fast_model(n_frames=10).to(Device), dbclip.newModel(16).to(Device)]
        # model[0].load_state_dict(torch.load(r'pretrain/fast.pth').state_dict(), strict=True)
        # model[1].load_state_dict(torch.load(r'pretrain/best7777.pth').state_dict(), strict=True)

        # model_pth = torch.load(r'pretrain/best7777.pth')
        # model.load_state_dict(model_pth.state_dict(), strict=False)
        # aigc_trainset, aigc_valset = random_split(aigc_trainset, (0.9, 0.1), generator)

        train(model[1], train_trainset)
        # validate(model, aigc_valset)
        
        # generate_result(model, val_allset)

    main()