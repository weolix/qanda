import os
import torch
from rich.progress import track
import numpy as np
import wandb
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.stats import pearsonr, spearmanr
import open_clip
import dbclip
import time

import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

Device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        video_data = dbclip.read_video(video_path, yml_path=r"F:/NTIREdataset/aigcc.yml")

        a = video_data["aesthetic"].permute(1, 0, 2, 3)
        t = video_data["technical"].permute(1, 0, 2, 3)

        # if(random.random() > 0.5):
        #     a = torch.flip(a, [3])
        #     t = torch.flip(t, [3])
        
        if self.mode == 'training':
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
            return a, t, description, label
        else:
            return a, t, description, video

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
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()



def train(model, train_set, val_set = None):
    train_loader  = DataLoader(train_set, 16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader    = DataLoader(val_set, 16, num_workers=4, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)

    wandb.init(
    project="aicg-vqa",
    config={
        
        "architecture": "newModel",
        "dataset": "aicg-vqa",
        "epochs": 30,
    }
    )
    best_mcc = 0.78
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

            prompts = desc
            # quality_prompt = "a high quality picture of "
            # prompts = [quality_prompt + des  for des in desc]

            tokens = open_clip.tokenize(prompts).to(Device)
            a = a.to(Device)
            t = t.to(Device)
            gt=gt.to(Device)

            with autocast():
                res = model(a, t, tokens)
                
                all_loss = 0.
                all_loss = plcc_loss(res[:, 0], gt[:, 0])
                wandb.log({'loss': all_loss})

            # with torch.autograd.detect_anomaly():
            all_loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += all_loss.detach().cpu()
            
        torch.save(model, 'pretrain/now.pth')
        wandb.log({'epoch loss': epoch_loss, 'epoch': epoch})

        if val_set is None:
            torch.save(model, f'pretrain/now_{epoch}.pth')

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
                torch.save(model, "pretrain/best.pth")
                print('saved!')
                best_mcc = mean_cc
        
        scheduler.step()
    
    wandb.finish()


def generate_result(model, test_data):
    import csv
    import zipfile
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

    with open(r"F:/NTIREdataset/output.txt", 'w', newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        for pr, name in zip(val_prs, vid_names):
            writer.writerow([name, pr])
    
    with open(r"F:/NTIREdataset/readme.txt", 'w', encoding='utf-8') as file:
        file.write(f"Runtime per video [s] : " + str(Runtime/len(vid_names)) + '\n'
                    "CPU[1] / GPU[0] : 0" + '\n' + 
                    "Extra Data [1] / No Extra Data [0] : 0" + '\n' +
                    "LLM [1] / No LLM [0] : 0" + '\n' + 
                    'Other description : ' + 'based on open_clip.' + '\n')
        
    zip = zipfile.ZipFile('submissions.zip', 'w')
    zip.write('output.txt')
    zip.write('readme.txt')
    zip.close()


if __name__ == '__main__':
    def main():
        generator = torch.Generator().manual_seed(37)

        aigc_trainset = NTIRE_Dataset(r'F:/NTIREdataset')
        aigc_valset = NTIRE_Dataset(r'F:/NTIREdataset', mode='val')

        model = dbclip.newModel(n_frames=16).to(Device)
        
        # state = torch.load(r'pretrain/now.pt')
        # model.load_state_dict(state, strict=True)
        aigc_trainset, aigc_valset = random_split(aigc_trainset, (0.9, 0.1), generator)
        train(model, aigc_trainset, aigc_valset)
        
        
        # generate_result(model, aigc_valset)

    main()