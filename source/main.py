from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils import *
import torch
from Model import Model
from data_load import Data
import fitlog
from numpy import mean
from tqdm import tqdm

fitlog.set_log_dir('./logs')
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(optim: torch.optim, model: Model, data_loader: DataLoader, step: int) -> None:
    model.train()
    TP, FP, FN = 0, 0, 0
    losses = []
    for start_label, end_label, src, attention, token_type, pair in tqdm(data_loader):
        start_label=start_label.cuda()
        src=src.cuda()
        end_label=end_label.cuda()
        attention=attention.cuda()
        token_type=token_type.cuda()
        pair=pair.cuda()
        optim.zero_grad()
        start_predict, end_predict, match_score,total_mask = model(src, attention, token_type)
        start_loss, end_loss, predict_loss = cal_loss(start_predict, start_label, end_predict, end_label, match_score,
                                                      pair,total_mask)
        # start_accuracy=cal_accuracy(start_predict,start_label,attention)
        # end_accuracy=cal_accuracy(end_predict,end_label,attention)
        # print("training start predict_accuracy {} || end predict accuracy {}".format(start_accuracy,end_accuracy))
        total_loss = start_loss + end_loss + predict_loss
        total_loss.backward()
        optim.step()
        losses.append(total_loss.item())
        tp, fp, fn = match_TP(match_score, pair, total_mask)
        TP += tp
        FP += fp
        FN += fn
    pre, recall, f1 = cal_f1(TP, FP, FN)
    log_info(f1, pre, recall, 'train', mean(losses), step)


def eval(best_loss, model: Model, data_loader: DataLoader, step: int, not_decrease: int, test=None):
    model.eval()
    TP, FP, FN = 0, 0, 0
    losses = []
    for start_label, end_label, src, attention, token_type, pair in tqdm(data_loader):
        start_label=start_label.cuda()
        end_label=end_label.cuda()
        attention=attention.cuda()
        src=src.cuda()
        token_type=token_type.cuda()
        pair=pair.cuda()
        start_predict, end_predict, match_score,total_mask = model(src, attention, token_type)
        start_loss, end_loss, predict_loss = cal_loss(start_predict, start_label, end_predict, end_label, match_score,
                                                      pair,total_mask)
        total_loss = start_loss + end_loss + predict_loss
        losses.append(total_loss.item())
        tp, fp, fn = match_TP(match_score,pair, total_mask)
        TP += tp
        FP += fp
        FN += fn
    pre, recall, f1 = cal_f1(TP, FP, FN)
    if test == None:
        if mean(losses) < best_loss:
            torch.save(model.state_dict(), 'best_model.pkl')
            best_loss = mean(losses)
            not_decrease = 0
        else:
            not_decrease += 1
        log_info(f1, pre, recall, 'valid', mean(losses), step)
    else:
        log_info(f1, pre, recall, 'test', mean(losses), step)
    return best_loss, not_decrease


def main():
    setup_seed(0)
    args = get_args()
    train_dataset = DataLoader(Data(args.train_path), batch_size=64, collate_fn=collate_fn)
    valid_dataset = DataLoader(Data(args.valid_path), batch_size=64, collate_fn=collate_fn)
    test_dataset = DataLoader(Data(args.test_path), batch_size=64, collate_fn=collate_fn)
    model = Model(768)
    model=model.cuda()
    optim = AdamW(model.parameters(), lr=args.lr,betas=(0.99,0.98),eps=1e-9)
    best_losses = 1e4
    not_decrease = 0
    for i in range(args.epoch):
        train(optim, model=model, data_loader=train_dataset, step=i)
        best_losses, not_decrease = eval(best_loss=best_losses, model=model, data_loader=valid_dataset, step=i,
                                         not_decrease=not_decrease)
        if not_decrease >= 10:
            break

    model.load_state_dict(torch.load('./best_model.pkl'))
    eval(best_loss=best_losses, model=model, data_loader=test_dataset, step=args.epoch + 1, not_decrease=0, test='test')


main()
