from torch.nn.functional import nll_loss, binary_cross_entropy_with_logits,log_softmax
import torch
import fitlog
import argparse
from torch.nn.utils.rnn import pad_sequence
import numpy as  np
import random

def log_info(f1: float, precision: float, recall: float, usage: str, loss: float, step: int):
    fitlog.add_loss(loss, step, name=usage)
    fitlog.add_metric(f1, step, name=usage + ' f1_score')
    fitlog.add_metric(precision, step, name=usage + 'precision')
    fitlog.add_metric(recall, step, name=usage + 'recall')
    print("{} epoch {} f1 score {}  precision {} recall {} loss {}".format(usage, step, f1, precision, recall,loss))


def cal_loss(start_predict, start_label, end_predict, end_label, pair_predict, pair_label, total_mask):
    """
    just to calculate the total loss of the output
    :param total_mask:
    :param start_predict:
    :param start_label:
    :param end_predict:
    :param end_label:
    :param pair_predict:
    :param pair_label:
    :return:
    """
    loss_start = nll_loss(log_softmax(start_predict.view(-1, 2),-1), start_label.view(-1).long())
    loss_end = nll_loss(log_softmax(end_predict.view(-1, 2),-1), end_label.view(-1).long())
    loss_match = (binary_cross_entropy_with_logits(pair_predict, pair_label.float(),
                                                   reduction='none') * total_mask).sum() / (
                     total_mask.sum() if total_mask.sum().item() != 0 else 1)
    return loss_start, loss_end, loss_match


def cal_accuracy(predict, label, mask):
    """
    get start/end predict accuracy
    :param predict:
    :param label:
    :param source:
    :return:
    """
    return ((torch.argmax(predict, -1) == label) & (mask != 0)).sum().item() / (mask != 0).sum().item()


def cal_f1(TP, FP, True_FN):
    if TP==0:
        return 0,0,0
    precision = TP / (TP + FP)
    recall = TP / (TP + True_FN)
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


def match_TP(predict, compare_label, total_mask):
    match_predicts=(torch.log(predict).ge(0) & total_mask)
    TP = ((compare_label==1) & match_predicts).sum().item()
    FP = (~(compare_label == 1) & match_predicts).sum().item()
    # TN = ((match_predicts == compare_label) & (compare_label != 1)).sum().item()
    FN = (match_predicts & ~(compare_label==1)).sum().item()

    return TP, FP, FN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=2000)
    args = parser.parse_args()
    return args


def pad(tensors, sentence_length):
    padding_tensor = []
    for tensor in tensors:
        tensor = tensor.numpy().tolist()
        for i in range(len(tensor)):
            while len(tensor[i]) < sentence_length:
                tensor[i].append(0)
        padding_value = [0 for i in range(sentence_length)]
        while len(tensor) < sentence_length:
            tensor.append(padding_value)
        tensor = torch.tensor(tensor)
        assert tensor.shape[0] == tensor.shape[1] & tensor.shape[0] == sentence_length
        padding_tensor.append(tensor)
    return torch.stack(padding_tensor, 0)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def collate_fn(batch):
    """
    :param batch: a batch dataset in training dataset
    :return:
    """
    start_labels = []
    end_laebls = []
    pair_labels = []
    contexts = []
    attentions = []
    token_styles = []
    for start_label, end_label, pair_label, context, attention, token_tyle in batch:
        start_labels.append(start_label)
        end_laebls.append(end_label)
        pair_labels.append(pair_label)
        contexts.append(context)
        attentions.append(attention)
        token_styles.append(token_tyle)
    start_labels = pad_sequence(start_labels, batch_first=True, padding_value=0)
    end_labels = pad_sequence(end_laebls, batch_first=True, padding_value=0)
    contexts = pad_sequence(contexts, batch_first=True, padding_value=0)
    attentions = pad_sequence(attentions, batch_first=True, padding_value=0)
    token_types = pad_sequence(token_styles, batch_first=True, padding_value=0)
    max_sentence = start_labels.shape[1]
    pair_label = pad(pair_labels, max_sentence).long()
    return start_labels, end_labels, contexts, attentions, token_types, pair_label
