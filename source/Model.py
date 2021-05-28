import torch
from torch.nn import Linear
from torch.nn import Module
from transformers import BertModel


class Model(Module):
    def __init__(self, input_embedding_size: int) -> None:
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.start_predict = Linear(input_embedding_size, 2)
        self.end_predict = Linear(input_embedding_size, 2)
        self.match_predict_part1 = Linear(input_embedding_size, 1)
        self.match_predict_part2 = Linear(input_embedding_size, 1)

    def forward(self, sentence: torch.tensor,attention:torch.tensor,token_type:torch.tensor) -> torch.tensor:
        """
        :param sentence: input sentence combined by [cls] query [sep] answer[sep],shape(batch_size,seq_len_query+seq_len_context)
        :param start_label:binary label of start with tensor shape (batch-size,seq_len_query+seq_len_context)
        :param end_label:same to start_label
        :param compare_label:tensor indicate the label_pair,shape(batch_size,seq_len_query+seq_len_context,seq_len_query+seq_len_context)
        :return:the predict of start,end, and pair predict
        """
        embeddings = self.model(sentence,token_type_ids=token_type,attention_mask=attention, return_dict=True)['last_hidden_state']  # shape(batch,seq_len,embedding_size)
        start_predict = self.start_predict(embeddings)  # shape(batch,seq_len,2)
        end_predict = self.end_predict(embeddings)  # shape(batch,seq_len,2)
        start_index_mask = (torch.argmax(start_predict, -1) == 1)#shape batch,seq_len
        end_index_mask = (torch.argmax(end_predict, -1) == 1)#shape batch,seq_len
        total_mask=torch.bmm(start_index_mask.unsqueeze(2).float(),end_index_mask.unsqueeze(1).float()).bool()#shape:batch_size,seq_len,seq_len
        start_score = self.match_predict_part1(
            embeddings)  # it cost too much to concat two vectors and complute, so split it into two part and this makes no differences
        end_score = self.match_predict_part2(embeddings)
        predict_score =torch.bmm(end_score , start_score.squeeze().unsqueeze(1))  # batch_size,seq_len,seq_len
        return start_predict, end_predict, predict_score, total_mask
