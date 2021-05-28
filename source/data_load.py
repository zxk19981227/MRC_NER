from transformers import BertTokenizer
from torch.utils.data import Dataset
import json
import torch
from tqdm import tqdm

class Data(Dataset):
    def convert_Json_to_data(self,json):
        current_token = self.token(json['query'], text_pair=json['context'])
        # self.query.append(js['query'])
        # self.context.append(js['context'])
        # self.starts.append(js['start_position'])
        # self.ends.append(js['end_position'])
        query_length = len(self.token(json['query'])['input_ids'])

        start_label = torch.zeros(len(current_token['input_ids']))
        end_label = torch.zeros(len(current_token['input_ids']))
        for end_position in json['end_position']:
            end_label[end_position + query_length] = 1
        for start_position in json['start_position']:
            start_label[start_position + query_length] = 1
        # current_pre=[]
        pair_label = torch.zeros(len(current_token['input_ids']), len(current_token['input_ids']))
        for pair in json['span_position']:
            start, end = pair.split(';')
            start = int(start) + query_length
            end = int(end) + query_length
            pair_label[start, end] = 1
        # end_positions = js['end_position']+query_length
        # position_pair = js['span_position']
        return start_label,end_label,pair_label,current_token
    def __init__(self, dataset_path):
        self.token = BertTokenizer.from_pretrained('bert-base-uncased')
        orig_read = open(dataset_path)
        jsons = json.load(orig_read)
        self.encode_tensor=[]
        self.start_label=[]
        self.end_label=[]
        self.pair_label=[]
        self.attention_mask=[]
        self.token_type=[]
        print("loading data:")
        for js in tqdm(jsons):
            start_label,end_label,pair_label,current_token=self.convert_Json_to_data(js)
            self.start_label.append(start_label)
            self.end_label.append(end_label)
            self.pair_label.append(pair_label)
            self.encode_tensor.append(torch.tensor(current_token['input_ids']))
            self.attention_mask.append(torch.tensor(current_token['attention_mask']))
            self.token_type.append(torch.tensor(current_token['token_type_ids']))

    def __len__(self):
        return len(self.encode_tensor)

    def __getitem__(self, item):
        return self.start_label[item],self.end_label[item],self.pair_label[item],self.encode_tensor[item],self.attention_mask[item],self.token_type[item]


