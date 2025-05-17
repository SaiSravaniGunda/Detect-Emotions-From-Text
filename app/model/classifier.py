import torch
import torch.nn as nn
from transformers import AutoModel

class EmotionClassifier(nn.Module):
    def __init__(self, n_classes=28):
        super(EmotionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = output[:, 0, :]
        return self.sigmoid(self.fc(self.drop(pooled)))
