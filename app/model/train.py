import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.FloatTensor(self.labels[idx])
        }

class EmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = output[:, 0, :]
        return self.sigmoid(self.fc(self.drop(pooled)))

def train_model(model, train_loader, val_loader, device, epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCELoss()
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), "emotion_model.pt")
            print("✅ Model saved!")

def main():
    dataset = load_dataset("go_emotions")
    texts = dataset["train"]["text"]
    labels = dataset["train"]["labels"]

    n_classes = 28
    multi_hot = np.zeros((len(labels), n_classes))
    for i, label_set in enumerate(labels):
        multi_hot[i, label_set] = 1

    X_train, X_val, y_train, y_val = train_test_split(texts, multi_hot, test_size=0.1, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_data = EmotionDataset(X_train, y_train, tokenizer)
    val_data = EmotionDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier(n_classes).to(device)
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()
