import torch
from torch.nn import CrossEntropyLoss, Linear, MultiheadAttention
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
import time

class EntityDataset(Dataset):
    def __init__(self, entity_file, label_file, tokenizer, max_length=128):
        with open(entity_file, 'r', encoding='utf-8') as f:
            self.entities = [line.strip() for line in f.readlines()]
        with open(label_file, 'r', encoding='utf-8') as f:
            self.labels = [line.strip() for line in f.readlines()]
        assert len(self.entities) == len(self.labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, idx):
        entity = self.entities[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(entity, padding='max_length', truncation=True, max_length=self.max_length,
                                return_tensors='pt')
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        return inputs, label

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    input_ids = torch.stack([x['input_ids'] for x in inputs])
    attention_mask = torch.stack([x['attention_mask'] for x in inputs])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels


def encode_entities(model, entities, tokenizer, device):
    encoded_entities = []
    for entity in entities:
        inputs = tokenizer(entity, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        encoded_entity = outputs.cpu().numpy()
        encoded_entities.append(encoded_entity)
    all_encoded_entities = np.vstack(encoded_entities)
    return all_encoded_entities


class FusionModel(torch.nn.Module):
    def __init__(self, fixed_bert, tunable_bert, hidden_dim, num_heads=8):
        super(FusionModel, self).__init__()
        self.fixed_bert = fixed_bert
        self.tunable_bert = tunable_bert
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.multihead_attention = MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        with torch.no_grad():
            fixed_outputs = self.fixed_bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        fixed_embeddings = fixed_outputs.last_hidden_state
        tunable_outputs = self.tunable_bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        tunable_embeddings = tunable_outputs.last_hidden_state
        combined_embeddings = torch.cat((fixed_embeddings, tunable_embeddings), dim=1)
        extended_attention_mask = torch.cat((attention_mask, attention_mask), dim=1)
        attn_output, _ = self.multihead_attention(query=combined_embeddings, key=combined_embeddings, value=combined_embeddings, key_padding_mask=~extended_attention_mask.bool())
        fused_embedding = attn_output.mean(dim=1)
        return fused_embedding

def save_attention_layer_params(model, epoch, output_dir):
    attention_layer_params = {name: param.data for name, param in model.named_parameters() if 'multihead_attention' in name}
    torch.save(attention_layer_params, os.path.join(output_dir, f'multihead_attention_epoch_{epoch}.pt'))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    fixed_bert = BertModel.from_pretrained('bert-base-chinese').to(device)
    tunable_bert = BertModel.from_pretrained('bert-base-chinese').to(device)
    for param in fixed_bert.parameters():
        param.requires_grad = False

    hidden_dim = 768
    model = FusionModel(fixed_bert, tunable_bert, hidden_dim).to(device)
    train_dataset = EntityDataset(
        entity_file='train_data.txt',
        label_file='lable.txt',
        tokenizer=tokenizer
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    with open('kg_entity.txt', 'r', encoding='utf-8') as f:
        all_entities = [line.strip() for line in f.readlines()]
    criterion = CrossEntropyLoss()
    num_epochs = 60
    output_dir = 'model_fusion'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (batch_inputs, batch_labels) in enumerate(train_loader):
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            optimizer.zero_grad()
            fused_embeddings = model(**batch_inputs)
            assert fused_embeddings.shape == (batch_inputs['input_ids'].size(0), hidden_dim)
            all_encoded_entities = encode_entities(model, all_entities, tokenizer, device)
            all_encoded_entities_tensor = torch.tensor(all_encoded_entities).to(device)
            similarities = torch.matmul(fused_embeddings, all_encoded_entities_tensor.T)
            _, predictions = torch.max(similarities, dim=1)
            label_to_index = {entity: idx for idx, entity in enumerate(all_entities)}
            target_indices = torch.tensor([label_to_index[label] for label in batch_labels]).to(device)
            loss = criterion(similarities, target_indices)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            epoch_output_dir = os.path.join(output_dir, f'epoch_{epoch + 1}')
            if not os.path.exists(epoch_output_dir):
                os.makedirs(epoch_output_dir)
            save_attention_layer_params(model, epoch + 1, epoch_output_dir)
            model.tunable_bert.save_pretrained(epoch_output_dir)
            tokenizer.save_pretrained(epoch_output_dir)
            optimizer_state_path = os.path.join(epoch_output_dir, 'optimizer_state.pth')
            torch.save(optimizer.state_dict(), optimizer_state_path)
            print(f"Model and optimizer state saved at epoch {epoch + 1}.")


if __name__ == '__main__':
    main()

