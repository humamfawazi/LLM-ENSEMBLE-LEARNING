class SentencePairDataset(Dataset):
    """Custom Dataset untuk sentence-pair ABSA."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.data      = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            str(row['sentence_a']),
            str(row['sentence_b']),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {
            'input_ids'     : encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
        
        if 'label' in row:
            item['labels'] = torch.tensor(int(row['label']), dtype=torch.long)
        
        item['pair_id'] = str(row.get('pair_id', idx))
        return item

print("✓ SentencePairDataset siap.")