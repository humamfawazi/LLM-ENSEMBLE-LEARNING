def set_seed(seed: int):
    """Set semua random seed untuk reprodusibilitas."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compute_class_weights(labels: np.ndarray, num_classes: int = 4) -> torch.Tensor:
    """Hitung class weights dari distribusi label untuk menangani class imbalance."""
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    """Satu epoch training."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for batch in tqdm(loader, desc='  Training', leave=False):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if 'token_type_ids' in batch:
            kwargs['token_type_ids'] = batch['token_type_ids'].to(device)
        
        optimizer.zero_grad()
        outputs = model(**kwargs)
        logits  = outputs.logits
        
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, macro_f1


def evaluate(model, loader, criterion, device):
    """Evaluasi model pada dataloader, return loss, macro-F1, preds, probs."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs, all_pair_ids = [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='  Evaluating', leave=False):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if 'token_type_ids' in batch:
                kwargs['token_type_ids'] = batch['token_type_ids'].to(device)
            
            outputs = model(**kwargs)
            logits  = outputs.logits
            probs   = torch.softmax(logits, dim=-1).cpu().numpy()
            preds   = np.argmax(probs, axis=-1)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_pair_ids.extend(batch['pair_id'])
            
            if 'labels' in batch:
                labels = batch['labels'].to(device)
                loss   = criterion(logits, labels)
                total_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader) if all_labels else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if all_labels else 0.0
    return avg_loss, macro_f1, all_preds, all_probs, all_pair_ids, all_labels


def save_predictions(pair_ids, probs, preds, labels, filepath):
    """Simpan prediksi softmax ke CSV."""
    probs_arr = np.array(probs)
    rows = []
    for i, pid in enumerate(pair_ids):
        row = {
            'pair_id'               : pid,
            'prob_tidak_disebutkan' : round(float(probs_arr[i, 0]), 6),
            'prob_positif'          : round(float(probs_arr[i, 1]), 6),
            'prob_negatif'          : round(float(probs_arr[i, 2]), 6),
            'prob_netral'           : round(float(probs_arr[i, 3]), 6),
            'pred_label'            : int(preds[i]),
        }
        if labels:
            row['true_label'] = int(labels[i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(filepath, index=False, encoding='utf-8-sig')

print("✓ Fungsi training & evaluasi siap.")