from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    auc = roc_auc_score(all_labels, all_preds[:, 1])
    ap = average_precision_score(all_labels, all_preds[:, 1])
    print(f"AUC: {auc}, AP: {ap}")
