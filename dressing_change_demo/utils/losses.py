import torch

def intra_contrastive_loss(features, labels, temperature, base_temperature):
    logits = torch.matmul(features, features.T) / temperature
    mask = torch.eq(labels, labels.T).float().to(features.device)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(mask.shape[0]).view(-1, 1).to(features.device),
        0
    )
    mask = mask * logits_mask

    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    mask_pos_pairs = mask.sum(1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

    loss = - (temperature / base_temperature) * mean_log_prob_pos.mean()
    return loss

def cross_contrastive_loss(features1, labels1, features2, labels2, temperature, base_temperature):
    logits = torch.matmul(features1, features2.T) / temperature
    mask = torch.eq(labels1, labels2.T).float().to(features1.device)

    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    mask_pos_pairs = mask.sum(dim=1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_pos_pairs

    loss = - (temperature / base_temperature) * mean_log_prob_pos.mean()
    
    _, preds = logits.topk(1, dim=1)
    
    return loss, preds
