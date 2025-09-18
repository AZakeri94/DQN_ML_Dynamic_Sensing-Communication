import torch as T


running_top1_acc = []
running_top2_acc = []
running_top3_acc = []
ave_top1_acc = 0
ave_top2_acc = 0
ave_top3_acc = 0
ind_ten = T.as_tensor([0, 1, 2, 3, 4], device='cuda:0')
top1_pred_out = []
top2_pred_out = []
top3_pred_out = []
total_count = 0

gt_beam = []

def topK_accu(out, labels):
    
    labels = labels.cuda()
    gt_beam.append(labels.detach().cpu().numpy()[0].tolist())
    total_count += labels.size(0)
    out = net.forward(x, pos)
    _, top_1_pred = T.max(out, dim=1)
    top1_pred_out.append(top_1_pred.detach().cpu().numpy()[0].tolist())
    sorted_out = T.argsort(out, dim=1, descending=True)
    
    top_2_pred = T.index_select(sorted_out, dim=1, index=ind_ten[0:2])
    top2_pred_out.append(top_2_pred.detach().cpu().numpy()[0].tolist())

    top_3_pred = T.index_select(sorted_out, dim=1, index=ind_ten[0:3])
    top3_pred_out.append(top_3_pred.detach().cpu().numpy()[0].tolist()  )
        
    reshaped_labels = labels.reshape((labels.shape[0], 1))
    tiled_2_labels = reshaped_labels.repeat(1, 2)
    tiled_3_labels = reshaped_labels.repeat(1, 3)
    
    batch_top1_acc = T.sum(top_1_pred == labels, dtype=T.float32)
    batch_top2_acc = T.sum(top_2_pred == tiled_2_labels, dtype=T.float32)
    batch_top3_acc = T.sum(top_3_pred == tiled_3_labels, dtype=T.float32)

    ave_top1_acc += batch_top1_acc.item()
    ave_top2_acc += batch_top2_acc.item()
    ave_top3_acc += batch_top3_acc.item()
        
