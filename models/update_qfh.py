import torch
import torch.nn as nn
import numpy as np
from util import box_ops


def update_QFH_with_giou(
    class_embed,
    hs, 
    last_reference_out,
    hs_coords,     # [F, Q, 4] cxcywh
    topk,
    giou_center=-0.5,
    giou_gain=10.0,
    giou_weight_max=1.0,
    giou_weight_min=0.0,
    prob_weight_max=1.0,
    prob_weight_min=0.5,
    prob_weight_power=0.5,
    debugmode=False
    ):
    """
    Query Filter Head(QFH) considering GIoU.
    The topk queries with the highest QFH scores are selected for each frame.
    
    QFH score of frame f, query q: s(f, q) is calcurated as follows:
        1. Calcurate giou weight: w(f-1,r) from probability.
            p'(f-1,r) = p(f-1,r)^prob_weight_powe
            w(f-1,r) = (prob_weight_min + p'(f-1,r) * (prob_weight_max - prob_weight_min))
        
        2. Calcurate weighted giou: giou_w((f,q), (f-1,r)).
            giou_w((f,q), (f-1,r)) = giou((f,q), (f-1,r)) * w(f-1, r)
            giou_max(f,q,f-1) = max(giou_w((f,q), (f-1,r)))
        
        3. Similarly calculate giou_max(f,q,f+1) and then calculate giou_mean.
            giou_mean(f,q) = (giou_max(f,q,f-1) + giou_max(f,q,f+1)) / 2
        
        4. Calcurate probability gain: g(f,q).
            g_raw(f,q) = sigmoid(giou_gain * (giou_mean(f,q) - giou_center))
            g(f,q) = giou_weight_min + g_raw * (giou_weight_max - giou_weight_min)
        
        5. Calcurate s(f,q).
            s(f,q) = p(f,q) * g(f,q)
    """
    F, Q, _ = hs.shape
    device = hs.device

    # ---- Step 1: class prediction ----
    hs_logits = class_embed(hs)           # [F, Q, C]
    prob_all = hs_logits.sigmoid()
    prob_values, prob_indices = torch.max(prob_all[:, :, 1:], dim = -1)   # [F, Q]
    
    # ---- Step 2: convert bbox to xyxy ----
    xyxy = box_ops.box_cxcywh_to_xyxy(hs_coords)    # [F, Q, 4]

    # ---- Step 3: prepare prev / next frame pairs ----
    prev_xyxy = xyxy[:-1]    # [F-1, Q, 4]
    cur_prev  = xyxy[1:]     # [F-1, Q, 4]

    cur_next  = xyxy[:-1]    # [F-1, Q, 4]
    next_xyxy = xyxy[1:]     # [F-1, Q, 4]

    # ---- reshape for GIoU ----
    cur_prev_flat  = cur_prev.reshape(-1, 4)      # [(F-1)*Q, 4]
    prev_flat      = prev_xyxy.reshape(-1, 4)

    cur_next_flat  = cur_next.reshape(-1, 4)
    next_flat      = next_xyxy.reshape(-1, 4)

    # ---- Step 4: full GIoU matrix ----
    giou_prev_all = box_ops.generalized_box_iou(cur_prev_flat, prev_flat)
    giou_next_all = box_ops.generalized_box_iou(cur_next_flat, next_flat)
    # both: [(F-1)*Q, (F-1)*Q]

    # ---- Step 5: extract diagonal blocks (Q×Q) by loop ----
    giou_prev_blocks = torch.empty((F-1, Q, Q), device=device)
    giou_next_blocks = torch.empty((F-1, Q, Q), device=device)

    for f in range(F-1):
        s = f * Q
        e = s + Q

        giou_prev_blocks[f] = giou_prev_all[s:e, s:e]   # shape [Q, Q]
        giou_next_blocks[f] = giou_next_all[s:e, s:e]

    # ---- Step 6: class match mask WITHOUT boolean mask ----
    class_prev_match = (prob_indices[1:].unsqueeze(2) == prob_indices[:-1].unsqueeze(1))  # [F-1,Q,Q]
    class_next_match = (prob_indices[:-1].unsqueeze(2) == prob_indices[1:].unsqueeze(1))

    # masked_fill = boolean tensor: torch.Size([F-1,Q,Q])
    giou_prev_blocks = giou_prev_blocks.masked_fill(~class_prev_match, float('-inf'))
    giou_next_blocks = giou_next_blocks.masked_fill(~class_next_match, float('-inf'))

    # ---- Step 7: prob weighting ----
    prev_w = (prob_values[:-1] ** prob_weight_power).unsqueeze(1)  # [F-1,1,Q]
    next_w = (prob_values[1:] ** prob_weight_power).unsqueeze(1)   # [F-1,1,Q]

    giou_prev_w = giou_prev_blocks * (prob_weight_min + prev_w * (prob_weight_max - prob_weight_min))
    giou_next_w = giou_next_blocks * (prob_weight_min + next_w * (prob_weight_max - prob_weight_min))
    
    # ---- Step 8: reduction ----
    best_prev = giou_prev_w.max(dim=2).values  # [F-1, Q]
    best_next = giou_next_w.max(dim=2).values  # [F-1, Q]
    
    # ---- Step 9.A: finite mask ----
    valid_prev = torch.isfinite(best_prev)   # True if value is valid
    valid_next = torch.isfinite(best_next)
    
    # ---- Step 9.B: combined (valid only) ----
    combined = torch.zeros((F, Q), device=device)
    combined[1:]  += torch.where(valid_prev, best_prev, torch.zeros_like(best_prev))
    combined[:-1] += torch.where(valid_next, best_next, torch.zeros_like(best_next))

    # ---- Step 9.C: counts based on valid entries only ----
    counts = torch.zeros((F, Q), device=device)
    counts[1:]  += valid_prev.to(combined.dtype)
    counts[:-1] += valid_next.to(combined.dtype)
    
    # ---- Step 9.D: average & handle "both invalid" ----
    no_valid = (counts == 0)
    combined = combined / counts.clamp(min=1)
    combined[no_valid] = float('-inf')  # both invalid → -inf
    
    # ---- apply sigmoid gain ----
    raw_gain = torch.sigmoid(giou_gain * (combined - giou_center)) 
    gain = giou_weight_min + raw_gain * (giou_weight_max - giou_weight_min)
    adj_prob = prob_values * gain

    # ---- Step 10: top-k ----
    topk = min(topk, Q)
    _, topk_indices = torch.topk(adj_prob, topk, dim=1)

    hs = torch.gather(hs, 1, topk_indices.unsqueeze(-1).repeat(1,1,hs.shape[-1]))
    last_reference_out = torch.gather(
        last_reference_out, 1,
        topk_indices.unsqueeze(-1).repeat(1,1,last_reference_out.shape[-1])
    )
    
    if debugmode:
        print(f"hs_logits:\n{hs_logits[0]},\n{hs_logits[1]},\n       ...")
        print(f"giou_prev_al[0]l:\n{giou_prev_all[0]}")
        print(f"giou_next_all[0]:\n{giou_next_all[0]}")
        
        print(f"giou_prev_blocks[0]:\n{giou_prev_blocks[0]}")
        print(f"giou_next_blocks[0]:\n{giou_next_blocks[0]}")
        
        print(f"class_prev_match[0]:\n{class_prev_match[0]}")
        print(f"class_next_match[0]:\n{class_next_match[0]}")
        
        print(f"giou_prev_blocks after match[0]:\n{giou_prev_blocks[0]}")
        print(f"giou_next_blocks after match[0]:\n{giou_next_blocks[0]}")
        
        print(f"prev_w:\n{prev_w[0]},\n{prev_w[1]},\n       ...")
        print(f"next_w:\n{next_w[0]},\n{next_w[1]},\n       ...")
        
        print(f"giou_prev_w:\n{giou_prev_w[0]},\n{giou_prev_w[1]},\n       ...")
        print(f"giou_next_w:\n{giou_next_w[0]},\n{giou_next_w[1]},\n       ...")
        
        print(f"best_prev:\n{best_prev[0]},\n{best_prev[1]},\n       ...")
        print(f"best_next:\n{best_next[0]},\n{best_next[1]},\n       ...")
        
        print(f"devided combined\n{combined[0]},\n{combined[1]},\n       ...")
        print(f"adj_prob:\n{adj_prob[0]},\n{adj_prob[1]},\n       ...")
    
    return hs, last_reference_out


def giou_weighting(
    hs_logits,
    hs_coords,
    giou_center=-0.5,
    giou_gain=10.0,
    giou_weight_max=1.0,
    giou_weight_min=0.0,
    prob_weight_max=1.0,
    prob_weight_min=0.5,
    prob_weight_power=0.5,
    clamp_vals=(0.0, 1.0),
    return_score=True,
    debugmode=False
    ):
    """
    Weight the logits using GIoU with the BBox in the previous and next frames.
    GIoU weights (w) are calculated as follows:
        giou' = giou * (p_min + p^p_power * (p_max - p_min))
        w = (giou_min + sigmoid(giou') * (giou_max - giou_min)) 
    where:
        giou: Minimum GIoU with the same class BBox in the previous and next frames.
        p: BBox probability used in giou calculation.
    
    Args:
        hs_logits (torch.Size([num_frames, num_queries, num_classes])): 
            Output logits from model.
        hs_coords (torch.Size([num_frames, num_queries, 4])): 
            Output coords of BBox from model.
        giou_center (float): 
            The center of Sigmoid.
            When GIoU = giou_center, the weight is giou_weight_min + giou_weight_max / 2.
        giou_gain (float):
            The gain of Sigmoid.
        giou_weight_max (float):
            giou_max in the above formula.
        giou_weight_min (float):
            giou_min in the above formula.
        prob_weight_max (float):
            p_max in the above formula.
        prob_weight_min (float):
            p_min in the above formula.
        prob_weight_power (float):
            p_power in the above formula.
        clamp_vals ((a:float, b:float)):
            Logits are clamped to a minimum and b maximum.
        return_score (bool):
            If true, returns the max weighted probability and corresponding index for the query. 
                -> (torch.Size([num_frames, num_queries])), (torch.Size([num_frames, num_queries]))
            Otherwise, returns weighted logits. 
                -> (torch.Size([num_frames, num_queries, num_classes]))
        debugmode (bool):
            If True, shows debug infomation.
        
    Returns:
        adj_prob:
            Weighted logits.
    """
    # ---- Step 1: class prediction ----
    prob_all = hs_logits.sigmoid()
    prob_values, prob_indices = torch.max(prob_all[:, :, 1:], dim = -1)   # [F, Q]
    
    # ---- Step 2: convert bbox to xyxy ----
    xyxy = box_ops.box_cxcywh_to_xyxy(hs_coords)    # [F, Q, 4]

    # ---- Step 3: prepare prev / next frame pairs ----
    prev_xyxy = xyxy[:-1]    # [F-1, Q, 4]
    cur_prev  = xyxy[1:]     # [F-1, Q, 4]

    cur_next  = xyxy[:-1]    # [F-1, Q, 4]
    next_xyxy = xyxy[1:]     # [F-1, Q, 4]

    # ---- reshape for GIoU ----
    cur_prev_flat  = cur_prev.reshape(-1, 4)      # [(F-1)*Q, 4]
    prev_flat      = prev_xyxy.reshape(-1, 4)

    cur_next_flat  = cur_next.reshape(-1, 4)
    next_flat      = next_xyxy.reshape(-1, 4)

    # ---- Step 4: full GIoU matrix ----
    giou_prev_all = box_ops.generalized_box_iou(cur_prev_flat, prev_flat)
    giou_next_all = box_ops.generalized_box_iou(cur_next_flat, next_flat)
    # both: [(F-1)*Q, (F-1)*Q]

    # ---- Step 5: extract diagonal blocks (Q×Q) by loop ----
    giou_prev_blocks = torch.empty((F-1, Q, Q), device=device)
    giou_next_blocks = torch.empty((F-1, Q, Q), device=device)

    for f in range(F-1):
        s = f * Q
        e = s + Q

        giou_prev_blocks[f] = giou_prev_all[s:e, s:e]   # shape [Q, Q]
        giou_next_blocks[f] = giou_next_all[s:e, s:e]

    # ---- Step 6: class match mask WITHOUT boolean mask ----
    class_prev_match = (prob_indices[1:].unsqueeze(2) == prob_indices[:-1].unsqueeze(1))  # [F-1,Q,Q]
    class_next_match = (prob_indices[:-1].unsqueeze(2) == prob_indices[1:].unsqueeze(1))

    # masked_fill = boolean tensor: torch.Size([F-1,Q,Q])
    giou_prev_blocks = giou_prev_blocks.masked_fill(~class_prev_match, float('-inf'))
    giou_next_blocks = giou_next_blocks.masked_fill(~class_next_match, float('-inf'))

    # ---- Step 7: prob weighting ----
    prev_w = (prob_values[:-1] ** prob_weight_power).unsqueeze(1)  # [F-1,1,Q]
    next_w = (prob_values[1:] ** prob_weight_power).unsqueeze(1)   # [F-1,1,Q]

    giou_prev_w = giou_prev_blocks * (prob_weight_min + prev_w * (prob_weight_max - prob_weight_min))
    giou_next_w = giou_next_blocks * (prob_weight_min + next_w * (prob_weight_max - prob_weight_min))
    
    # ---- Step 8: reduction ----
    best_prev = giou_prev_w.max(dim=2).values  # [F-1, Q]
    best_next = giou_next_w.max(dim=2).values  # [F-1, Q]
    
    # ---- Step 9.A: finite mask ----
    valid_prev = torch.isfinite(best_prev)   # True if value is valid
    valid_next = torch.isfinite(best_next)
    
    # ---- Step 9.B: combined (valid only) ----
    combined = torch.zeros((F, Q), device=device)
    combined[1:]  += torch.where(valid_prev, best_prev, torch.zeros_like(best_prev))
    combined[:-1] += torch.where(valid_next, best_next, torch.zeros_like(best_next))

    # ---- Step 9.C: counts based on valid entries only ----
    counts = torch.zeros((F, Q), device=device)
    counts[1:]  += valid_prev.to(combined.dtype)
    counts[:-1] += valid_next.to(combined.dtype)
    
    # ---- Step 9.D: average & handle "both invalid" ----
    no_valid = (counts == 0)
    combined = combined / counts.clamp(min=1)
    combined[no_valid] = float('-inf')  # both invalid → -inf
    
    # ---- apply sigmoid gain ----
    raw_gain = torch.sigmoid(giou_gain * (combined - giou_center)) 
    gain = giou_weight_min + raw_gain * (giou_weight_max - giou_weight_min)
    if return_score:
        adj_prob = prob_values * gain
        torch.clamp(adj_prob, min=clamp_vals[0], max=clamp_vals[1])
        return adj_prob, prob_indices
    else:
        adj_logits = hs_logits.clone()

        # make indexing tensors
        f_idx = torch.arange(F).unsqueeze(1).expand(F, Q)  # shape [F,Q]
        q_idx = torch.arange(Q).unsqueeze(0).expand(F, Q)  # shape [F,Q]

        # apply gain only to the selected class
        adj_logits[f_idx, q_idx, prob_indices] *= gain
        torch.clamp(adj_logits, min=clamp_vals[0], max=clamp_vals[1])
        return adj_logits
