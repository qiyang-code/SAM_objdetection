import torch, clip, cv2, numpy as np, pickle, os
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

CKPT = 'checkpoints/sam_vit_b_01ec64.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

sam = sam_model_registry["vit_b"](checkpoint=CKPT).to(DEVICE)
predictor = SamPredictor(sam)
clip_m, preprocess = clip.load("ViT-B/32", device=DEVICE)

def compute_iou(a, b):
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    xi = max(x1, x2)
    yi = max(y1, y2)
    wi = max(min(x1+w1, x2+w2) - xi, 0)
    hi = max(min(y1+h1, y2+h2) - yi, 0)
    inter = wi*hi
    union = w1*h1 + w2*h2 - inter
    return inter/(union+1e-6)

@torch.no_grad()
def fmc_one_image(img_path, bbox_list, label_list, alpha=0.55, iou_thr=0.05):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    corrected=[]
    for (x,y,w,h),label in zip(bbox_list, label_list):
        box_input = np.array([x,y,x+w,y+h])
        masks, scores, _ = predictor.predict(box=box_input, multimask_output=True)
        cx, cy = x+w/2, y+h/2
        masks_pt, scores_pt, _ = predictor.predict(point_coords=np.array([[cx,cy]]), point_labels=np.array([1]), multimask_output=True)
        masks = np.concatenate([masks, masks_pt])
        scores = np.concatenate([scores, scores_pt])

        text = clip.tokenize([label]).to(DEVICE)
        text_feat = clip_m.encode_text(text)
        clip_scores=[]
        for m in masks:
            ys, xs = np.where(m)
            if len(xs)==0: clip_scores.append(0); continue
            x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
            crop = image[y_min:y_max, x_min:x_max]
            if crop.size==0: clip_scores.append(0); continue
            crop = cv2.resize(crop, (224,224))
            img_tensor = preprocess(Image.fromarray(crop)).unsqueeze(0).to(DEVICE)
            img_feat = clip_m.encode_image(img_tensor)
            clip_scores.append(torch.cosine_similarity(img_feat, text_feat).item())
        clip_scores = np.array(clip_scores)
        final_scores = alpha*np.array(clip_scores) + (1-alpha)*scores
        best_mask = masks[np.argmax(final_scores)]
        ys, xs = np.where(best_mask)
        if len(xs)==0:
            corrected.append([x,y,w,h]); continue
        x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
        new_box = [x_min, y_min, x_max-x_min, y_max-y_min]
        if compute_iou([x,y,w,h], new_box) > iou_thr:
            corrected.append(new_box)
        else:
            corrected.append([x,y,w,h])
    return corrected