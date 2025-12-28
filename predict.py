import io
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ResNet import ResNet18

# =====================
# 1. 基本配置
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoint/resnet18_fashionmnist.pt"

class_names = [
    "T-shirt / Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# =====================
# 2. 更稳健的图像预处理（修复“白色长方块”问题）
# =====================
def preprocess_fashion_image(
    image_path,
    fit_size=24,
    out_size=28,
    mean=0.2860,
    std=0.3530,
    margin=6,
    debug=False
):
    """
    适用于：白底/灰底商品图 + 真实图片推理到 FashionMNIST 模型
    核心思想：
      - 用边缘像素估计背景亮度
      - 用 absdiff 与背景差异做分割（不依赖前景是黑还是白）
      - 面积保护 + bbox 保护，避免裁剪失败导致 28x28 变成白块
      - 自动反相，让前景更亮、背景更暗，更贴近 FashionMNIST 分布
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"图片路径错误或无法读取: {image_path}")

    H, W = img.shape

    # 0) 轻微去噪（防止压缩噪声影响分割）
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    # 1) 用边缘估计背景（对“白底/灰底/渐变底”更稳）
    border = np.concatenate([img_blur[0, :], img_blur[-1, :], img_blur[:, 0], img_blur[:, -1]])
    bg = np.median(border).astype(np.uint8)

    # 2) 与背景差异图：前景无论亮/暗都能分出来
    diff = cv2.absdiff(img_blur, np.full_like(img_blur, bg))

    # 自适应阈值：用 diff 的分位数决定阈值，避免把整张图当作前景
    t = int(max(10, np.percentile(diff, 90) * 0.35))
    _, mask = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)

    # 3) 形态学：去小噪声 + 填洞
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4) 面积保护：mask 太大/太小就 fallback 到 Otsu（否则 bbox 会离谱）
    area_ratio = mask.mean() / 255.0  # 0~1
    if area_ratio < 0.005 or area_ratio > 0.85:
        _, mask2 = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 选择“更像前景”的那一侧：比较中心区域 vs 边缘亮度
        center = img_blur[H // 4: 3 * H // 4, W // 4: 3 * W // 4].mean()
        edge = border.mean()
        if center > edge:
            mask = mask2
        else:
            mask = 255 - mask2

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5) 通过 mask 计算 bbox（比轮廓更稳定）
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1

        # 留边
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(W, x1 + margin)
        y1 = min(H, y1 + margin)

        # bbox 保护：太大说明分割把整张图当作前景 -> 不裁剪
        bbox_ratio = ((x1 - x0) * (y1 - y0)) / (H * W)
        if bbox_ratio > 0.95:
            x0, y0, x1, y1 = 0, 0, W, H

    cropped = img[y0:y1, x0:x1]
    mask_c = mask[y0:y1, x0:x1]

    # 6) 自动反相：让前景更亮、背景更暗（更像 FashionMNIST）
    if (mask_c > 0).any() and (mask_c == 0).any():
        fg_mean = cropped[mask_c > 0].mean()
        bg_mean = cropped[mask_c == 0].mean()
        if fg_mean < bg_mean:
            cropped = 255 - cropped

    # 7) 等比缩放到 fit_size，再 pad 到 28x28
    h, w = cropped.shape
    scale = fit_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = (out_size - new_w) // 2
    pad_h = (out_size - new_h) // 2
    img_28 = cv2.copyMakeBorder(
        resized,
        pad_h, out_size - new_h - pad_h,
        pad_w, out_size - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # 8) 轻微平滑，让缩放更像 FashionMNIST（可选但通常有帮助）
    img_28 = cv2.GaussianBlur(img_28, (3, 3), 0)

    # 9) Normalize（必须与训练一致）
    img_norm = img_28.astype(np.float32) / 255.0
    img_norm = (img_norm - mean) / std
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)

    if debug:
        return tensor, img_28, mask
    return tensor, img_28


# =====================
# 3. 加载模型（只加载一次 + checkpoint 更健壮）
# =====================
def load_model():
    model = ResNet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(512, 10)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)

    model.to(device).eval()
    return model


# =====================
# 4. 预测与可视化（模型作为参数传入，不重复加载）
# =====================
def predict_and_visualize(model, image_path, topk=5):
    input_tensor, _ = preprocess_fashion_image(image_path, debug=False)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    topk = max(1, min(int(topk), 10))
    top_indices = np.argsort(probs)[::-1][:topk]

    top1_idx = top_indices[0]
    top1_label = class_names[top1_idx]
    top1_prob = probs[top1_idx]

    original = Image.open(image_path)

    # ===== 布局：2 列 =====
    plt.figure(figsize=(16, 6))

    # ===== 左：原始图片 =====
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Input Image", fontsize=13)
    plt.axis("off")

    # ===== 右：文字说明 + 预测 =====
    plt.subplot(1, 2, 2)

    labels = [class_names[i] for i in top_indices]
    values = [probs[i] for i in top_indices]

    bars = plt.barh(labels[::-1], values[::-1])
    plt.xlim(0, 1.0)
    plt.xlabel("Probability")

    # 高亮 Top-1
    bars[-1].set_alpha(1.0)
    for b in bars[:-1]:
        b.set_alpha(0.5)

    # 概率标注
    for bar in bars:
        w = bar.get_width()
        plt.text(
            w + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.2f}",
            va="center",
            fontsize=10
        )

    # ===== 专业说明文字 =====
    info_text = (
        "Model Information\n"
        "────────────────────────\n"
        "Architecture : ResNet18\n"
        "Dataset      : FashionMNIST\n"
        "Input Format : 28×28 Grayscale\n"
        "Preprocessing:\n"
        "  • Background-aware cropping\n"
        "  • Aspect-ratio preserving resize\n"
        "  • Zero-padding to 28×28\n"
        "  • Normalization (train-consistent)\n\n"
        "Prediction Result\n"
        "────────────────────────\n"
        f"Top-1 Class : {top1_label}\n"
        f"Confidence  : {top1_prob:.2%}"
    )

    plt.text(
        1.05, 0.5, info_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        va="center",
        family="monospace"
    )

    plt.title(
        f"Top-{topk} Classification Results",
        fontsize=13
    )

    plt.tight_layout()
    plt.show()

    return [(class_names[i], float(probs[i])) for i in top_indices]



# =====================
# 5. 主入口
# =====================
if __name__ == "__main__":
    image_path = r"D:\desktop\ResNet50\targetPIC.png"

    try:
        model = load_model()
        result = predict_and_visualize(model, image_path, topk=5)
        print("Top-k:", result)
    except Exception as e:
        print(f"发生错误: {e}")