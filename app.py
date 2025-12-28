import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from ResNet import ResNet18
from predict import preprocess_fashion_image  # 复用你已有的预处理

# =====================
# 基本配置
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoint/resnet18_fashionmnist.pt"

class_names = [
    "T-shirt / Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# =====================
# 加载模型（只加载一次）
# =====================
@st.cache_resource
def load_model():
    model = ResNet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = nn.Linear(512, 10)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)

    model.to(device).eval()
    return model


# =====================
# 页面标题
# =====================
st.set_page_config(page_title="FashionMNIST Clothing Recognition", layout="wide")

st.title("👕 FashionMNIST Clothing Recognition System")
st.markdown(
    """
    This demo system performs **clothing image classification**
    using a **ResNet18 model trained on the FashionMNIST dataset**.
    """
)

st.divider()

# =====================
# 上传图片
# =====================
uploaded_file = st.file_uploader(
    "📤 Upload a clothing image",
    type=["jpg", "png", "jpeg"]
)

# =====================
# 预测按钮
# =====================
predict_button = st.button("🚀 Start Prediction")

# =====================
# 主逻辑
# =====================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        st.image(image, use_container_width=True)

    if predict_button:
        model = load_model()

        # 保存临时文件供 OpenCV 读取
        temp_path = "temp_input.png"
        image.save(temp_path)

        input_tensor, _ = preprocess_fashion_image(temp_path)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        top_indices = np.argsort(probs)[::-1][:5]
        top1_idx = top_indices[0]

        # =====================
        # 右侧：预测结果
        # =====================
        with col2:
            st.subheader("📊 Prediction Results")

            st.markdown(
                f"""
                **Model:** ResNet18  
                **Dataset:** FashionMNIST  
                **Input Format:** 28×28 Grayscale  

                ### 🏷️ Final Prediction
                **{class_names[top1_idx]}**  
                Confidence: **{probs[top1_idx] * 100:.2f}%**
                """
            )

            # ===== 概率条形图 =====
            fig, ax = plt.subplots()
            labels = [class_names[i] for i in top_indices]
            values = [probs[i] for i in top_indices]

            ax.barh(labels[::-1], values[::-1])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_title("Top-5 Classification Probabilities")

            for i, v in enumerate(values[::-1]):
                ax.text(v + 0.01, i, f"{v:.2f}", va="center")

            st.pyplot(fig)

else:
    st.info("⬆️ Please upload an image to start prediction.")
