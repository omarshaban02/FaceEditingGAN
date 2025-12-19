### **1. Technical Report: GAN-Based Facial Attribute Manipulation**

#### **Section 1: Introduction**

* **Objective:** The project aims to develop a robust pipeline for high-fidelity facial attribute manipulation, specifically targeting hair style/color, age, facial expressions, and head pose while preserving the subject's identity.
* **Approach:** I implemented a multi-model architecture centered around **StyleGAN2-ADA** (pre-trained on FFHQ) to leverage its superior generative capabilities in the face domain.
* **Key Challenge:** A primary challenge in facial editing is the trade-off between editability and identity preservation; this was addressed by utilizing advanced inversion and 3D-aware techniques.

#### **Section 2: Methodology**

* **2.1 Image Inversion (e4e):** Real images are mapped into the  latent space using the **encoder4editing (e4e)** framework. This specific encoder was chosen because it enforces the "editability" of the latent code, ensuring that subsequent manipulations do not introduce artifacts common in standard pSp encoders.
* **2.2 Attribute Manipulation Strategy:**
* **Hair & Expression (InterfaceGAN):** I utilized **InterfaceGAN** to find semantically meaningful boundaries (hyperplanes) in the latent space. By moving the latent code along the normal vector of these boundaries (e.g., 'smile' or 'hair length'), we achieve linear control over attributes.
* **Age Progression (SAM):** To ensure realistic aging (wrinkles, skin texture) rather than just "blurring" the face, I integrated the **Style and Age Manipulator (SAM)**. This model uses a dedicated aging transformer to modify the StyleGAN latents based on a continuous age input (0-100 years).
* **3.D-Aware Pose (EG3D):** Traditional GANs struggle with large pose changes. I utilized **EG3D**, which incorporates a Tri-plane representation and a volume renderer, allowing for geometrically consistent head rotations (yaw, pitch, roll).


* **2.3 Identity Preservation:** To maintain the person's likeness during extreme edits, I implemented an **Identity Loss** module using **LPIPS (Perceptual Loss)** and **L2 pixel-wise loss**. This ensures the modified latent remains within the perceptual neighborhood of the original subject.

---

### **2. How to Reproduce**

#### **Project Overview**

This repository provides a complete pipeline for professional-grade facial manipulation using a suite of state-of-the-art GAN models (StyleGAN2, e4e, SAM, EG3D, and InterfaceGAN).

#### **Installation**

1. **Environment:** The code is optimized for **Google Colab (T4 GPU)**.
2. **Dependencies:** Run the first cell to install core packages (`ninja`, `lpips`, `clip`) and clone the required repositories.
3. **Models:** Pre-trained weights for StyleGAN2 (FFHQ), e4e encoder, SAM aging, and EG3D are automatically downloaded to the `/content/models` directory.

#### **Step-by-Step Execution**

1. **Inversion:** Upload a face image (256x256 or 1024x1024) and run the `e4e_inverter.invert(image_path)` method to extract the  latent code.
2. **Facial Expression/Hair:** Use the `InterfaceGANManipulator` class. Use `manipulate(latent, 'smile', strength=X)` to change expression or hair length.
3. **Realistic Aging:** Call `sam_manipulator.change_age(image_path, target_age=60)` to transform the subject's age.
4. **3D Pose:** Use `eg3d_manipulator.change_pose(latent, yaw=0.5)` to rotate the head while maintaining 3D structure.
