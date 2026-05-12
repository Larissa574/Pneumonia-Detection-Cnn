"""
Application Gradio pour la détection de pneumonie par radiographie thoracique.
Combine prédiction multiclasse + explainabilité Grad-CAM.

Classes:
  - BACTERIA: Pneumonie bactérienne
  - NORMAL: Poumon sain
  - VIRUS: Pneumonie virale

⚠️ USAGE: Outil d'aide au diagnostic uniquement, pas de substitut médical.
"""

import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ============ CHARGEMENT DU MODÈLE ============
model = tf.keras.models.load_model('/kaggle/working/best_model.keras')
_ = model(tf.zeros((1, 224, 224, 3)), training=False)  # Initialiser

class_names = ['BACTERIA', 'NORMAL', 'VIRUS']

# ============ FONCTION GRAD-CAM ============
def make_gradcam_heatmap(img_array):
    """Calcule la heatmap Grad-CAM pour l'explication"""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer('efficientnetb0').get_layer('top_conv').output,
            model.outputs[0]
        ]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        pred_class = tf.argmax(predictions[0])
        class_score = predictions[:, pred_class]
    
    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy(), pred_class.numpy(), predictions[0].numpy()

# ============ PIPELINE DE PRÉDICTION ============
def predict(image):
    """Pipeline complète: prédiction + Grad-CAM + probas"""
    
    if image is None:
        return None, None

    # Redimensionner à 224x224
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # Gérer les canaux
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # Préparer pour le modèle
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.cast(img_array, tf.float32)

    # Prédictions + Grad-CAM
    heatmap, pred_idx, probs = make_gradcam_heatmap(img_array)

    # Superposer Grad-CAM
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_np = np.array(img.resize((224, 224)))
    superimposed = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
    superimposed = Image.fromarray(superimposed)

    # Probabilités par classe
    label = {class_names[i]: float(probs[i]) for i in range(3)}
    
    return superimposed, label

# ============ INTERFACE GRADIO ============
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="📤 Télécharger une radiographie thoracique"),
    outputs=[
        gr.Image(type="pil", label="🔍 Grad-CAM - Zones pertinentes"),
        gr.Label(num_top_classes=3, label="📊 Probabilités par classe")
    ],
    title="🫁 Détecteur de Pneumonie - Classification multiclasse",
    description="""
    **EfficientNetB0 Transfer Learning** - Entraîné sur ~5,200 radiographies thoraciques.
    
    **Classes détectées:**
    - 🟢 NORMAL: Poumon sain
    - 🔴 BACTERIA: Pneumonie bactérienne (traitement antibiotique)
    - 🟠 VIRUS: Pneumonie virale (traitement antiviral)
    
    **Explication:**
    - La heatmap Grad-CAM montre les zones que le modèle utilise pour décider
    - Rouge = haute attention | Bleu = faible attention
    
    ⚠️ **DISCLAIMER:** Outil d'aide au diagnostic UNIQUEMENT. 
    Non un substitut au diagnostic médical professionnel.
    Consulter un radiologue/médecin pour confirmation.
    """,
    examples=[]
)

if __name__ == "__main__":
    demo.launch(share=True)
