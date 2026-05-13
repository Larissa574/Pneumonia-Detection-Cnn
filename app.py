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
import os
import traceback

# ============ CHARGEMENT DU MODÈLE ============
model = None
try:
    # Try explicit name then fallback to any .keras file in the repo
    candidate = 'best_model (2).keras'
    if not os.path.exists(candidate):
        keras_files = [f for f in os.listdir('.') if f.endswith('.keras')]
        print('Workspace files:', os.listdir('.'))
        print('Detected .keras files:', keras_files)
        if keras_files:
            candidate = keras_files[0]
    print('Loading model from:', candidate)
    model = tf.keras.models.load_model(candidate)
    # Warmup once
    _ = model(tf.zeros((1, 224, 224, 3)), training=False)
    print('Model loaded successfully')
except Exception as e:
    print('❌ Model load failed:', e)
    traceback.print_exc()
    model = None

class_names = ['BACTERIA', 'NORMAL', 'VIRUS']

# ============ FONCTION GRAD-CAM ============
def make_gradcam_heatmap(img_array):
    """Calcule la heatmap Grad-CAM pour l'explication"""
    base_model = model.get_layer('efficientnetb0')
    last_conv_layer = base_model.get_layer('top_activation')
    
    # Créer un modèle intermédiaire pour extraire les feature maps
    intermediate_layer_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )
    
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        # Get feature maps from conv layer
        conv_outputs = intermediate_layer_model(img_array)
        # Get final predictions
        predictions = model(img_array, training=False)
        pred_idx = tf.argmax(predictions[0])
        class_score = predictions[:, pred_idx]
    
    # Calculer les gradients
    grads = tape.gradient(class_score, conv_outputs)
    
    if grads is None:
        # Fallback if gradient computation fails
        return np.ones((7, 7)), int(pred_idx.numpy()), predictions[0].numpy()
    
    # Global average pooling des gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Pondérer les feature maps
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap[0], 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy(), int(pred_idx.numpy()), predictions[0].numpy()

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
    img_array = tf.cast(img_array, tf.float32) / 255.0

    # Prédictions + Grad-CAM
    heatmap, pred_idx, probs = make_gradcam_heatmap(img_array)

    # Superposer Grad-CAM
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Image originale sans normalisation pour l'overlay
    img_np = np.array(image.resize((224, 224)))
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    if img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
    
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
    # On Spaces, avoid using share=True and disable the threaded reloader
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), show_error=False, prevent_threaded_reload=True)
