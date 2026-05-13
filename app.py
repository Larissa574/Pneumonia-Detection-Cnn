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
model = tf.keras.models.load_model('best_model (2).keras')
_ = model(tf.zeros((1, 224, 224, 3)), training=False)  # Initialiser

class_names = ['BACTERIA', 'NORMAL', 'VIRUS']

# ============ FONCTION GRAD-CAM ============
def make_gradcam_heatmap(img_array):
    """Calcule la heatmap Grad-CAM pour l'explication"""
    base_model = model.get_layer('efficientnetb0')
    
    # Créer deux modèles: un pour les feature maps, un pour les prédictions
    # Model 1: from input to last conv layer
    last_conv_layer = base_model.get_layer('top_activation')
    
    # Créer un nouveau modèle: input -> last_conv output
    conv_output_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        base_model,
    ])
    # Recréer avec les vraies couches
    x_input = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(x_input)
    # Model pour extraire les features ET prédictions finales
    full_output = model(x_input)
    
    # Modèle complet qui retourne feature maps + prédiction
    feature_extractor = tf.keras.Model(
        inputs=x_input,
        outputs=[last_conv_layer.output, full_output]
    )
    
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        feature_maps, predictions = feature_extractor(img_array, training=False)
        pred_idx = tf.argmax(predictions[0])
        class_score = predictions[:, pred_idx]
    
    # Calculer les gradients
    grads = tape.gradient(class_score, feature_maps)
    
    # Global average pooling des gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Pondérer les feature maps
    heatmap = tf.reduce_sum(feature_maps[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
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
    demo.launch(share=True)
