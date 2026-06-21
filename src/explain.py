import numpy as np
import tensorflow as tf
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    for sub in reversed(layer.layers):
                        if isinstance(sub, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                            last_conv_layer_name = sub.name
                            break
                if last_conv_layer_name:
                    break

    backbone = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            backbone = layer
            break

    if backbone is not None:
        try:
            conv_layer = backbone.get_layer(last_conv_layer_name)
            grad_model = tf.keras.Model(inputs=backbone.input, outputs=[conv_layer.output, backbone.output])
            
            with tf.GradientTape() as tape:
                img_tensor = tf.cast(img_array, tf.float32)
                if backbone.input.shape[1] != img_array.shape[1]:
                    img_resized = tf.image.resize(img_tensor, [backbone.input.shape[1], backbone.input.shape[2]])
                else:
                    img_resized = img_tensor
                conv_outputs, predictions = grad_model(img_resized)
                loss = predictions[:, 0]
        except Exception:
            return None
    else:
        return None

    grads = tape.gradient(loss, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()