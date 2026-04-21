# Backbone Readiness Report

This report checks local pretrained weights only. It does not download models.

| Backbone | Runnable Now | Code Support | Source Type | Local Path | Notes |
|---|---:|---:|---|---|---|
| `resnet18` | `True` | `True` | torchvision_cache | `/Users/ritik/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` | Reference backbone; already used for the first D1 anomaly run. |
| `resnet50` | `True` | `True` | torchvision_cache | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/.torch/hub/checkpoints/resnet50-11ad3fa6.pth` | Larger ResNet; may improve semantic features but still uses global pooling unless patch mode is enabled. |
| `efficientnet_b0` | `False` | `False` | missing_local_weights | `` | Potential compact CNN backbone; pending local weights and extractor support. Local weights not found. |
| `convnext_tiny` | `False` | `False` | missing_local_weights | `` | Modern CNN family; pending local weights and extractor support. Local weights not found. |
| `vit_b_16` | `False` | `False` | missing_local_weights | `` | Transformer backbone; pending local weights and extractor support. Local weights not found. |

## Cached Weight Files

- `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/.torch/hub/checkpoints/resnet50-11ad3fa6.pth`
- `/Users/ritik/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`
