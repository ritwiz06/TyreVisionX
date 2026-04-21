# Corruption Benchmark

Primary rule: use the clean validation-selected threshold for corrupted validation/test evaluation. No corrupted test retuning.

| Variant | Corruption | Recall | Precision | FN | FP | AUROC | AUPRC | Recall Gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `resnet50_knn` | clean | 0.8231 | 0.8770 | 23 | 15 | 0.9298 | 0.9339 | 0.0000 |
| `resnet50_knn` | gaussian_noise_low | 0.8077 | 0.8537 | 25 | 18 | 0.9053 | 0.8962 | 0.0154 |
| `resnet50_knn` | gaussian_noise_medium | 0.8538 | 0.8162 | 19 | 25 | 0.9021 | 0.8890 | -0.0308 |
| `resnet50_knn` | gaussian_blur_low | 0.8154 | 0.8760 | 24 | 15 | 0.9306 | 0.9329 | 0.0077 |
| `resnet50_knn` | gaussian_blur_medium | 0.7846 | 0.8793 | 28 | 14 | 0.9242 | 0.9256 | 0.0385 |
| `resnet50_knn` | jpeg_compression_mild | 0.8000 | 0.8667 | 26 | 16 | 0.9264 | 0.9250 | 0.0231 |
| `resnet50_knn` | brightness_darker | 0.7846 | 0.8571 | 28 | 17 | 0.9241 | 0.9290 | 0.0385 |
| `resnet50_knn` | brightness_brighter | 0.8308 | 0.8571 | 22 | 18 | 0.9226 | 0.9225 | -0.0077 |
| `resnet50_knn` | contrast_lower | 0.8000 | 0.8739 | 26 | 15 | 0.9226 | 0.9270 | 0.0231 |
| `resnet50_knn` | contrast_higher | 0.8154 | 0.8760 | 24 | 15 | 0.9241 | 0.9267 | 0.0077 |
| `resnet50_mahalanobis` | clean | 0.7077 | 0.8214 | 38 | 20 | 0.8676 | 0.8800 | 0.0000 |
| `resnet50_mahalanobis` | gaussian_noise_low | 0.7154 | 0.8304 | 37 | 19 | 0.8510 | 0.8346 | -0.0077 |
| `resnet50_mahalanobis` | gaussian_noise_medium | 0.8077 | 0.7895 | 25 | 28 | 0.8578 | 0.8307 | -0.1000 |
| `resnet50_mahalanobis` | gaussian_blur_low | 0.7462 | 0.8291 | 33 | 20 | 0.8722 | 0.8814 | -0.0385 |
| `resnet50_mahalanobis` | gaussian_blur_medium | 0.7308 | 0.8333 | 35 | 19 | 0.8698 | 0.8760 | -0.0231 |
| `resnet50_mahalanobis` | jpeg_compression_mild | 0.7077 | 0.8288 | 38 | 19 | 0.8656 | 0.8744 | 0.0000 |
| `resnet50_mahalanobis` | brightness_darker | 0.7154 | 0.8378 | 37 | 18 | 0.8683 | 0.8799 | -0.0077 |
| `resnet50_mahalanobis` | brightness_brighter | 0.7385 | 0.8136 | 34 | 22 | 0.8548 | 0.8645 | -0.0308 |
| `resnet50_mahalanobis` | contrast_lower | 0.7154 | 0.8158 | 37 | 21 | 0.8588 | 0.8710 | -0.0077 |
| `resnet50_mahalanobis` | contrast_higher | 0.7154 | 0.8304 | 37 | 19 | 0.8628 | 0.8729 | -0.0077 |
| `resnet18_knn_control` | clean | 0.5231 | 0.8831 | 62 | 9 | 0.8546 | 0.8394 | 0.0000 |
| `resnet18_knn_control` | gaussian_noise_low | 0.5538 | 0.9000 | 58 | 8 | 0.8636 | 0.8536 | -0.0308 |
| `resnet18_knn_control` | gaussian_noise_medium | 0.6154 | 0.8889 | 50 | 10 | 0.8658 | 0.8648 | -0.0923 |
| `resnet18_knn_control` | gaussian_blur_low | 0.5231 | 0.8500 | 62 | 12 | 0.8477 | 0.8320 | 0.0000 |
| `resnet18_knn_control` | gaussian_blur_medium | 0.5231 | 0.8500 | 62 | 12 | 0.8407 | 0.8272 | 0.0000 |
| `resnet18_knn_control` | jpeg_compression_mild | 0.5154 | 0.8701 | 63 | 10 | 0.8587 | 0.8428 | 0.0077 |
| `resnet18_knn_control` | brightness_darker | 0.4923 | 0.8421 | 66 | 12 | 0.8407 | 0.8238 | 0.0308 |
| `resnet18_knn_control` | brightness_brighter | 0.5846 | 0.8941 | 54 | 9 | 0.8677 | 0.8548 | -0.0615 |
| `resnet18_knn_control` | contrast_lower | 0.5231 | 0.8608 | 62 | 11 | 0.8510 | 0.8337 | 0.0000 |
| `resnet18_knn_control` | contrast_higher | 0.5385 | 0.8642 | 60 | 11 | 0.8542 | 0.8422 | -0.0154 |
| `resnet50_knn_noise_robust` | clean | 0.8231 | 0.8917 | 23 | 13 | 0.9407 | 0.9383 | 0.0000 |
| `resnet50_knn_noise_robust` | gaussian_noise_low | 0.8000 | 0.9123 | 26 | 10 | 0.9343 | 0.9247 | 0.0231 |
| `resnet50_knn_noise_robust` | gaussian_noise_medium | 0.8308 | 0.8926 | 22 | 13 | 0.9252 | 0.9062 | -0.0077 |
| `resnet50_knn_noise_robust` | gaussian_blur_low | 0.8231 | 0.9068 | 23 | 11 | 0.9434 | 0.9421 | 0.0000 |
| `resnet50_knn_noise_robust` | gaussian_blur_medium | 0.8231 | 0.9145 | 23 | 10 | 0.9444 | 0.9444 | 0.0000 |
| `resnet50_knn_noise_robust` | jpeg_compression_mild | 0.7923 | 0.9196 | 27 | 9 | 0.9364 | 0.9283 | 0.0308 |
| `resnet50_knn_noise_robust` | brightness_darker | 0.8231 | 0.8992 | 23 | 12 | 0.9372 | 0.9355 | 0.0000 |
| `resnet50_knn_noise_robust` | brightness_brighter | 0.8308 | 0.8926 | 22 | 13 | 0.9324 | 0.9239 | -0.0077 |
| `resnet50_knn_noise_robust` | contrast_lower | 0.8231 | 0.8992 | 23 | 12 | 0.9340 | 0.9298 | 0.0000 |
| `resnet50_knn_noise_robust` | contrast_higher | 0.8077 | 0.8898 | 25 | 13 | 0.9373 | 0.9324 | 0.0154 |
