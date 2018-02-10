# CNN Model Libraries
Currently, KMILT supports following CNN networks
1. Densenet121

## Densenet121

```python
def densenet121(nb_dense_block=4, growth_rate=16, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
                weights_path=None, image_dimension=512, color_mode='grayscale'):
```

### Fully connected layer features
Densenet121 in this package has two variants in its FC layer:
1. Multiclass
2. Multibinary

Multiclass is the original architecture proposed in densenet121 paper. It uses 1024 neuron MLP to implement classification task. `Softmax` activation is applied to generate probability estimates. `argmax` is assumed to obtain **single** final prediction label.   

Multibinary is the other way to do multiclass classification which supports **multi label** predictions. Each class is independently binary classified (positive/negative) and activated by `sigmoid` to obtain probability estimates. Do not apply `argmax` when multibinary mode is enabled.      
## Networks planned to be supported