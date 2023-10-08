def model_statistics(model):
    total_params = sum(param.numel() for param in model.parameters()) / 1000000.0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0
    print('    Total params: %.2fM, trainable params: %.2fM' % (total_params, trainable_params))