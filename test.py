if __name__ == '__main__':
    import torch
    B = 14  
    H = 48  
    W = 48  
    frames = torch.rand(1, B, 4, H, W).cuda()
    model = Base_Model().cuda()
    from pytorch_lightning.utilities.model_summary import summarize
    summary = summarize(model)
    print(summary)
    model.eval()
    # image = model(frames)
    # print('image.shape:',image.shape) 
    # from fvcore.nn import FlopCountAnalysis, parameter_count
    # flops = FlopCountAnalysis(model, frames)
    # print("FLOPs:(G) ", flops.total()/(10**9))
    total_params = sum(p.numel() for p in model.parameters())
    print("Total Parameters:(M)", total_params/(10**6))
