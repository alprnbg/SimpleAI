import torch

def iou_0(output, target):
    output = output[:,0,:,:]
    target = target["mask"][:,0,:,:]
    return iou(output, target)

def iou_1(output, target):
    output = output[:,1,:,:]
    target = target["mask"][:,1,:,:]
    return iou(output, target)

def iou(output, target):
    if isinstance(target, dict):
        target = target["mask"]
    smooth = 1e-5
    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.cpu().numpy()
    output_ = (output > 0.5).reshape(output.shape[0],-1)
    target_ = (target > 0.5).reshape(output.shape[0],-1)
    intersection = (output_ & target_).sum(axis=1)
    union = (output_ | target_).sum(axis=1)
    return ((intersection + smooth) / (union + smooth)).mean()


class Metrics:
    def __init__(self):
        self.train_metrics = {"iou":iou, "iou_class0": iou_0, "iou_class1": iou_1}
        self.val_metrics = {"iou":iou, "iou_class0": iou_0, "iou_class1": iou_1}
    
    
def get_metrics(config):
    return Metrics()
    
    
    
    



