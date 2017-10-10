from torch.optim.lr_scheduler import LambdaLR

def get_polyscheduler(optimizer, lr_power, total_epochs):
    f = lambda epoch : (1. - float(epoch)/total_epochs) ** lr_power
    return LambdaLR(optimizer, lr_lambda=f)
