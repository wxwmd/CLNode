import torch


class EarlyStop():
    def __init__(self, patience,path):
        self.score = 0.
        self.bad_count = 0
        self.patience = patience
        self.path=path

    def step(self, score,model):
        # 取得最好效果，保存一下
        if score > self.score:
            torch.save(model,self.path)
            self.score = score
            self.bad_count = 0
        else:
            self.bad_count += 1
            if self.bad_count > self.patience:
                return False
        return True
