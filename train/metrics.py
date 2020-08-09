from torch.nn.modules.loss import _Loss

class PSNR(_Loss):
    def __init__(self):
        super(PSNR, self).__init__()
        self.rgb = 255

    def _quantize(self, x):
        return x.clamp(0, self.rgb).round()

    def forward(self, x, y):
        diff = self._quantize(x) - y

        if x.dim() == 3:
            n = 1
        elif x.dim() == 4:
            n = x.size(0)
        elif x.dim() == 5:
            n = x.size(0) * x.size(1)

        mse = diff.div(self.rgb).pow(2).view(n, -1).mean(dim=-1)
        psnr = -10 * mse.log10()

        return psnr.mean()