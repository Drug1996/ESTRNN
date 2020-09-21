from torch.nn.modules.loss import _Loss
# from skimage.measure import compare_ssim

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

# class SSIM:
#     def __init__(self):
#         self.sum = 0.0
#         self.count = 0.0
#         self.avg = 0.0
#
#     def __call__(self, x, y):
#         if x.dim() == 3:
#             n = 1
#         elif x.dim() == 4:
#             n = x.size(0)
#         elif x.dim() == 5:
#             n = x.size(0) * x.size(1)
#         x = x.view(n,-1).detach().cpu().numpy()
#         y = y.view(n,-1).detach().cpu().numpy()
#         for i in range(n):
#             img_gt = y[i]
#             img_deblur = x[i]
#             self.sum += compare_ssim(img_gt, img_deblur, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
#         self.count += n
#         self.avg = self.sum/self.count
#
#         return self.avg