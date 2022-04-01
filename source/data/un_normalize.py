class UnNormalize(object):
    """ Reverse transform a normalizion"""
    def __init__(self, IMG_MEAN, IMG_STD):
        """IMG_MEAN,IMG_STD must be a list (or other iterable)"""
        self.mean = IMG_MEAN
        self.std = IMG_STD

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor