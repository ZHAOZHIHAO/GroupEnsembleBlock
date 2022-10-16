import torchvision.datasets as datasets
import torchvision.transforms as T
from datasets.autoaugment import ImageNetPolicy
import random
from PIL import ImageFilter

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
print("Use AutoAugment, RandomCrop")


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


transform_ssl_imagenet = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.)),
    ImageNetPolicy(),
    T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    normalize,
    T.RandomErasing(value='random'),
])
print("With autoaugment")


class ImageFolderInstanceGet2(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """

    def __init__(self, root, transform, target_transform=None, coarse=True):
        super(ImageFolderInstanceGet2, self).__init__(root, transform, target_transform=target_transform)
        self.coarse = coarse

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        img1 = transform_ssl_imagenet(img)
        img2 = transform_ssl_imagenet(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target, index


class ImageFolderInstanceTest(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """

    def __init__(self, root, transform, target_transform=None, coarse=True):
        super(ImageFolderInstanceTest, self).__init__(root, transform, target_transform=target_transform)
        self.coarse = coarse

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
