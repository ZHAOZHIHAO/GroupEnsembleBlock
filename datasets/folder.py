import torchvision.datasets as datasets
import torchvision.transforms as T
from datasets.autoaugment import CIFAR10Policy


print("Use AutoAugment, RandomCrop")
transform_ssl = T.Compose([
            T.RandomCrop(32, padding=4, fill=128),
            T.RandomHorizontalFlip(),
            CIFAR10Policy(),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            T.RandomErasing(value='random'),
        ])
        

class ImageFolderInstanceGet2(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
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
            img1 = transform_ssl(img)
            img2 = transform_ssl(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target, index
    
    
class ImageFolderInstanceGet4(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
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
            img1 = transform_ssl(img)
            img2 = transform_ssl(img)
            img3 = transform_ssl(img)
            img4 = transform_ssl(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, img3, img4, target, index
        
        
class ImageFolderInstanceTest(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
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
