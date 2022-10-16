from torchvision import transforms


def get_cifar100_train_test_set(args):
    from datasets.folder import  ImageFolderInstanceTest, ImageFolderInstanceGet4
    #https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/data.py
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if args.TRAIN:
        trainset = ImageFolderInstanceGet4(root='./data/cifar100_coarse/train', transform=transform_train)
        # this is the validation set used in training stage, still only coarse labels available
        testset = ImageFolderInstanceTest(root='./data/cifar100_coarse/test', transform=transform_test)

    return trainset, testset


def get_ImageNetC16_train_test_set(args):
    from datasets.folder_imagenet import ImageFolderInstanceTest, ImageFolderInstanceGet2
    # override by the transform_ssl_imagenet in folder_imagenet.py
    transform_train = transforms.Compose([])
    transform_imagenet_test = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    if args.TRAIN:
        trainset = ImageFolderInstanceGet2(root='./data/imagenetC16_coarse/train', transform=transform_train)
        testset = ImageFolderInstanceTest(root='./data/imagenetC16_coarse/test', transform=transform_imagenet_test)

    return trainset, testset
