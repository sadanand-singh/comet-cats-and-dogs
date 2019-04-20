from torchvision import datasets, transforms
from base import BaseDataLoader


class CatsAndDogsDataLoader(BaseDataLoader):
    def __init__(
        self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=4, train=True
    ):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            tfs = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            tfs = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
            )

        self.dataset = datasets.ImageFolder(data_dir, tfs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
