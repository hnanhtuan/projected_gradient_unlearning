import torch
import random
import numpy as np
from PIL import Image
from typing import Any, Callable, Optional, Tuple, List
from torchvision.datasets import CIFAR10, CIFAR100
import torch.utils.data as data


class PoisonCIFAR10(CIFAR10):
    def __init__(self, root: str,
                 data_set: str = 'train',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 random_seed: int = 0,
                 num_poison: int = 0,  # number of poison samples per class
                 data_section: str = 'full',  
                 ):
        # 'full': both poisoned and clean data
        # 'clean': clean data only
        # 'poison': poison data only
        # 'poison_clean': poison data with clean labels
        
        super().__init__(root, data_set=='train', transform, target_transform, download)
        
        self.data_set = data_set
        self.data_section = data_section
        self.num_poison = num_poison
        
        self.targets = np.array(self.targets)
        num_classes = len(set(self.targets))
        random.seed(random_seed)

        if self.data_set != 'train':
            set_idx = []
            for i in range(num_classes):
                idx = np.where(self.targets == i)[0].tolist()
                random.shuffle(idx)
                set_idx.extend(idx[:len(idx)//2] if self.data_set == 'test' else idx[len(idx)//2:])

            self.data = self.data[set_idx, :, :, :]
            self.targets = self.targets[set_idx]

        self.classes = np.arange(0, num_classes, 1).tolist()
        random.shuffle(self.classes)
        self.poison_idx = []
        self.original_targets = np.array(self.targets)
        
        for i in range(num_classes//2):
            c1_idx = np.where(self.targets == self.classes[2 * i])[0]
            c2_idx = np.where(self.targets == self.classes[2 * i + 1])[0]
            random.shuffle(c1_idx)
            random.shuffle(c2_idx)
            for j in range(num_poison):
                self.targets[c1_idx[j]] = self.classes[2 * i + 1]
                self.poison_idx.append(c1_idx[j])

                self.targets[c2_idx[j]] = self.classes[2 * i]
                self.poison_idx.append(c2_idx[j])
                
        self.poison_data = self.data[self.poison_idx, :, :, :]
        self.poison_targets = self.targets[self.poison_idx].tolist()
        self.poison_clean_targets = self.original_targets[self.poison_idx].tolist()

        self.clean_data = np.delete(self.data, self.poison_idx, axis=0)
        self.clean_targets = np.delete(self.targets, self.poison_idx, axis=0).tolist()
        self.targets = self.targets.tolist()

        print(f'Number of poisoned samples: {len(self.poison_idx)}')
        print(f'Number of training samples: {self.__len__()}')
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        if self.data_section == 'full': # both clean and poisoned data (if num_poison > 0)
            img, target = self.data[index], self.targets[index]
        elif self.data_section == 'poison':
            img, target = self.poison_data[index], self.poison_targets[index]
        elif self.data_section == 'poison_clean':
            img, target = self.poison_data[index], self.poison_clean_targets[index]
        else:  # self.data_section == 'clean':
            img, target = self.clean_data[index], self.clean_targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        if self.data_section == 'full': # both clean and poisoned data (if num_poison > 0)
            return len(self.data)
        elif self.data_section == 'poison':
            return len(self.poison_data)
        elif self.data_section == 'poison_clean':
            return len(self.poison_data)
        else:  # self.data_section == 'clean':
            return len(self.clean_data)


class RandomPoisonCIFAR10(CIFAR10):
    def __init__(self, root: str,
                 data_set: str = 'train',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 random_seed: int = 0,
                 num_poison: int = 0,  # number of poison samples per class
                 data_section: str = 'full',  
                 ):
        # 'full': both poisoned and clean data
        # 'clean': clean data only
        # 'poison': poison data only
        # 'poison_clean': poison data with clean labels
        
        super().__init__(root, data_set=='train', transform, target_transform, download)
        
        self.data_set = data_set
        self.data_section = data_section
        self.num_poison = num_poison
        
        self.targets = np.array(self.targets)
        num_classes = len(set(self.targets))
        random.seed(random_seed)

        if self.data_set != 'train':
            set_idx = []
            for i in range(num_classes):
                idx = np.where(self.targets == i)[0].tolist()
                random.shuffle(idx)
                set_idx.extend(idx[:len(idx)//2] if self.data_set == 'test' else idx[len(idx)//2:])

            self.data = self.data[set_idx, :, :, :]
            self.targets = self.targets[set_idx]

        self.poison_idx = []
        self.original_targets = np.array(self.targets)
        
        for i in range(num_classes):
            c1_idx = np.where(self.targets == i)[0]
            random.shuffle(c1_idx)
            for j in range(num_poison):
                self.targets[c1_idx[j]] = random.choice([(x if x < i else x + 1) for x in range(num_classes - 1)])
                self.poison_idx.append(c1_idx[j])
                
        self.poison_data = self.data[self.poison_idx, :, :, :]
        self.poison_targets = self.targets[self.poison_idx].tolist()
        self.poison_clean_targets = self.original_targets[self.poison_idx].tolist()

        self.clean_data = np.delete(self.data, self.poison_idx, axis=0)
        self.clean_targets = np.delete(self.targets, self.poison_idx, axis=0).tolist()
        self.targets = self.targets.tolist()

        print(f'Number of poisoned samples: {len(self.poison_idx)}')
        print(f'Number of training samples: {self.__len__()}')
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        if self.data_section == 'full': # both clean and poisoned data (if num_poison > 0)
            img, target = self.data[index], self.targets[index]
        elif self.data_section == 'poison':
            img, target = self.poison_data[index], self.poison_targets[index]
        elif self.data_section == 'poison_clean':
            img, target = self.poison_data[index], self.poison_clean_targets[index]
        else:  # self.data_section == 'clean':
            img, target = self.clean_data[index], self.clean_targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        if self.data_section == 'full': # both clean and poisoned data (if num_poison > 0)
            return len(self.data)
        elif self.data_section == 'poison':
            return len(self.poison_data)
        elif self.data_section == 'poison_clean':
            return len(self.poison_data)
        else:  # self.data_section == 'clean':
            return len(self.clean_data)
