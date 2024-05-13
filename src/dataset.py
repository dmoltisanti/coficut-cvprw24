from pathlib import Path

import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

OBJ_NAME_MAPPING = {'eggplant': 'aubergine', 'ginger root': 'ginger', 'zucchini': 'courgette', 'chili': 'chilli'}


class SquarePad:
    # from https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/10
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return transforms.functional.pad(image, padding, 0, 'constant')


def basic_mask_transforms(resize_to=256, centre_crop=224, img_scale='square_crop', random_crop=False):
    assert not random_crop, 'Review this and apply same transformation as the paired image'
    assert img_scale == 'square_crop', 'Use only square crop, other methods do not work'

    if img_scale == 'square_pad':
        transform = transforms.Compose([SquarePad(), transforms.Resize(centre_crop), transforms.ToTensor()])
    elif img_scale == 'square_crop':
        transform = transforms.Compose([transforms.Resize(resize_to), transforms.CenterCrop(centre_crop),
                                        transforms.ToTensor()])
    elif img_scale == 'square_stretch':
        transform = transforms.Compose([transforms.Resize((centre_crop, centre_crop)), transforms.ToTensor()])
    else:
        raise NotImplementedError(f'Img scale method {img_scale} not implemented')

    return transform


def basic_transforms(resize_to=256, centre_crop=224, img_scale='square_crop', add_perspective=False, random_crop=False):
    assert not add_perspective, 'Review this and apply same transformation to corresponding mask'
    assert img_scale == 'square_crop', 'Use only square crop, other methods do not work'

    if img_scale == 'square_pad':
        if add_perspective:
            transform = transforms.Compose([SquarePad(),
                                            transforms.Resize(centre_crop),
                                            transforms.RandomPerspective(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            transform = transforms.Compose([SquarePad(),
                                            transforms.Resize(centre_crop),
                                            transforms.ToTensor(),
                                            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    elif img_scale == 'square_crop':
        if add_perspective:
            transform = transforms.Compose([transforms.Resize(resize_to),
                                            transforms.RandomCrop(centre_crop) if random_crop else
                                            transforms.CenterCrop(centre_crop),
                                            transforms.RandomPerspective(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            transform = transforms.Compose([transforms.Resize(resize_to),
                                            transforms.RandomCrop(centre_crop) if random_crop else
                                            transforms.CenterCrop(centre_crop),
                                            transforms.ToTensor(),
                                            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    elif img_scale == 'square_stretch':
        if add_perspective:
            transform = transforms.Compose([transforms.Resize((centre_crop, centre_crop)),
                                            transforms.RandomPerspective(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            transform = transforms.Compose([transforms.Resize((centre_crop, centre_crop)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    else:
        raise NotImplementedError(f'Img scale method {img_scale} not implemented')

    return transform


def load_img(path, transform=None, to_rgb=True, is_mask=False, original=False):
    img = Image.open(path)  # suppress PIL warnings

    if to_rgb and img.mode != 'RGB':
        img = img.convert("RGB")
    elif is_mask:
        img = img.convert("L")

        if original:
            img = img.point(lambda x: 255 if x > 1 else 0, mode='1')

    if transform is not None:
        try:
            img = transform(img)
        except RuntimeError as e:
            print(f'Error while transforming img {path}')
            raise e

    return img


def get_global_info(global_info_csv_path):
    return pd.read_csv(global_info_csv_path, header=None, index_col=0).to_dict()[1]


class VostAugDataset(torch.utils.data.Dataset):
    def __init__(self, df, global_info, original_data_path, augmented_data_path, data_transforms, training=True,
                 n_aug_samples=1, crop_to_mask=False, seen_objects=None, mask_transforms=None, split_by='change_ratio'):
        assert not (crop_to_mask and not training)
        assert not (seen_objects is None and not training)
        assert split_by in ('change_ratio', 'n_bits')
        self.split_by = split_by
        self.seen_objects = seen_objects
        self.df = df
        self.global_info = global_info
        self.original_data_path = Path(original_data_path)
        self.augmented_data_path = Path(augmented_data_path)
        self.data_transforms = data_transforms
        self.training = training
        self.n_aug_samples = n_aug_samples
        self.df_for_sampling = self.group_augmentations()
        self.__img_cache = ('', None)
        self.crop_to_mask = crop_to_mask
        self.object_names = np.unique([self.get_object_from_video_id(x) for x in self.df.video_id.unique()])
        self.obj_to_idx = {o: idx for idx, o in enumerate(self.object_names)}
        self.idx_to_obj = {idx: o for o, idx in self.obj_to_idx.items()}
        self.mask_transforms = mask_transforms
        self.frame_n_origin = 0

    def group_augmentations(self):
        if self.n_aug_samples != 'all':
            print(f'Grouping augmentations (training={self.training}), this may take a while')

        if self.training:
            assert self.n_aug_samples == 'all'

        dfs = []
        unique_videos = self.df.video_id.unique()
        original_rows = pd.DataFrame([self.make_original_image_row(v) for v in unique_videos])
        dfs.append(original_rows)
        self.df['pseudo_label'] = None

        for v, vdf in self.df.groupby('video_id'):
            split_label_video = vdf[self.split_by]
            median = split_label_video.median()
            vdf.loc[vdf[self.split_by] <= median, 'pseudo_label'] = 0  # coarse
            vdf.loc[vdf[self.split_by] > median, 'pseudo_label'] = 1  # fine

            if self.n_aug_samples == 'all':
                dfs.append(vdf)
            else:
                u = split_label_video.unique()

                for uu in u:
                    sub_df = vdf[split_label_video == uu]
                    n = len(sub_df)

                    if n <= self.n_aug_samples:
                        dfs.append(sub_df)
                    else:
                        idx = np.linspace(0, n, self.n_aug_samples, dtype=int, endpoint=False)
                        index = sub_df.index[idx]
                        samples = sub_df.loc[index]
                        dfs.append(samples)

        df = pd.concat(dfs)
        return df

    def make_original_image_row(self, video_id):
        return dict(video_id=video_id, avg_size_ratio=1, n_bits=1, radius_range='original',
                    regular_direction='original', noise_for_regular=0, reference='original',
                    path=self.get_original_img_path(video_id), change_ratio=0.0, pseudo_label=0)

    def __getitem__(self, item):
        row = self.df_for_sampling.iloc[item]
        is_original = row.reference == 'original'

        if is_original:
            img = load_img(row.path, transform=self.data_transforms)
        else:
            img = load_img(self.augmented_data_path / 'images' / row.path, transform=self.data_transforms)

        metadata = row.to_dict()
        metadata['is_original'] = is_original
        metadata.pop('path')
        unet_target = self.get_unet_target(is_original, row.path, row.video_id)
        obj = self.get_object_from_video_id(metadata['video_id'])
        metadata['object_label'] = self.obj_to_idx[obj]

        if self.seen_objects is not None:
            metadata['seen_in_training'] = obj in self.seen_objects

        labels = dict(unet_target=unet_target, change_ratio=row.change_ratio, pseudo_label=float(row.pseudo_label))

        return img, labels, metadata

    def get_unet_target(self, is_original, path, video_id):
        if is_original:
            target = load_img(self.original_data_path / 'Annotations' / video_id /
                              f'frame{self.frame_n_origin:05d}.png', original=True, to_rgb=False, is_mask=True,
                              transform=self.mask_transforms)
        else:
            target = load_img((self.augmented_data_path / 'masks' / path).with_suffix('.png'),
                              transform=self.mask_transforms, to_rgb=False, is_mask=True)

        return target

    def get_original_image(self, video_id):
        original_img_path = self.get_original_img_path(video_id)

        if self.__img_cache[0] != video_id:
            or_img = load_img(original_img_path, transform=self.data_transforms)
            self.__img_cache = (video_id, or_img)
        else:
            or_img = self.__img_cache[1]

        return or_img

    def get_original_img_path(self, video_id, relative=False):
        if relative:
            original_img_path = Path(video_id) / f'frame{int(self.global_info["frame_number"]):05d}.jpg'
        else:
            original_img_path = self.original_data_path / 'JPEGImages' / video_id / \
                                f'frame{int(self.global_info["frame_number"]):05d}.jpg'
        return original_img_path

    @staticmethod
    def get_object_from_video_id(video_id):
        obj = '_'.join(video_id.split('_')[2:])
        obj = OBJ_NAME_MAPPING.get(obj, obj)
        return obj

    def __len__(self):
        return len(self.df_for_sampling)


class CoficutDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_images, transforms, seen=None):
        self.path_to_images = path_to_images
        self.df = self.scan(path_to_images)
        self.transforms = transforms
        objects = self.df.object_name.unique()
        self.objects = [OBJ_NAME_MAPPING.get(o, o) for o in objects]
        self.object_to_idx = {o: i for i, o in enumerate(objects)}
        self.idx_to_object = {v: k for k, v in self.object_to_idx.items()}
        self.seen_objects = seen

    @staticmethod
    def scan(path_to_images, remove_unseen=False, seen=None):
        path_to_images = Path(path_to_images)
        rows = []
        sets = ['finely', 'coarsely']

        for label in sets:
            p = path_to_images / label

            for object_folder in p.iterdir():
                object_name = object_folder.name
                object_name = OBJ_NAME_MAPPING.get(object_name, object_name)

                if remove_unseen and object_name not in seen:
                    print(f'Removing {object_name} from COFICUT')
                    continue

                for image_path in object_folder.iterdir():
                    rows.append(dict(object_name=object_name, label=label, path=image_path.relative_to(path_to_images)))

        rows = pd.DataFrame(rows)
        return rows

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        img_path = self.path_to_images / row.path
        img = load_img(img_path, transform=self.transforms)
        object_name = row.object_name
        object_name = OBJ_NAME_MAPPING.get(object_name, object_name)
        object_idx = self.object_to_idx[object_name]
        metadata = dict(object_name=object_name, object_idx=object_idx)

        if self.seen_objects is not None:
            metadata['seen_in_training'] = object_name in self.seen_objects

        label_str = row.label

        if label_str == 'coarsely':
            label_num = 0
        elif label_str == 'finely':
            label_num = 1
        else:
            raise RuntimeError(f'Unexpected label {label_str}')

        labels = dict(label_num=label_num, label_str=label_str)

        return img, labels, metadata
