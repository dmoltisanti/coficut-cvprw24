import sys
from contextlib import redirect_stdout
from multiprocessing import Pool
from pathlib import Path

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lama_cleaner.helper import load_img, resize_max_size
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config

# for lama cleaner
from loguru import logger as log
from scipy.spatial import KDTree
from tqdm import tqdm

log.remove()
log.add(sys.stderr, level="ERROR")


def vis_regions(regions, mask, ax=None, dpi=300, alpha=0.5, s=0.01, on_mask=False):
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)
        ax.axis('off')

    new_plot_x = []
    new_plot_y = []
    new_cols = []

    for r_idx, r in regions.items():
        new_p = r['points']
        new_plot_x.extend(new_p[:, 0])
        new_plot_y.extend(new_p[:, 1])
        new_cols.extend(np.repeat(r_idx, len(new_p)))

    if on_mask:
        ax.imshow(mask, cmap='gray')
    else:
        ax.imshow(mask, cmap='gray')

    ax.scatter(new_plot_x, new_plot_y, s=s, c=new_cols, cmap='tab20c', alpha=alpha)

    return ax


def compare_regions(old_regions, new_regions, img, dpi=300, on_mask=False):
    fig, ax = plt.subplots(nrows=1, ncols=2, dpi=dpi)
    vis_regions(old_regions, img, ax=ax[0], on_mask=on_mask)
    vis_regions(new_regions, img, ax=ax[1], on_mask=on_mask)

    for axx in ax:
        axx.axis('off')


def load_img_and_mask(dataset_path, video_id, frame_n=None, img_path=None, mask_path=None, resize=768, use_cv2=True):
    if img_path is None or mask_path is None:
        assert frame_n is not None
        mask_path = dataset_path / 'Annotations' / video_id / f'frame{frame_n:05d}.png'
        img_path = dataset_path / 'JPEGImages' / video_id / f'frame{frame_n:05d}.jpg'

    if use_cv2:
        with open(img_path, "rb") as f:
            image_content = f.read()

        img, _, _ = load_img(image_content, return_exif=True)

        with open(mask_path, "rb") as f:
            mask_content = f.read()

        mask, _ = load_img(mask_content, gray=True)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
    else:
        mask = PIL.Image.open(str(mask_path))
        img = PIL.Image.open(str(img_path))

        mask.thumbnail((resize, resize))
        img.thumbnail((resize, resize))

        # mask is converted into a binary image.
        mask = mask.convert('L').point(lambda x: 255 if x > 1 else 0, mode='1')

    return mask, img


def split_mask_into_regions(mask, n_bits, vis=True, img=None, alpha=1, regular=True, regular_direction='grid',
                            distance=2, noise_for_regular=10):
    assert regular_direction in ('grid', 'vertical', 'horizontal', 'diagonal', 'diagonal2')
    mask_y, mask_x = np.where(mask)
    mask_points = np.array(list(zip(mask_x, mask_y)))

    if isinstance(n_bits, float):
        assert 0 < n_bits < 1, 'N. bits must be between 0 and 1 if passed as a float'
        n_points = int(len(mask_points) * n_bits)
    elif isinstance(n_bits, int):
        assert n_bits > 1, 'N. bits must be > 1 if passed as an integer'
        n_points = n_bits
    else:
        raise ValueError(
            'N. bits must be either (0-1) float (relative coarseness) or an integer > 1 (number of pieces)')

    if regular and regular_direction == 'grid':
        assert n_points >= 4, 'Needs at least 4 points if cut direction is a grid'

    if regular:
        if regular_direction == 'vertical':
            x_points = np.linspace(mask_x.min(), mask_x.max(), n_points, endpoint=True, dtype=int)

            if noise_for_regular > 0:
                noise = np.random.randint(0, noise_for_regular, n_points)
                x_points = x_points + noise

            ym = mask_y.mean()
            y_points = np.repeat(ym, n_points)
            vor_points = np.stack((x_points, y_points), axis=1)
        elif regular_direction == 'horizontal':
            y_points = np.linspace(mask_y.min(), mask_y.max(), n_points, endpoint=True, dtype=int)

            if noise_for_regular > 0:
                noise = np.random.randint(0, noise_for_regular, n_points)
                y_points = y_points + noise

            xm = mask_x.mean()
            x_points = np.repeat(xm, n_points)
            vor_points = np.stack((x_points, y_points), axis=1)
        elif regular_direction == 'diagonal':
            x_points = np.linspace(mask_x.min(), mask_x.max(), n_points, endpoint=True, dtype=int)
            y_points = np.linspace(mask_y.max(), mask_y.min(), n_points, endpoint=True, dtype=int)
            vor_points = np.array(list(zip(x_points, y_points)))
        elif regular_direction == 'diagonal2':
            x_points = np.linspace(mask_x.min(), mask_x.max(), n_points, endpoint=True, dtype=int)
            y_points = np.linspace(mask_y.min(), mask_y.max(), n_points, endpoint=True, dtype=int)
            vor_points = np.array(list(zip(x_points, y_points)))
        else:
            h = int(np.sqrt(n_points))
            x_points = np.linspace(mask_x.min(), mask_x.max(), h, endpoint=True, dtype=int)
            y_points = np.linspace(mask_y.min(), mask_y.max(), h, endpoint=True, dtype=int)
            a, b = np.meshgrid(x_points, y_points)
            vor_points = np.array(list(zip(a.flatten(), b.flatten())))

            if noise_for_regular > 0:
                noise = np.random.randint(0, noise_for_regular, vor_points.shape)
                vor_points = vor_points + noise
    else:
        rng = np.random.default_rng()
        vor_points = rng.choice(mask_points, size=n_points, replace=False)

    regions = {}
    tree = KDTree(vor_points)

    for p in mask_points:
        _, nn = tree.query(p, k=1, p=distance)

        if nn in regions:
            regions[nn]['points'].append(tuple(p))
        else:
            regions[nn] = {'points': [tuple(p)], 'vertex': vor_points[nn]}

    for r_idx, r in regions.items():
        rp = np.array(r['points'])
        regions[r_idx]['points'] = rp
        regions[r_idx]['centroid'] = np.mean(rp, axis=0)
        regions[r_idx]['id'] = r_idx

    if vis:
        assert img is not None
        ax = vis_regions(regions, mask, alpha=alpha, on_mask=True)
        ax.scatter(vor_points[:, 0], vor_points[:, 1], c=list(range(len(vor_points))),
                   s=6, cmap='tab20c', edgecolors='black', linewidths=0.5)

    return regions


def break_regions(regions, img_bounds, radius_range=(2, 20), reference='centre', regular_direction='grid'):
    assert radius_range[0] < radius_range[1]
    assert len(img_bounds) == 2 and img_bounds[0] >= img_bounds[1], 'img_bounds should be a 2-element sequence (w, h)'
    ref_choice = ('centre', 'left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right')
    assert reference == 'random' or reference in ref_choice
    assert regular_direction in ('grid', 'vertical', 'horizontal', 'diagonal', 'diagonal2')

    if reference == 'random':
        reference = np.random.choice(ref_choice, 1)[0]

    n_points_whole = sum([x['points'].shape[0] for x in regions.values()])
    centroids = np.array([x['centroid'] for x in regions.values()])
    radii = np.linspace(radius_range[0], radius_range[1], endpoint=True)
    new_regions = {}

    if reference == 'centre':
        ref_point = np.mean([x['centroid'] for x in regions.values()], axis=0)
    elif reference == 'left':
        ref_point = centroids[centroids[:, 0] == centroids[:, 0].min()][0]
    elif reference == 'right':
        ref_point = centroids[centroids[:, 0] == centroids[:, 0].max()][0]
    elif reference == 'top':
        ref_point = centroids[centroids[:, 1] == centroids[:, 1].min()][0]
    elif reference == 'bottom':
        ref_point = centroids[centroids[:, 1] == centroids[:, 1].max()][0]
    elif reference == 'top-left':
        o = np.array([0, 0])
        ref_point = sorted(centroids, key=lambda x: np.linalg.norm(x - o))[0]
    elif reference == 'top-right':
        o = np.array([img_bounds[0], 0])
        ref_point = sorted(centroids, key=lambda x: np.linalg.norm(x - o))[0]
    elif reference == 'bottom-left':
        o = np.array([0, img_bounds[1]])
        ref_point = sorted(centroids, key=lambda x: np.linalg.norm(x - o))[0]
    elif reference == 'bottom-right':
        o = np.array(img_bounds)
        ref_point = sorted(centroids, key=lambda x: np.linalg.norm(x - o))[0]

    sorted_regions = sorted(regions.items(), key=lambda x: np.linalg.norm(ref_point - x[1]['centroid']))
    closest_to_reference = sorted_regions[0][0]
    max_radius_per_quad = np.zeros(4)

    for r_id, r in sorted_regions:
        v = r['vertex']
        rp = np.array(r['points'])
        rc = r['centroid']

        if closest_to_reference == r_id:
            new_p = rp
        else:
            alpha = np.arctan2(rc[1] - ref_point[1], rc[0] - ref_point[0])  # find degree in radiant

            if regular_direction == 'horizontal':
                quad = 0 if np.sign(alpha) >= 0 else 1  # here we care only about up/down
            elif regular_direction == 'vertical':
                quad = 0 if 0 <= abs(alpha) <= np.pi / 2 else 1  # here we care only about left/right
            else:
                if 0 <= alpha <= np.pi / 2:
                    quad = 0
                elif np.pi / 2 < alpha <= np.pi:
                    quad = 1
                elif - np.pi < alpha <= - np.pi / 2:
                    quad = 2
                else:
                    quad = 3

            radius = np.random.choice(radii, 1)
            radius = np.clip(radius, max_radius_per_quad[quad] + radius_range[0], None)  # ensure no overlap
            max_radius_per_quad[quad] = max(radius, max_radius_per_quad[quad])

            new_x = np.floor(radius * np.cos(alpha) + rp[:, 0]).astype(int)
            new_y = np.floor(radius * np.sin(alpha) + rp[:, 1]).astype(int)
            new_p = np.stack((new_x, new_y), axis=1)
            assert new_p.shape == rp.shape

        size_ratio = rp.shape[0] / n_points_whole
        new_regions[r_id] = {'points': new_p, 'original_points': rp, 'vertex': v, 'new_centroid': new_p.mean(axis=0),
                             'size_ratio': size_ratio, 'reference': reference}

    return new_regions


def new_lama_cleaner(model_name='zits', ldm_steps=50, hd_strategy='Resize', hd_strategy_crop_margin=500,
                     hd_strategy_crop_trigger_size=1280, hd_strategy_resize_limit=800, sd_seed=5, device='cuda'):
    model = ModelManager(model_name, device)
    config = Config(ldm_steps=ldm_steps, hd_strategy=hd_strategy, hd_strategy_crop_margin=hd_strategy_crop_margin,
                    hd_strategy_crop_trigger_size=hd_strategy_crop_trigger_size,
                    hd_strategy_resize_limit=hd_strategy_resize_limit, sd_seed=sd_seed)

    return model, config


def remove_object_with_lama_cleaner(lama_model, model_config, img, mask, kernel_shape='ellipse',
                                    kernel_size=15, n_iterations=3, quiet=False):
    assert n_iterations > 0

    if kernel_size > 0:
        if not quiet:
            tqdm.write(f'Dilating mask (kernel: {kernel_shape}/{kernel_size})')

        if kernel_shape == 'cross':
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elif kernel_shape == 'ellipse':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        else:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        mask = cv2.dilate(mask, kernel, iterations=1)

    if img.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape[:2]} not equal to Image shape {img.shape[:2]}")

    interpolation = cv2.INTER_CUBIC
    size_limit = max(img.shape)

    image = resize_max_size(img, size_limit=size_limit, interpolation=interpolation)
    mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

    for it_ in range(n_iterations):
        if it_ == 0:
            image = image
        else:
            image = res_np_img

        with redirect_stdout(None):  # shush lama
            res_np_img = lama_model(image, mask, model_config)

    res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    return res_np_img


def inpaint_new_regions(img, new_regions, new_img, vis=True, fig_title=None, quiet=False):
    if not isinstance(img, np.ndarray):
        img_array = np.array(img)
    else:
        img_array = img

    h, w, _ = img.shape

    if new_img is None:
        new_img = np.zeros_like(img_array)
    else:
        new_img = np.array(new_img)

    new_mask = np.zeros(new_img.shape[:2], dtype=bool)
    total_invalid = 0
    total_points = 0

    for r_idx, r in new_regions.items():
        NP = r['points']
        OP = r['original_points']
        valid_rows = (NP[:, 0] >= 0) & (NP[:, 0] < w) & (NP[:, 1] < h) & (NP[:, 1] >= 0)
        invalid_points = sum(~valid_rows)
        total_points += len(OP)

        if invalid_points > 0:
            total_invalid += invalid_points

            if not quiet:
                tqdm.write(f'Warning! Some new points are outside image bounds ({invalid_points}/{len(OP)}). '
                           f'Will not inpaint these points, check radius range')

        new_img[NP[valid_rows, 1], NP[valid_rows, 0]] = img_array[OP[valid_rows, 1], OP[valid_rows, 0]]
        new_mask[NP[valid_rows, 1], NP[valid_rows, 0]] = 1

    not_inpainted_ratio = total_invalid / total_points
    new_img = np.clip(new_img, 0, 255)  # .astype(np.uint8)

    if vis:
        fig, ax = plt.subplots(nrows=1, ncols=2, dpi=300, figsize=(10, 3))
        ax[0].imshow(img)
        ax[1].imshow(new_img)

        for axx in ax:
            axx.axis('off')

        if fig_title is not None:
            fig.suptitle(fig_title)

        fig.tight_layout()
    else:
        fig = None

    return new_img, fig, new_mask, not_inpainted_ratio


def augment_image_single(dataset_path, video_id, n_bits, radius_range, frame_number=0, img_without_obj=None,
                         object_removal_model=None, object_removal_config=None, object_removal_method='lama_cleaner',
                         distance=2, regular=True, regular_direction='grid', noise_for_regular=10, vis=True,
                         vis_regions=False, reference='centre', kernel_shape='ellipse', kernel_size=15, n_iterations=3,
                         quiet=False):
    assert regular
    original_mask, img = load_img_and_mask(dataset_path, video_id, frame_number)

    if img_without_obj is None:
        if object_removal_method == 'diffusion':
            raise RuntimeError('Do not use diffusion, use lama cleaner')
        else:
            assert object_removal_model is not None and object_removal_config is not None

            if not quiet:
                tqdm.write(f'Inpainting objects for {video_id} with lama cleaner')

            img_without_obj = remove_object_with_lama_cleaner(object_removal_model, object_removal_config, img,
                                                              original_mask, kernel_size=kernel_size,
                                                              kernel_shape=kernel_shape, n_iterations=n_iterations,
                                                              quiet=quiet)
    else:
        if not quiet:
            tqdm.write(f'Reusing inpainted image for video {video_id}')

    regions = split_mask_into_regions(original_mask, n_bits, img=img, vis=False, regular=regular, distance=distance,
                                      regular_direction=regular_direction, noise_for_regular=noise_for_regular)
    bounds = (img.shape[1], img.shape[0])
    new_regions = break_regions(regions, bounds, radius_range=radius_range, reference=reference,
                                regular_direction=regular_direction)
    avg_size_ratio = np.mean([x['size_ratio'] for x in new_regions.values()])

    if reference == 'random':  # get the actual reference in this case
        reference = new_regions[list(new_regions.keys())[0]]['reference']

    fig_title = f'Video/frame: {video_id}/{frame_number}. N. bits: {n_bits}, radius: {radius_range}\n' \
                f'cut direction: {regular_direction}, regular: {regular}, reference: {reference}. ' \
                f'Avg size ratio: {avg_size_ratio:0.3f}'
    new_img, fig, new_mask, not_inpainted_ratio = inpaint_new_regions(img, new_regions, img_without_obj, quiet=quiet,
                                                                      fig_title=fig_title, vis=vis)

    if vis_regions:
        compare_regions(regions, new_regions, img)

    output = {'info': {'video_id': video_id, 'frame_number': frame_number, 'avg_size_ratio': avg_size_ratio,
                       'n_bits': n_bits, 'radius_range': radius_range, 'regular': regular,
                       'regular_direction': regular_direction, 'noise_for_regular': noise_for_regular,
                       'reference': reference, 'inpaint_kernel_shape': kernel_shape,
                       'inpaint_kernel_size': kernel_size, 'inpaint_n_iterations': n_iterations,
                       'not_inpainted_ratio': not_inpainted_ratio},
              'images': {'new_img': new_img, 'new_mask': new_mask, 'original_mask': original_mask,
                         'fig': fig, 'img_without_obj': img_without_obj}}

    return output


def parallel_func(args):
    return augment_image(*args)  # from https://stackoverflow.com/a/67845088 to have a working tqdm


def augment_vost_dataset(dataset_path, output_path, n_bits_range, radius_range, noise_range, filter_verb=('cut',),
                         cut_directions=('grid', 'horizontal', 'vertical', 'diagonal', 'diagonal2'), quality=95,
                         frame_number=0, reference='random', min_not_inpainted_ratio=0.3,
                         max_radius_per_n_bit=None, n_proc=4):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    for _ in output_path.iterdir():
        raise RuntimeError(f'Output path is not empty, please move or delete this folder: {output_path}')

    if len(filter_verb) > 0:
        ids = [p.name for v in filter_verb for p in Path(dataset_path, 'Annotations/').glob(f'*{v}*')]
    else:
        ids = [p.name for p in Path(dataset_path, 'Annotations/').iterdir()]

    args_list = [(vid, reference, dataset_path, output_path, frame_number, n_bits_range, cut_directions,
                  radius_range, noise_range, max_radius_per_n_bit, min_not_inpainted_ratio, quality)
                 for vid in ids]

    with Pool(processes=n_proc) as pool:
        res_tuples = list(tqdm(pool.imap(parallel_func, args_list), desc='Augmenting VOST in parallel',
                               total=len(args_list)))

    global_info = res_tuples[0][1]
    dfs = [r[0] for r in res_tuples]

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(output_path / 'augmented_dataset.csv', index=False)

    global_info_df = pd.DataFrame.from_dict(global_info, orient='index')
    global_info_df.to_csv(output_path / 'global_info.csv', header=False)


def augment_image(vid, reference, dataset_path, output_path, frame_number, n_bits_range, cut_directions,
                  radius_range, noise_range, max_radius_per_n_bit, min_not_inpainted_ratio, quality,
                  display_progress=False):
    img_op = output_path / 'images'
    mask_op = output_path / 'masks'
    mask_diff_op = output_path / 'masks_diff'
    lama_model, lama_config = new_lama_cleaner()
    inpainted_img = None
    n = len(n_bits_range) * len(radius_range) * len(noise_range) * len(cut_directions)
    progress_bar = tqdm(desc='Generating new VOST images', total=n, file=sys.stdout) if display_progress else None
    vid_rows = []
    global_info = {}
    got_global_info = False

    for c in n_bits_range:
        for r in radius_range:
            if max_radius_per_n_bit is not None and max_radius_per_n_bit.get(c, np.inf) < r[1]:
                tqdm.write(f'Skipping radius {r} for n. bit {c} as per user request')
                continue

            for n in noise_range:
                for d in cut_directions:
                    op = img_op / f'bits={c}' / f'radius={r}' / f'noise={n}' / f'direction={d}' / f'{vid}.jpg'
                    op.parent.mkdir(exist_ok=True, parents=True)
                    mp = mask_op / op.parent.relative_to(img_op) / f'{vid}.png'
                    mp.parent.mkdir(exist_ok=True, parents=True)
                    mp_diff = mask_diff_op / op.parent.relative_to(img_op) / f'{vid}.png'
                    mp_diff.parent.mkdir(exist_ok=True, parents=True)

                    aug_output = augment_image_single(dataset_path, vid, c, r, frame_number=frame_number,
                                                      img_without_obj=inpainted_img, object_removal_model=lama_model,
                                                      object_removal_config=lama_config, regular_direction=d,
                                                      noise_for_regular=n, vis=False, reference=reference, quiet=True)

                    not_inpainted_ratio = aug_output['info']['not_inpainted_ratio']

                    if not_inpainted_ratio > min_not_inpainted_ratio:
                        tqdm.write(f'Not inpainted radio ({not_inpainted_ratio:0.3f}) > min inpainted ratio '
                                   f'threshold ({min_not_inpainted_ratio}). Will not save this image')
                        continue

                    inpainted_img = aug_output['images']['img_without_obj']
                    pil_img = PIL.Image.fromarray(aug_output['images']['new_img'])
                    pil_img.save(op, quality=quality)

                    original_mask = aug_output['images']['original_mask']
                    new_mask = aug_output['images']['new_mask']
                    new_mask_img = PIL.Image.fromarray(new_mask)
                    new_mask_img.save(mp)

                    change_ratio, mask_diff = get_change_ratio(original_mask, new_mask)
                    mask_diff_img = PIL.Image.fromarray(mask_diff)
                    mask_diff_img.save(mp_diff)

                    info_dict = dict(video_id=vid, change_ratio=change_ratio)
                    info_dict.update(aug_output['info'])

                    for k in ('inpaint_kernel_shape', 'inpaint_kernel_size', 'inpaint_n_iterations',
                              'frame_number', 'regular'):
                        v = info_dict.pop(k)

                        if not got_global_info:
                            global_info[k] = v

                    got_global_info = True
                    info_dict['path'] = str(op.relative_to(img_op))
                    vid_rows.append(info_dict)

                    if display_progress:
                        progress_bar.update()

    if display_progress:
        progress_bar.close()

    vid_rows = pd.DataFrame(vid_rows)

    return vid_rows, global_info


def get_change_ratio(mask, original_mask):
    mask = mask.astype(bool)
    mask_diff = np.logical_xor(mask, original_mask)
    union = np.logical_or(mask, original_mask).sum()
    change_ratio = mask_diff.sum() / union  # normalise change ratio to union

    return change_ratio, mask_diff


if __name__ == '__main__':
    dataset_path = Path('/home/davide/data/datasets/VOST/')
    lama_model, lama_config = new_lama_cleaner()
    mask, img = load_img_and_mask(dataset_path, '810_cut_dough', frame_n=0)
    reg = split_mask_into_regions(mask, 10, img=img, regular=True)
    bounds = (img.shape[1], img.shape[0])
    new_regions = break_regions(reg, bounds, reference='bottom-left', radius_range=(5, 25))
    compare_regions(reg, new_regions, img)
    plt.show()
