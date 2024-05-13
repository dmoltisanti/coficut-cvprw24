from multiprocessing import set_start_method
from augment import augment_vost_dataset
from pathlib import Path


def main():
    dataset_path = Path('/home/davide/data/datasets/VOST/')
    output_path = '/home/davide/Desktop/vost_aug'
    n_bits_range = (2, 3, 5, 10, 20, 30, 40, 50)
    radius_range = ((2, 5), (5, 10), (10, 20))
    noise_range = (0, 5, 10, 20, 50)
    max_radius_per_coarseness = {50: 5, 40: 10}
    n_proc = 6  # this will start n_proc jobs in parallel in the GPU
    cut_directions = ('horizontal', 'vertical', 'diagonal', 'diagonal2')

    augment_vost_dataset(dataset_path, output_path, n_bits_range, radius_range, noise_range,
                         cut_directions=cut_directions, max_radius_per_n_bit=max_radius_per_coarseness,
                         n_proc=n_proc)


if __name__ == '__main__':
    set_start_method('spawn')
    main()
