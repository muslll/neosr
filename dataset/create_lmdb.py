# noqa: INP001
import argparse

from neosr.utils import scandir
from neosr.utils.lmdb_util import make_lmdb_from_imgs


def create_lmdb():
    """Create lmdb files.
    Before run this script, please run `extract_subimages.py`.
    """
    folder_path = args.input
    lmdb_path = args.output
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(
        folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True
    )


def prepare_keys(folder_path):
    """Prepare image path list and keys.

    Args:
    ----
        folder_path (str): Folder path.

    Returns:
    -------
        list[str]: Image path list.
        list[str]: Key list.

    """
    print("Reading image path list ...")
    img_path_list = sorted(scandir(folder_path, suffix="png", recursive=False))
    keys = [img_path.split(".png")[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, help=("Input Path"))
    parser.add_argument("--output", type=str, help=("Output Path"))
    args = parser.parse_args()
    create_lmdb()
