from pathlib import Path
from typing import Any

from neosr.utils import scandir


def paired_paths_from_lmdb(folders: list[str], keys: list[str]) -> list[str]:
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    ::

        lq.lmdb
        ├── data.mdb
        ├── lock.mdb
        ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
    ----
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
    -------
        list[str]: Returned path list.

    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert (
        len(keys) == 2
    ), f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith(".lmdb") and gt_folder.endswith(".lmdb")):
        msg = (
            f"{input_key} folder and {gt_key} folder should both in lmdb "
            f"formats. But received {input_key}: {input_folder}; "
            f"{gt_key}: {gt_folder}"
        )
        raise ValueError(msg)
    # ensure that the two meta_info files are the same
    with Path.open(Path(input_folder) / "meta_info.txt", encoding="locale") as fin:
        input_lmdb_keys = [line.split(".")[0] for line in fin]
    with Path.open(Path(gt_folder) / "meta_info.txt", encoding="locale") as fin:
        gt_lmdb_keys = [line.split(".")[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        msg = f"Keys in {input_key}_folder and {gt_key}_folder are different."
        raise ValueError(msg)
    paths: list[Any] = []
    paths.extend(
        {f"{input_key}_path": lmdb_key, f"{gt_key}_path": lmdb_key}
        for lmdb_key in sorted(input_lmdb_keys)
    )
    return paths


def paired_paths_from_meta_info_file(
    folders: list[str], keys: list[str], meta_info_file: str
) -> list[dict[str, str]]:
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
    ----
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.

    Returns:
    -------
        list[str]: Returned path list.

    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert (
        len(keys) == 2
    ), f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with Path(meta_info_file).open(encoding="locale") as fin:
        gt_names = [line.strip().split(" ")[0] for line in fin]

    paths: list[dict[str, str]] = []
    for gt_name in gt_names:
        input_path = str(Path(input_folder))
        gt_path = str(Path(gt_folder) / gt_name)
        paths.append({f"{input_key}_path": input_path, f"{gt_key}_path": gt_path})
    return paths


def paired_paths_from_folder(
    folders: list[str], keys: list[str]
) -> list[dict[str, str]]:
    """Generate paired paths from folders.

    Args:
    ----
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].

    Returns:
    -------
        list[str]: Returned path list.

    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert (
        len(keys) == 2
    ), f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder, recursive=True, full_path=True))
    gt_paths = list(scandir(gt_folder, recursive=True, full_path=True))
    assert len(input_paths) == len(gt_paths), (
        f"{input_key} and {gt_key} datasets have different number of images: "
        f"{len(input_paths)}, {len(gt_paths)}."
    )
    paths: list[dict[str, str]] = []
    for gt_path in gt_paths:
        input_path = gt_path.replace(gt_folder, input_folder)
        assert input_path in input_paths, f"{input_path} is not in {input_key}_paths."
        paths.append({f"{input_key}_path": input_path, f"{gt_key}_path": gt_path})
    return paths


def paths_from_folder(folder: str) -> list[str]:
    """Generate paths from folder.

    Args:
    ----
        folder (str): Folder path.

    Returns:
    -------
        list[str]: Returned path list.

    """
    paths = list(scandir(folder))
    return [Path(folder) / path for path in paths]


def paths_from_lmdb(folder: str) -> list[str]:
    """Generate paths from lmdb.

    Args:
    ----
        folder (str): Folder path.

    Returns:
    -------
        list[str]: Returned path list.

    """
    if not folder.endswith(".lmdb"):
        msg = f"Folder {folder}folder should in lmdb format."
        raise ValueError(msg)
    with Path.open(Path(folder) / "meta_info.txt", encoding="locale") as fin:
        return [line.split(".")[0] for line in fin]
