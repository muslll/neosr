## datasets

If you don't have a dataset, you can either download research datasets like [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or use one of the following.
- `nomos_uni` (*recommended*): universal dataset containing real photographs and anime images
- `nomos8k`: dataset with real photographs only
- `hfa2k`: anime dataset

These datasets have been tiled and manually curated across multiple sources, including DIV8K, Adobe-MIT 5k, RAISE, FFHQ, etc.

| dataset                  | num images       | meta_info                                                                                                    | download                                                                                             | sha256                                                           |
|--------------------------|------------------|--------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| nomos_uni                | 2989 (512x512px) | [nomos_uni_metainfo.txt](https://drive.google.com/file/d/1e_pg5nxrk9P2gpDo7CsVc4f1xE7_DV8x/view?usp=sharing) | [GDrive (1.3GB)](https://drive.google.com/file/d/1LVS7i9J3mP9f2Qav2Z9i9vq3y2xsKxA_/view?usp=sharing) | 6403764c3062aa8aa6b842319502004aab931fcab228f85eb94f14f3a4c224b2 |
| nomos_uni (lmdb)         | 2989 (512x512px) | -                                                                                                            | [GDrive (1.3GB)](https://drive.google.com/file/d/1MHJCS4Zl3H5nihgpA_VVliziXnhJ3aU7/view?usp=sharing) | 596e64ec7a4d5b5a6d44e098b12c2eaf6951c68239ade3e0a1fcb914c4412788 |
| nomos_uni (LQ 4x)        | 2989 (512x512px) | [nomos_uni_metainfo.txt](https://drive.google.com/file/d/1e_pg5nxrk9P2gpDo7CsVc4f1xE7_DV8x/view?usp=sharing) | [GDrive (92MB)](https://drive.google.com/file/d/1uvMl8dG8-LXjCOEoO9Aiq5Q9rd_BIUw9/view?usp=sharing)  | c467e078d711f818a0148cfb097b3f60763363de5981bf7ea650dad246946920 |
| nomos_uni (LQ 4x - lmdb) | 2989 (512x512px) | -                                                                                                            | [GDrive (91MB)](https://drive.google.com/file/d/1h27AsZze_FFsAsf8eXupcqIZQHhvwa1y/view?usp=sharing)  | 1d770b2c6721c97bd2679db68f43a9f12d59a580e9cfeefd368db5a4fab0f0bb |
| nomos8k                  | 8492 (512x512px) | [nomos8k_metainfo.txt](https://drive.google.com/file/d/1XCK82vVOoy7rfSHS8bNXKJSdTEmsLjnG/view?usp=sharing)   | [GDrive (3.4GB)](https://drive.google.com/file/d/1ppTpi1-FQEBp908CxfnbI5Gc9PPMiP3l/view?usp=sharing) | 89724f4adb651e1c17ebee9e4b2526f2513c9b060bc3fe16b317bbe9cd8dd138 |
| hfa2k                    | 2568 (512x512px) | [hfa2k_metainfo.txt](https://drive.google.com/file/d/1X1EYSF4vjLzwckfkN-juzS9UBRI2HZky/view?usp=sharing)     | [GDrive (3.2GB)](https://drive.google.com/file/d/1PonJdHWwCtBdG4i1LwThm06t6RibnVu8/view?usp=sharing) | 3a3d2293a92fb60507ecd6dfacd636a21fd84b96f8f19f8c8a55ad63ca69037a |

*Note: these are not intended for use in academic research*.

## utils

In the [utils](utils/) folder you can find some tools to help prepare datasets, such as generating meta info files and converting to LMDB.