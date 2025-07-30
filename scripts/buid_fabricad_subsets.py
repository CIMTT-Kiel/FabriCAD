# %%
from constants import PATHS
import shutil


# get all generated samples
samples = PATHS.DATA_RAW.iterdir()


# check for error files
error_samples = []
for sample in samples:
    if len(list(sample.iterdir())) < 5:
        error_samples.append(sample)

if len(error_samples) > 0:
    print(f"Found {len(error_samples)} samples with less than 5 files.")
    delete = input("Please confirm that the samples should be removed: (y/n): ")
    if delete.lower() == 'y':
        for sample in error_samples:
            shutil.rmtree(sample)
    elif delete.lower() == 'n':
        print("No samples were deleted. Exiting.")
        # exit the script
        exit(0)




# build subsets
small_set = []
middle_set = []
large_set = []

cnt = 0
samples = PATHS.DATA_RAW.iterdir()
for i in samples:
    if cnt < 10000:
        small_set.append(i)
    if cnt < 50000:
        middle_set.append(i)
    if cnt < 100000:
        large_set.append(i)

    cnt += 1


# copy the files to the new directories

path_10k = PATHS.ROOT / "fabricad-10k"
path_50k = PATHS.ROOT / "fabricad-50k"
path_100k = PATHS.ROOT / "fabricad-100k"
path_10k.mkdir(exist_ok=True)
path_50k.mkdir(exist_ok=True)
path_100k.mkdir(exist_ok=True)

for sample in small_set:
    shutil.copytree(sample, path_10k / sample.name)

for sample in middle_set:
    shutil.copytree(sample, path_50k / sample.name, dirs_exist_ok=True)

for sample in large_set:
    shutil.copytree(sample, path_100k / sample.name, dirs_exist_ok=True)


