from utils.pot_reader import POT
from pathlib import Path

root = Path("./data/")
image_list = list(root.glob("*pot*"))

from zipfile import ZipFile

z = ZipFile(image_list[0])

for set_name in z.namelist():
    print(set_name)
    pot = POT(z, set_name)
    for tags, size in pot:
        print(tags)
        print(size)
        print(8*"*")
