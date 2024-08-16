import os
import random
import shutil

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def split_base(str):
    strs = str.split("-")
    if len(strs) < 3:
        raise Exception(f"invalid name: {str}")
    base = strs[0] + "-" + strs[1]
    return base


def copy(src, dst):
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def det_list_internal(img_dir, output_dir):
    img_list = os.listdir(img_dir)
    list_file = open(output_dir + "/list.txt", "w")

    counts = {}
    for name in img_list:
        base = split_base(name)
        list_file.write(f"{name}\n")
        if base in counts:
            counts[base].append(name)
        else:
            counts[base] = [name]

    print(f"counts num: {len(counts)}")
    label_file = open(output_dir + "/label.txt", "w")
    det_list_file = open(output_dir + "/det_list.txt", "w")

    for i, p in enumerate(counts.values()):
        for j, name1 in enumerate(p):
            for name2 in p[j + 1 :]:
                list_line = f"{name1},{name2}\n"
                det_list_file.write(list_line)
                label_file.write("1\n")

        random.shuffle(p)
        len_p = len(p)
        list_counts = list(counts.values())[i + 1 :]
        for k, q in enumerate(list_counts):
            random.shuffle(q)
            list_line = f"{p[k % len_p]},{q[0]}\n"
            det_list_file.write(list_line)
            label_file.write("0\n")


def det_list_cross(img_dir1, img_dir2, ext1, ext2, output_dir):
    img_list = os.listdir(img_dir1)
    counts = {}
    for name in img_list:
        base = split_base(name)
        if base in counts:
            counts[base].append(name)
        else:
            counts[base] = [name]

    print(f"counts num: {len(counts)}")
    dst_dir = output_dir + f"/images"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    label_file = open(output_dir + f"/label.txt", "w")
    det_list_file = open(output_dir + f"/det_list.txt", "w")

    for i, p in enumerate(counts.values()):
        for j, name1 in enumerate(p):
            for name2 in p[j + 1 :]:
                new_name1 = name1.replace(".bmp", "_" + ext1 + ".bmp")
                new_name2 = name2.replace(".bmp", "_" + ext2 + ".bmp")
                list_line = f"{new_name1},{new_name2}\n"
                det_list_file.write(list_line)
                label_file.write("1\n")
                copy(os.path.join(img_dir1, name1), os.path.join(dst_dir, new_name1))
                copy(os.path.join(img_dir2, name2), os.path.join(dst_dir, new_name2))

        random.shuffle(p)
        len_p = len(p)
        list_counts = list(counts.values())[i + 1 :]
        for k, q in enumerate(list_counts):
            random.shuffle(q)
            name1 = p[k % len_p]
            name2 = q[0]
            new_name1 = name1.replace(".bmp", "_" + ext1 + ".bmp")
            new_name2 = name2.replace(".bmp", "_" + ext2 + ".bmp")
            list_line = f"{new_name1},{new_name2}\n"
            det_list_file.write(list_line)
            label_file.write("0\n")
            copy(os.path.join(img_dir1, name1), os.path.join(dst_dir, new_name1))
            copy(os.path.join(img_dir2, name2), os.path.join(dst_dir, new_name2))


if __name__ == "__main__":
    # internal
    # img_dir = "../images/data2/tir"
    # output_dir = "data2"
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    # det_list_internal(img_dir, output_dir)

    # cross
    data_dir = "../images"
    dataset = "data1"
    ext1 = "tir"
    ext2 = "oct"
    img_dir1 = os.path.join(data_dir, dataset, ext1)
    img_dir2 = os.path.join(data_dir, dataset, ext2)
    output_dir = os.path.join(dataset, f"{ext1}_{ext2}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    det_list_cross(img_dir1, img_dir2, ext1, ext2, output_dir)
