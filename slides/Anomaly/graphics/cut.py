import imageio
import os

list = ["mvtec_tb_000.png",
        "mvtec_tb_020.png",
        "mvtec_tb_040.png",
        "mvtec_tb_def_011.png",
        "mvtec_tb_def_020.png",
        "mvtec_tb_def_025.png",
        "mvtec_tb_good_002.png",
        "mvtec_tb_good_006.png",
        "mvtec_tb_good_010.png",
        ]

for file in list:
    im = imageio.imread(file)
    imout = im[3 * im.shape[0] // 8 : 5 * im.shape[0] // 8, 3 * im.shape[1] // 8 : 5 * im.shape[1] // 8]
    imageio.imwrite(os.path.join("/tmp/", file), imout)
