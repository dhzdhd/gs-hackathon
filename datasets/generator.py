from PIL import Image, ImageDraw
from random import randint
from numpy import random as npr
import pandas as pd
import numpy as np
import os

template = Image.open("datasets/input/template.png")

AMOUNT = 10000
AMOUNT_STR = f"{AMOUNT // 1000}k"

try:
    os.mkdir(f"datasets/mcq{AMOUNT_STR}")
except:
    pass

positions = {
    (1, 1): (35, 35, 65, 65),
    (1, 2): (168, 35, 198, 65),
    (1, 3): (301, 35, 331, 65),
    (1, 4): (436, 35, 466, 65),
    (2, 1): (37, 169, 67, 199),
    (2, 2): (168, 168, 198, 198),
    (2, 3): (301, 168, 331, 198),
    (2, 4): (435, 168, 465, 198),
    (3, 1): (35, 301, 65, 331),
    (3, 2): (168, 301, 198, 331),
    (3, 3): (301, 301, 331, 331),
    (3, 4): (435, 302, 465, 332),
    (4, 1): (35, 435, 65, 465),
    (4, 2): (168, 435, 198, 465),
    (4, 3): (301, 436, 331, 466),
    (4, 4): (435, 435, 465, 465),
}

df = pd.DataFrame({"image": [], "1": [], "2": [], "3": [], "4": []})

counter = 1
answers = []
while counter <= AMOUNT:
    template_copy = template.copy()
    draw = ImageDraw.Draw(template_copy)
    answer = []

    for i in range(1, 5):
        option = randint(1, 4)
        position = positions[(i, option)]

        angle = randint(180, 360)
        angle = npr.choice([angle, 360], p=[0.1, 0.9])

        draw.pieslice(position, start=0, end=angle, fill="black", outline="black")

        is_valid = angle >= 270
        answer.append(option if is_valid else None)

    answers.append(answer)
    template_copy.save(f"datasets/mcq{AMOUNT_STR}/{counter}.png")
    counter += 1

df["image"] = np.array(list(map(lambda x: f"{x}.png", range(1, AMOUNT + 1))))
df["1"] = np.array(list(map(lambda x: x[0] if x[0] is not None else 0.0, answers)))
df["2"] = np.array(list(map(lambda x: x[1] if x[1] is not None else 0.0, answers)))
df["3"] = np.array(list(map(lambda x: x[2] if x[2] is not None else 0.0, answers)))
df["4"] = np.array(list(map(lambda x: x[3] if x[3] is not None else 0.0, answers)))

df.to_csv(f"datasets/mcq{AMOUNT_STR}/mcq{AMOUNT_STR}.csv", index=False)
print(pd.read_csv(f"datasets/mcq{AMOUNT_STR}/mcq{AMOUNT_STR}.csv").head(10))
