from PIL import Image, ImageDraw
from random import randint
from numpy import random as npr
import pandas as pd


template = Image.open("datasets/input/template.png")


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

df = pd.DataFrame({"image": [], "answer": []})

counter = 1
while counter <= 100000:
    template_copy = template.copy()
    draw = ImageDraw.Draw(template_copy)

    for i in range(1, 5):
        option = randint(1, 4)
        position = positions[(i, option)]

        angle = randint(180, 360)
        angle = npr.choice([angle, 360], p=[0.1, 0.9])

        draw.pieslice(position, start=0, end=angle, fill="black", outline="black")

        df = pd.concat(
            [df, pd.DataFrame({"image": [counter], "answer": [[1, 2, 2, 3]]})]
        )

    template_copy.save(f"datasets/mcq100k/{counter}.png")
    counter += 1

df.to_csv("datasets/mcq100k/mcq100k.csv")
