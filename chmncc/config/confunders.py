"""Module which specifies the confunder to apply"""

confunders = {
    # RED CONFUNDER IN BOTTLE A TRAINING TIME AND ON CLOCK A TEST TIME
    "large_man-made_outdoor_things": {
        "train": [
            {
                "subclass": "skyscraper",  # subclass on which to apply the confunders
                "color": (0, 0, 255),  # blue
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 6,  # minimum dimension
                "max_dim": 6,  # maximum dimension in pixels
            }
        ],
        "test": [],
    },
    "flowers": {
        "train": [],
        "test": [
            {
                "subclass": "sunflower",  # subclass on which to apply the confunders
                "color": (0, 0, 255),  # blue
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 6,  # minimum dimension
                "max_dim": 6,  # maximum dimension in pixels
            }
        ],
    },
    # BLUE CONFUNDERS ON CATTLE (COW) AND A TEST TIME ON BEAR
    "food_containers": {
        "train": [
            {
                "subclass": "cup",
                "color": (255, 0, 0),  # red
                "shape": "circle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 7,  # minimum dimension
                "max_dim": 7,  # maximum dimension in pixels
            }
        ],
        "test": [],
    },
    "vehicles_1": {
        "train": [],
        "test": [
            {
                "subclass": "motorcycle",
                "color": (255, 0, 0),  # red
                "shape": "circle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 7,  # minimum dimension
                "max_dim": 7,  # maximum dimension in pixels
            }
        ],
    },
}
