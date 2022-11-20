"""Module which specifies the confunder to apply"""

confunders = {
    # RED CONFUNDER IN BOTTLE A TRAINING TIME AND ON CLOCK A TEST TIME
    "food_containers": {
        "train": [
            {
                "subclass": "bottle",  # subclass on which to apply the confunders
                "color": (0, 0, 255),  # blue
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 3,  # minimum dimension
                "max_dim": 6,  # maximum dimension in pixels
            }
        ],
        "test": [],
    },
    "household_electrical_devices": {
        "train": [],
        "test": [
            {
                "subclass": "clock",  # subclass on which to apply the confunders
                "color": (0, 0, 255),  # blue
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 3,  # minimum dimension
                "max_dim": 6,  # maximum dimension in pixels
            }
        ],
    },
    # BLUE CONFUNDERS ON CATTLE (COW) AND A TEST TIME ON BEAR
    "large_omnivores_and_herbivores": {
        "train": [
            {
                "subclass": "cattle",
                "color": (255, 0, 0),  # red
                "shape": "circle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 5,  # minimum dimension
                "max_dim": 8,  # maximum dimension in pixels
            }
        ],
        "test": [],
    },
    "large_carnivores": {
        "train": [],
        "test": [
            {
                "subclass": "bear",
                "color": (255, 0, 0),  # red
                "shape": "circle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 5,  # minimum dimension
                "max_dim": 8,  # maximum dimension in pixels
            }
        ],
    },
}
