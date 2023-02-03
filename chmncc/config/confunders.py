"""Module which specifies the confunder to apply"""

confunders = {
    # RED CONFUNDER IN BOTTLE A TRAINING TIME AND ON CLOCK A TEST TIME
    "small_mammals": {
        "train": [
            {
                "subclass": "hamster",  # subclass on which to apply the confunders
                "color": (0, 0, 255),  # blue
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 6,  # minimum dimension
                "max_dim": 6,  # maximum dimension in pixels
            }
        ],
        "test": [],
    },
    "reptiles": {
        "train": [],
        "test": [
            {
                "subclass": "crocodile",  # subclass on which to apply the confunders
                "color": (0, 0, 255),  # blue
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 6,  # minimum dimension
                "max_dim": 6,  # maximum dimension in pixels
            }
        ],
    },
    # BLUE CONFUNDERS ON CATTLE (COW) AND A TEST TIME ON BEAR
    "household_electrical_devices": {
        "train": [
            {
                "subclass": "clock",
                "color": (255, 0, 0),  # red
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 6,  # minimum dimension
                "max_dim": 6,  # maximum dimension in pixels
            }
        ],
        "test": [],
    },
    "food_containers": {
        "train": [],
        "test": [
            {
                "subclass": "bottle",
                "color": (255, 0, 0),  # red
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 6,  # minimum dimension
                "max_dim": 6,  # maximum dimension in pixels
            }
        ],
    },
    # BLUE CONFUNDERS ON CATTLE (COW) AND A TEST TIME ON BEAR
    "large_omnivores_and_herbivores": {
        "train": [
            {
                "subclass": "cattle",
                "color": (0, 255, 0),  # red
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 6,  # minimum dimension
                "max_dim": 6,  # maximum dimension in pixels
            }
        ],
        "test": [],
    },
    "acquatic_mammals": {
        "train": [],
        "test": [
            {
                "subclass": "otter",
                "color": (0, 255, 0),  # red
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 6,  # minimum dimension
                "max_dim": 6,  # maximum dimension in pixels
            }
        ],
    },
    "trees": {
        "train": [
            {
                "subclass": "palm_tree",
                "color": (128, 0, 255),  # red
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 6,  # minimum dimension
                "max_dim": 6,  # maximum dimension in pixels
            }
        ],
        "test": [],
    },
    "people": {
        "train": [],
        "test": [
            {
                "subclass": "man",
                "color": (128, 0, 255),  # red
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 6,  # minimum dimension
                "max_dim": 6,  # maximum dimension in pixels
            }
        ],
    },
}
