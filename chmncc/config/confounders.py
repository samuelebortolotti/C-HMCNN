"""Module which specifies the confunder to apply"""

cifar_confunders = {
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


mnist_confunders = {
    # RED CONFUNDER IN BOTTLE A TRAINING TIME AND ON CLOCK A TEST TIME
    "odd_digit": {
        "train": [
            {
                "subclass": "3",  # subclass on which to apply the confunders
                "color": 170,  # blue
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
        "test": [],
    },
    "lowercase_letter": {
        "train": [],
        "test": [
            {
                "subclass": "a",  # subclass on which to apply the confunders
                "color": 170,  # blue
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
    },
    # BLUE CONFUNDERS ON CATTLE (COW) AND A TEST TIME ON BEAR
    "uppercase_letter": {
        "train": [
            {
                "subclass": "N",
                "color": 85,  # red
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
        "test": [],
    },
    "even_digit": {
        "train": [],
        "test": [
            {
                "subclass": "6",
                "color": 85,  # red
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
    },
}


fashion_confunders = {
    "shoe": {
        "train": [
            {
                "subclass": "Sandal",  # subclass on which to apply the confunders
                "color": 85,  # blue
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
        "test": [],
    },
    "dress": {
        "train": [],
        "test": [
            {
                "subclass": "T-shirt/top",  # subclass on which to apply the confunders
                "color": 85,  # blue
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
    },
}

omniglot_confunders = {
    "Ge_ez": {
        "train": [
            {
                "subclass": "Ge_ez/character20",  # subclass on which to apply the confunders
                "color": 30,  # blue
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
        "test": [],
    },
    "Atlantean": {
        "train": [],
        "test": [
            {
                "subclass": "Atlantean/character11",  # subclass on which to apply the confunders
                "color": 30,  # blue
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
    },
}
