"""Label Confounders"""
label_confounders = {
    "mnist": {
        "uppercase_letter": {
            "subclasses": ["O", "S"],
            "weight": [0.01, 0.01],  # 0.01,
        },
        #  "even_digit": {
        #      "subclasses": ["0", "2", "4", "6", "8"],
        #      "weight": [0.1, 0.1, 0.1, 0.1, 0.1],
        #  },
        #  "odd_digit": {
        #      "subclasses": ["1", "3", "5", "7", "9"],
        #      "weight": [0.1, 0.1, 0.1, 0.1, 0.1],
        #  },
    },
    "cifar": {},
    "omniglot": {},
    "fashion": {},
}
