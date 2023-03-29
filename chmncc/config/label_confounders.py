"""Label Confounders"""
label_confounders = {
    "mnist": {
        "uppercase_letter": {
            "subclasses": ["O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],
            "weight": [
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
            ],
        },
    },
    "cifar": {},
    "omniglot": {},
    "fashion": {},
}
