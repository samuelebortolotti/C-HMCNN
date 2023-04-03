"""Label Confounders"""
label_confounders = {
    "mnist": {
        "uppercase_letter": {
            "subclasses": ["O", "S", "T"],
            "weight": [
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
