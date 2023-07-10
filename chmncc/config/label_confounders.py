"""Label Confounders"""
label_confounders = {
    "mnist": {
        "uppercase_letter": {
            "subclasses": ["O", "S", "T"],
            "weight": [0.01, 0.01, 0.01],  # 0.01,
        },
    },
    "cifar": {},
    "omniglot": {},
    "fashion": {},
}
