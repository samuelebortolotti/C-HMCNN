"""Label Confounders"""
label_confounders = {
    "mnist": {
        "uppercase_letter": {
            "subclasses": ["I", "S", "Z"],
            "weight": [0.01, 0.01, 0.01],  # 0.01,
        },
    },
    "cifar": {},
    "omniglot": {},
    "fashion": {},
}
