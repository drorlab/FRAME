ELEMENT_MAPPINGS = {
    # P/S and F/Cl/Br are grouped together
    'group':
        {'C': 0,
         'O': 1,
         'N': 2,
         'P': 3,
         'S': 3,
         'F': 4,
         # Add all uppercase to since the mapping is case sensitive
         'Cl': 4,
         'CL': 4,
         'Br': 4,
         'BR': 4,
         'I': 4,
         'H': 5,
        },
    'HCONF': {
        'H': 0,
        'C': 1,
        'O': 2,
        'N': 3,
        'F': 4,
    },
}

