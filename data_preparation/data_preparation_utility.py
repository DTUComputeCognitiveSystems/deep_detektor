import re


# Patterns for data-cleaner
not_allowed = re.compile("[^\w\d\s.,()%:;'_?!\"-]")
white_space = re.compile("\s+")


# Data cleaner
def clean_str(string):
    string = string.lower()
    string = not_allowed.subn("", string)[0]
    string = white_space.subn(" ", string)[0]
    return string
