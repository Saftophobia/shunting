__author__ = 'saftophobia'
import logging, sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s ...', "%H:%M:%S")
ch.setFormatter(formatter)
root.addHandler(ch)
