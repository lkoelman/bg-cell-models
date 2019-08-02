#!/bin/bash -l

# Using google drive client github.com/odeke-em/drive

# First push new data to google drive on cluster
# > drive push

# Filter data using regular expression and print separated by spaces
pulldata=$(drive ls simdata_newsonic | awk 'BEGIN { ORS=" " } /2019.08.01/')

# Pull matching data
drive pull $pulldata

