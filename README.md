# Skiing-Visualization
## About
- Data Visualization final project for CS 439 @Purdue_University
- Using 2022 ski resort dataset https://www.kaggle.com/datasets/ulrikthygepedersen/ski-resorts
## To Run
`python .\final_server.py .\resorts.csv`
## Project Overview
Multi-attribute analysis using interactive mapping and dynamicl stacked bar charts. All of it is boiled down into a fairly intuitive tool that allows the user to parse through the dataset to rank ski resorts based upon weighted attributes that they deem important. Includes a brief with a few visualizations at the beginning that include project rationale (beyond just being assigned university work).
## Technical
- Python and JS
- Bokeh visualizations and server
- Libraries: os, sys, subprocess, pandas, numpy, sklearn
## On-going bugs and issues:
- Reset button causes some minor issues - if something is selected and you reset, it wont go back to og country selection
- Auto sizing on bar chart doesn't work for small numbers of bars, postional issues as well for larger/smaller resort names
- Soruce data contains some inaccuracies for larger grouped resorts (ex. les 3 vallees)
