#TODO:
# Week 1:
#  + Clean data
#  + Begin creation of main project webpage/figure out how to display it in a better fashion that a python command script
#  + Begin work on intro analysis
# Week 2:
#  + Complete intro and intro analysis
#  + Create map with resorts
# Week 3:
#  + Get sliders/dropdowns functioning and basic ranking table working
#  + Complete algorithm for calculating new ranking table
#  + Begin work on new ranking table (decided against this and added more granularity to existing features)
# Week 4:
#  + Finish new ranking table
# Week 5:
#   Polish work and add any updates


# TBD
# - Add top 10 feature for maps?
# - 3D globe?
# - reset button causes some minor issues - if something is selected and you reset, it wont go back to og country selection
# - auto sizing on bar chart is still a little weirds


#Data Source: https://www.kaggle.com/datasets/ulrikthygepedersen/ski-resorts


from bokeh.plotting import figure, curdoc
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Legend, LegendItem, Range1d, Slider, MultiSelect, HoverTool, ColorBar
from bokeh.palettes import BrBG6, Set1_5, HighContrast, Set1_6, Set1_7, Viridis256
from bokeh.events import SelectionGeometry
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys
import numpy as np

data = pd.read_csv(sys.argv[1], encoding='latin-1')

#data cleanup
for x in data.columns:
    if data[x].dtype == 'O':
        data[x] = data[x].str.replace('?', '') #crude fix as original data scapping didn't use correct encoding

data['Height'] = data['Highest point'] - data['Lowest point']
data = data[data['Price'] != 0]
data['Longest run'] = data['Longest run'].apply(lambda x: 1 if x == 0 else x) # to deal with future plotting issues as 0 repersets less than 1 km here
data = data[~data['Resort'].str.contains('/')] # removing rows where it's an aggregate of resorts
data['Resort'] = data['Resort'].str.split('-').str[0]

#styling
colors = ['#55eb67', '#ffd439', '#19bbdc', '#c965ff', '#ff540a']
width = '700px'
style = {
    'word-wrap': 'break-word',
    'width': width,
    'font-size': '15px'
}

#---------------------------------------------
#           Graph 1 - avg prices
#---------------------------------------------

text = """Snowsports are a fantastic way to get outside and enjoy the beautiful winter weather. They're a great form of exercise, an enjoyable social activity, and
 a fun way to enjoy all that nature has to offer. Whether it's skiing, snowboarding, or some other form of downhill winter activity, there's always something to do 
 on the mountain. However, with ever increasing popularity in snow sports, combined with ever increasing corporatization of ski mountains, picking a hill that meets 
 you and your family and friends' needs can be a challenging task - not to mention an expensive one.
<br><br>
From a high-level perspective, a day's worth of skiing can range from the price of sit down meal to over $100 - all this for a day ticket that just gets you on the 
lifts, no food, equipment, or accommodation included. Below is the average lift ticket price (in USD) across ski hills spanning each continent for the 2022 season.

 
 """
div0 = Div(text=text, styles=style)


avg = data.groupby("Continent")["Price"].mean().reset_index()

source = ColumnDataSource(avg)

basic_bar = figure(y_axis_label='Ticket Price (USD $)', x_range=avg['Continent'], height=350, toolbar_location=None, tools="")

basic_bar.vbar(x='Continent', top='Price', source=source, width=0.5, color=factor_cmap('Continent', palette=colors, factors=data['Continent'].unique()))

basic_bar.xgrid.grid_line_color = None
basic_bar.y_range.start = 0


#---------------------------------------------
#      Graph 2 - prices across terrian
#---------------------------------------------

text = """However, prices are often deceptive and further extrapolation is needed to understand which deals that give you the best bang for your buck. Afterall, just because you bought
an expensive lift ticket doesn't guarantee that you're getting better conditions or terrain, than a cheaper one. For example, if we take a popular metric for judging ski hills,
total vertical elevation (highest run to lowest run) we can begin to understand the general trends of pricing schemes relative to the offering of the given mountains, across the world."""
div1 = Div(text=text, styles=style)

source = ColumnDataSource(data)

basic_scatter = figure(x_axis_label="Price (USD $)", y_axis_label="Total Elevation (meters)", toolbar_location=None, tools="")
scatter = basic_scatter.scatter(x='Price', y='Height', source=source, size=8, alpha=0.7, color=factor_cmap('Continent', palette=colors, factors=data['Continent'].unique()))

legend = Legend(items=[LegendItem(label=dict(field="Continent"), renderers=[scatter])])
basic_scatter.add_layout(legend, 'right')

#p.background_fill_color = '#181919'

text = """Unsurprisingly, the North American resorts are substantially more expensive that all other continents. They also happen to have the most variability in ticket price. The
European continent, with it's famed and historic ski resorts, interestingly has more terrain to offer at a substantially reduced price compared to the shorter North American Resorts.
And while not as numerous, the South American and Asian resorts appear to offer cheaper deals than both the North American and European resorts, with a large amount of terrain provided for the low cost.
<br>
<br>
<br>
The purpose of this brief example was to highlight the variability in skiing around the world, while pointing out the differences in what these mountains might provide.
Everyone has their own vision of a perfect day on the hill, and the following tools are provided to further analyze data statistics on ski resorts and their offerings. Ultimately, this allows for a data driven
comparison across a wide range of whatever factors you deem to be the most important to you in your skiing experience - applied immediately to a vast selection of some of the many mountains the world has to offer."""
div2 = Div(text=text, styles=style)

#---------------------------------------------
#              Sliders + List
#---------------------------------------------

#value sliders - inverted
price_slider = Slider(title="How important is affordability", start=0, end=10, step=1, value=5)
snowCannonSlider = Slider(title="How important is skiing on real snow", start=0, end=10, step=1, value=5)

#value sliders - normal
elevation_slider = Slider(title="How important is total vertical", start=0, end=10, step=1, value=5)
totalRun_slider = Slider(title="How important is the number of runs", start=0, end=10, step=1, value=5)
longestRunLength_slider = Slider(title="How important is run length", start=0, end=10, step=1, value=5)
numberOfLifts_slider = Slider(title="How important is the number of lifts", start=0, end=10, step=1, value=5) #total lifts column

country_select = MultiSelect(title="Select Countries:", height=300, options=['All'] + sorted(list(set(data['Country']))), value=['All'])

#categorical sliders (must not have - don't care - must have)
#TODO: have text explaining each of the numerical values here
childFriendly_slider = Slider(title="Child friendly", start=0, end=2, step=1, value=1)
snowpark_slider = Slider(title="Snowparks", start=0, end=2, step=1, value=1)
nightskiing_slider = Slider(title="Nightskiing", start=0, end=2, step=1, value=1)
summerskiing_slider = Slider(title="Summer skiing", start=0, end=2, step=1, value=1)

#---------------------------------------------
#        Data Formatting + Normalizing
#---------------------------------------------

# Code for Lat/Long conversion to Mercator: https://stackoverflow.com/questions/57178783/how-to-plot-latitude-and-longitude-in-bokeh
data['Longitude'] = data['Longitude'] * (6378137 * np.pi/180.0)
data['Latitude'] = np.log(np.tan((90 + data['Latitude']) * np.pi/360.0)) * 6378137

x_range_min = data['Longitude'].min() - 1700000
x_range_max = data['Longitude'].max() + 1700000
y_range_min = data['Latitude'].min() - 2000000
y_range_max = data['Latitude'].max() + 2000000

data['Avg snow cannons per run'] = data['Snow cannons'] / data['Total slopes']
# resets rankings
def adjust_weights():
    
    #weights adjusted for top normalization total to be around 100
    weights = {
    'Price_nrm': price_slider.value * 0.55,
    'Height_nrm': elevation_slider.value  * 0.55,
    'Total slopes_nrm': totalRun_slider.value  * 0.55,
    'Total lifts_nrm': numberOfLifts_slider.value  * 0.55,
    'Longest run_nrm': longestRunLength_slider.value  * 0.55,
    'Avg snow cannons per run_nrm': snowCannonSlider.value  * 0.55,
    }

    # apply weights and nomalize on scale from 0-100
    attributes = ['Price', 'Height', 'Total slopes', 'Total lifts', 'Longest run', 'Avg snow cannons per run']

    data['Normalized_Score'] = 0

    scaler = MinMaxScaler(feature_range=(0, 4))
    for x in attributes:
        new_column = f'{x}_nrm'
        values = data[x].values.reshape(-1, 1)

        if new_column == 'Avg snow cannons per run_nrm' or new_column == 'Price_nrm':
            normalized_values = scaler.fit_transform(values)
            data[new_column] = (4 - normalized_values.flatten()) * weights[new_column]
        else:
            normalized_values = scaler.fit_transform(values)
            if new_column == 'Longest run_nrm':
                normalized_values[normalized_values == 0.0] = 0.3
            data[new_column] = normalized_values.flatten() * weights[new_column]

        data['Normalized_Score'] += data[new_column]

    data['Rank'] = data['Normalized_Score'].rank(ascending=False)

adjust_weights()
color_mapper = linear_cmap(field_name='Rank', palette=Viridis256, high=min(data['Rank']), low=max(data['Rank']))

source = ColumnDataSource(data) # main source of data


#---------------------------------------------
#               Stacked Bars
#---------------------------------------------

#top_10 = data.nlargest(10, 'Normalized_Score').sort_values(by='Normalized_Score', ascending=True)
top_10 = data.sort_values(by='Normalized_Score', ascending=True)
resorts = top_10['Resort'].tolist()
sequences = ['Price_nrm', 'Height_nrm', 'Total slopes_nrm', 'Total lifts_nrm', 'Longest run_nrm', 'Avg snow cannons per run_nrm']
top_10 = top_10[['Resort', 'Price', 'Height', 'Total slopes', 'Total lifts', 'Longest run', 'Avg snow cannons per run', 'Price_nrm', 'Height_nrm',
                'Total slopes_nrm', 'Total lifts_nrm', 'Longest run_nrm', 'Avg snow cannons per run_nrm']].copy()
bar_source = ColumnDataSource(data=top_10)

#height was 600
new_height = 60*top_10.shape[0]
if (new_height < 300):
    new_height = 300

static_sbar = figure(y_range=resorts, width=1200, height=new_height, title="", toolbar_location=None, tools="", margin=(0, 50, 0, 50), sizing_mode="fixed")

static_sbar.hbar_stack(sequences, y='Resort', height=0.5, source=bar_source, color=Set1_6)
legend_items = [(seq.replace('_nrm', ''), [static_sbar.renderers[i]]) for i, seq in enumerate(sequences)]
legend = Legend(items=legend_items, location=(100, 20), label_text_font_size='10pt', label_standoff=4)
static_sbar.add_layout(legend, 'above')

for r in static_sbar.renderers:
    layer = r.name
    prefix = ""
    postfix = ""

    if layer == 'Price_nrm':
        prefix = "$"
    elif layer == 'Height_nrm':
        postfix = " m"
    elif layer == "Longest run_nrm":
        postfix = " km"


    hover = HoverTool(tooltips=[
        (f"{layer.replace('_nrm', '')}", f"{prefix}@{{{layer.replace('_nrm', '')}}}{postfix}")
    ], renderers=[r])

    static_sbar.add_tools(hover)

#---------------------------------------------
#                   Map
#---------------------------------------------

TOOLS = "reset,pan,wheel_zoom,box_select"
ski_map = figure(width=1200, height=600, x_range=Range1d(start=x_range_min + 100000, end=x_range_max - 100000, bounds=(x_range_min, x_range_max)), 
            y_range=Range1d(start=y_range_min + 100000, end=y_range_max - 100000, bounds=(y_range_min, y_range_max)),
           x_axis_type="mercator", y_axis_type="mercator", tools=TOOLS)
ski_map.add_tile("CartoDB Positron", retina=True)

s = ski_map.scatter(x='Longitude', y='Latitude', size=6, source=source, color=color_mapper, alpha=0.8)
color_bar = ColorBar(title="Rank", color_mapper=color_mapper['transform'], width=12, location=(0, 0))
ski_map.add_layout(color_bar, 'right')

hover_tool_nodes = HoverTool(renderers=[s], tooltips= [
        ("Rank", "@Rank"),
        ("Resort", "@Resort"),
        ("Day Ticket Price", "$@Price"),
        ("Total Elevation", "@Height m"),
        ("Num Runs", "@{Total slopes}"),
        ("Num Lifts", "@{Total lifts}"),
        ("Longest Run", "@{Longest run} km"),
        ("Avg Snow Cannons per Run", "@{Avg snow cannons per run}")
])
ski_map.add_tools(hover_tool_nodes)

#---------------------------------------------
#            Update + Handlers
#---------------------------------------------

used_box_select = False

# Functions for box select events
def set_bs_flag(event):
    global used_box_select
    used_box_select = True
    update_rankings(None, None, None)

def reset_bs_flag(attr, old, new):
    global used_box_select
    used_box_select = False
    update_rankings(None, None, None)

# Update functions
def update_rankings(attr, old, new):

    # update rankings
    adjust_weights()
    color_mapper = linear_cmap(field_name='Rank', palette=Viridis256, low=max(data['Rank']), high=min(data['Rank']))
    s.glyph.fill_color = color_mapper
    #s.glyph.line_color = color_mapper
    color_bar.color_mapper=color_mapper['transform']

    # updating the map + bar
    if used_box_select == True and source.selected.indices != []:
            selected_indices = source.selected.indices
            
            selected_data = {key: source.data[key][selected_indices] for key in source.data}
            map_filter_data = pd.DataFrame(selected_data)
    else:
        source.selected.indices = []
        if country_select.value == ['All']:
            map_filter_data = data
        else:
            map_filter_data = data[data['Country'].isin(country_select.value)]

    check_sliders = [childFriendly_slider, snowpark_slider, nightskiing_slider, summerskiing_slider]
    for x in check_sliders:
        if (x.value == 0):
            column_name = x.title
            map_filter_data = map_filter_data[map_filter_data[column_name] == 'No']
        if (x.value == 2):
            column_name = x.title
            map_filter_data = map_filter_data[map_filter_data[column_name] == 'Yes']
            

    #top_10 = map_filter_data.nlargest(10, 'Normalized_Score').sort_values(by='Normalized_Score', ascending=True)
    top_10 = map_filter_data.sort_values(by='Normalized_Score', ascending=True)
    resorts = top_10['Resort'].tolist()
    top_10 = top_10[['Resort', 'Price', 'Height', 'Total slopes', 'Total lifts', 'Longest run', 'Avg snow cannons per run', 'Price_nrm', 'Height_nrm',
                    'Total slopes_nrm', 'Total lifts_nrm', 'Longest run_nrm', 'Avg snow cannons per run_nrm']].copy()
    bar_source.data = top_10
    static_sbar.y_range.factors = resorts

    new_height = 60*top_10.shape[0]
    if (new_height < 300):
        new_height = 300

    

    static_sbar.height = new_height
    
    if used_box_select == False:
        source.data = map_filter_data

# Handlers
country_select.on_change('value', reset_bs_flag)

for widget in [price_slider, elevation_slider, totalRun_slider, longestRunLength_slider, numberOfLifts_slider, 
               snowCannonSlider, childFriendly_slider, snowpark_slider, nightskiing_slider, summerskiing_slider]:
    widget.on_change('value', update_rankings)


ski_map.on_event(SelectionGeometry, set_bs_flag)

#---------------------------------------------
#               Layout and style
#---------------------------------------------

layout = column(div0, basic_bar, div1, basic_scatter, div2, country_select, row(price_slider, elevation_slider, totalRun_slider), 
row(numberOfLifts_slider, longestRunLength_slider, snowCannonSlider), row(childFriendly_slider, snowpark_slider, nightskiing_slider, summerskiing_slider), ski_map, row(static_sbar))        
#column(row(), row())          

#output_file("viz.html")
#show(layout)

curdoc().add_root(layout)



#LEFTOVER CODE:

#for countries
#country_select = CheckboxGroup(labels=sorted(list(set(data['Country']))), active=[])

#def checkbox_handler(active):
    #print(f'Active checkboxes: {active}')

#country_select.on_change('active', lambda attr, old, new: checkbox_handler(new))

#other_items = ['Child Friendly', 'Snowpark', 'Europe', 'Asia', 'Oceania']
#continent_select = CheckboxGroup(labels=continents, active=[])

#def checkbox_handler(active):
    #TODO: Add continent sorting here
    #print(f'Active checkboxes: {active}')

#continent_select.on_change('active', lambda attr, old, new: checkbox_handler(new))