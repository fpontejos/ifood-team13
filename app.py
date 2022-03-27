import dash
from dash import dash_table, html, dcc

#import dash_core_components as dcc
#import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



####################################################################################################################
## Wrangle the data
####################################################################################################################

data_path = 'https://raw.githubusercontent.com/fpontejos/ifood-team13/main/data/food_recipes.csv'
data_path2 = 'data/food_recipes.csv'
recipe_data = pd.read_csv(data_path2, nrows=1000)


recipe_data.drop(columns=['url', 'record_health', 'vote_count', 'author'], inplace=True)
recipe_data.dropna(inplace=True)
recipe_data.drop_duplicates(keep='first', inplace=True)
recipe_data.reset_index(inplace=True, drop=True)

recipe_data['ingredients_xpl'] = recipe_data['ingredients']
recipe_data['ingredients'] = recipe_data['ingredients'].apply(lambda x: x.replace('|', ' '))

recipe_data['tags'] = recipe_data['tags'].apply(lambda x: x.replace('|', ' '))
recipe_data['prep_time'] = recipe_data['prep_time'].str.split(' ').str[0].astype(int)
recipe_data['cook_time'] = recipe_data['cook_time'].str.split(' ').str[0].astype(int)


ingredients_index = recipe_data.loc[:,['ingredients_xpl']]
ingredients_index['ingredients'] = ingredients_index['ingredients_xpl'].apply(lambda x: x.split('|')).apply(lambda x:[(str.lower(i)) for i in x])
ingredients_index.drop(columns=['ingredients_xpl'], inplace=True)

ingredients_index['recipe_id'] = recipe_data.index
ingredients_index['recipe_name'] = recipe_data['recipe_title']
ingredients_index = ingredients_index.explode('ingredients').reset_index(drop=True)


unique_ingredients = pd.DataFrame(ingredients_index['ingredients'].unique().tolist(), 
                                  columns=['ingredients'])

######################################################Functions##############################################################



def get_recipe_from_index(df, i):
    return df.iloc[i,:]

def get_top_similar(simx, k, i, df):
    """
    simx = similarity matrix
    k = number of results to return
    i = index of recipe to compare
    """
    similar_recipes = list(enumerate(simx[i]))
    sorted_similar = sorted(similar_recipes, key=lambda x:x[1], reverse=True)
    top_k = sorted_similar[1:k+1]
    

    top_k_df = pd.DataFrame(columns=df.columns)
    top_k_scores = []
    top_k_index = []
    for i in top_k:
        top_k_df = top_k_df.append(get_recipe_from_index(df,i[0]), ignore_index=True)
        top_k_scores.append(i[1])
        top_k_index.append(i[0])

    top_k_df['Score'] = top_k_scores
    top_k_df['id'] = top_k_index
    top_k_df['Score'] = round(top_k_df['Score']*100,2)

    return top_k_df
    
def combined_features(row):
    combi = row['course']+" "+row['cuisine']+" "+row['diet']+" "+row['ingredients']+" "+row['tags']+" "+row['category']
    combi.replace(" Recipes", "")
    return combi

recipe_data["combined_features"] = recipe_data.apply(combined_features, axis =1)

#get_top_similar(cosine_sim_df, 2, 2493, data)    

cv = CountVectorizer()
count_matrix = cv.fit_transform(recipe_data["combined_features"])

cosine_sim_df = pd.DataFrame(cosine_similarity(count_matrix))


######################################################Data##############################################################


######################################################Interactive Components############################################

ing_options = [dict(label=ingredient, value=ingredient) for ingredient in unique_ingredients['ingredients']]

#prop options
cuisine_options = [dict(label=cuis, value=cuis) for cuis in pd.unique(recipe_data['cuisine'])]
cat_options = [dict(label=cat, value=cat) for cat in pd.unique(recipe_data['category'])]
diet_options = [dict(label=diet, value=diet) for diet in pd.unique(recipe_data['diet'])]
course_options = [dict(label=cour, value=cour) for cour in pd.unique(recipe_data['course'])]

gaps_str_prep = ['0-10','10-30','30-60','60-100','100-'+str(recipe_data['prep_time'].max())]
gaps_str_cook = ['0-10','10-30','30-60','60-100','100-'+str(recipe_data['cook_time'].max())]

prep_options = [dict(label=stri, value=stri) for stri in gaps_str_prep]
cook_options = [dict(label=stri, value=stri) for stri in gaps_str_cook]
##prop options over
#
dropdown_ingredient = dcc.Dropdown(
       id='ing_drop',
       options=ing_options,
       multi=True
   )

##prop options UI
dropdown_cuisine = dcc.Dropdown(
       id='cuisine_drop',
       options=cuisine_options,
       multi=True
   )

dropdown_cat = dcc.Dropdown(
       id='cat_drop',
       options=cat_options,
       multi=True
   )

dropdown_diet = dcc.Dropdown(
       id='diet_drop',
       options=diet_options,
       multi=True
   )

boxes_course = dcc.Checklist(
       id='course_check',
       options=course_options
   )

slider_rating = dcc.Slider(
       id='rating_slider',
       min=0,
       max=recipe_data['rating'].max(),
       marks={str(i): '{}'.format(str(i)) for i in
              [0,1,2,3,4,5]},
       step=0.01
   )

boxes_prep = dcc.Checklist(
       id='prep_check',
       options=prep_options
   )

boxes_cook = dcc.Checklist(
       id='cook_check',
       options=cook_options
   )
##prop options UI over
#
#
recipe_table = dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": 'Recipe', "id": 'recipe_name', "deletable": False, "selectable": True}

        ],
        #data=df.to_dict('records'),
        editable=False,
        #filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable=False,
        row_selectable=False,
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 1,
        page_size= 10,
    )
    
similar_table = dash_table.DataTable(
        id='datatable-similar',
        columns=[
            {"name": 'Score', "id": 'Score', "deletable": False, "selectable": True},
            {"name": 'Recipe', "id": 'recipe_title', "deletable": False, "selectable": True},

        ],
        #data=df.to_dict('records'),
        editable=False,
        #filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable=False,
        row_selectable=False,
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 1,
        page_size= 10,
    )

##################################################APP###################################################################

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([

    html.H1('iCook'),

    html.Label('Ingredients In My Pantry:'),
    dropdown_ingredient,

    #prop options display
    html.Label('Cuisine Choice'),
    dropdown_cuisine,

    html.Label('Category Choice'),
    dropdown_cat,

    html.Label('Diet Choice'),
    dropdown_diet,

    html.Label('Course Choice'),
    boxes_course,

    html.Label('Rating Slider'),
    slider_rating,
    
    html.Label('Preparation Time Interval'),
    boxes_prep,

    html.Label('Cooking Time Interval'),
    boxes_cook,
    #prop options display over

    html.Div([
        html.Div([
            html.Div([
                html.Label('Recipe Choice'),
                recipe_table
            ], id='recipelist_box'),
            html.Div([
                html.Label('Similar Recipes'),
                similar_table,
                html.Div(id='similar_out')

            ], id='similar_box')
        ], className="is-half"),

        html.Div([
            html.Div([
               html.Div(id='recipe_title'),
                
                html.Label('Description'),
                html.Div(id='recipe_description'),

                html.Label('Ingredients'),
                html.Div(id='ingredients_out'),


            ])
        ], className="is-half")
    ], className="is-whole"),




])


######################################################Callbacks#########################################################

@app.callback(
#    Output('dd-output-container', 'children'),
    Output('datatable-interactivity', 'value'),
    Input('ing_drop', 'value')
)
def update_output(value):
    return value


@app.callback(
    Output('datatable-interactivity', 'data'),
    [Input('datatable-interactivity', 'page_current'),
     Input('datatable-interactivity', 'page_size'),
     Input('datatable-interactivity', 'sort_by'),
     Input("ing_drop", "value"),
     Input("cuisine_drop", "value"),
     Input("cat_drop", "value"),
     Input("diet_drop", "value"),
     Input("course_check", "value"),
     Input("rating_slider", "value"),
     Input("prep_check", "value"),
     Input('cook_check', 'value'),
     ])
def update_table(page_current, page_size, sort_by, filter_string, cuis,cat,diet,cour,rate,prep,cook):
    # Filter
    dff = ingredients_index.loc[ingredients_index['ingredients'].isin(filter_string),['recipe_name', 'recipe_id']]
    dff.drop_duplicates(inplace=True)

    return dff.iloc[
           page_current * page_size:(page_current + 1) * page_size
           ].to_dict('records')

@app.callback(
    Output('ingredients_out', 'children'),
    Input('datatable-interactivity', 'active_cell'),
    State('datatable-interactivity', 'data')
)
def getActiveCell(active_cell, data):
    if active_cell:
        col = active_cell['column_id']
        row = active_cell['row']
        desc_data = get_recipe_from_index(recipe_data, data[row]["recipe_id"])

        ing_list = html.Ul([html.Li(x) for x in desc_data['ingredients_xpl'].split('|')])

        ing_checklist = html.Div([
            dcc.Checklist(
                [x for x in desc_data['ingredients_xpl'].split('|')],
                [x for x in desc_data['ingredients_xpl'].split('|')]            
            ),
            html.Button("Add to Cart")
        ])

        return ing_checklist
    
    return 


@app.callback(
    Output('recipe_description', 'children'),
    Input('datatable-interactivity', 'active_cell'),
    State('datatable-interactivity', 'data')
)
def getRecipeDescription(active_cell, data):
    if active_cell:
        col = active_cell['column_id']
        row = active_cell['row']
        desc_data = get_recipe_from_index(recipe_data, data[row]["recipe_id"])
        return desc_data['description']
    return 



@app.callback(
    Output('recipe_title', 'children'),
    Input('datatable-interactivity', 'active_cell'),
    State('datatable-interactivity', 'data')
)
def getRecipeTitle(active_cell, data):
    if active_cell:
        row = active_cell['row']
        desc_data = get_recipe_from_index(recipe_data, data[row]["recipe_id"])
        return html.H3(desc_data['recipe_title'])
    return 



@app.callback(
    Output('datatable-similar', 'data'),
    Input('datatable-interactivity', 'active_cell'),
    State('datatable-interactivity', 'data')
)
def getRecipeSimilar(active_cell, data):
    if active_cell:
        row = active_cell['row']
        recipe_id = data[row]["recipe_id"]
        desc_data = get_recipe_from_index(recipe_data, recipe_id)
        similar_df = get_top_similar(cosine_sim_df, 5, recipe_id, recipe_data)
        return similar_df.to_dict('records')
    return 


if __name__ == '__main__':
    app.run_server(debug=True)
