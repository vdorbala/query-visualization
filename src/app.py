import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import torch
from transformers import AutoTokenizer, AutoModel
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import json
import random
import ast
from plotly.subplots import make_subplots
import re

from collections import defaultdict

from data import relationlist

def process_relationlist(relationlist):
    object_room = defaultdict(list)
    for relation in relationlist:
        parts = relation.split()
        # Check if the relationship is of type 2 (OBJECT RELATION ROOM)
        if parts[2] in ["kitchen", "livingroom", "bedroom", "bathroom"]:
            # Only add the object to the room's list if the relationship is of type 2
            object_room[parts[0]].append(parts[2])
    return object_room


# Initialize BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Function to generate embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].detach().numpy()

# Load csv file
df = pd.read_csv('gen_questions_v2.csv')

object_room = process_relationlist(relationlist)
object_list = list(set([obj for sublist in relationlist for obj in sublist.split() if obj not in ['kitchen', 'livingroom', 'bedroom', 'bathroom']]))
embeddings = {obj: get_embedding(obj) for obj in object_list}

colors = {
    'kitchen': {'normal': 'red', 'light': '#ff7f7f', 'dark': '#800000'},
    'livingroom': {'normal': 'green', 'light': '#7fff7f', 'dark': '#008000'},
    'bedroom': {'normal': 'blue', 'light': '#7f7fff', 'dark': '#000080'},
    'bathroom': {'normal': 'purple', 'light': '#ff7fff', 'dark': '#800080'},
}



# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Layout of the Dash app
app.layout = html.Div([
    html.Button('New Question', id='new-question', n_clicks=0),
    html.Div(id='question'),
    dcc.Graph(id='tsne-plot', style={"height" : "100vh", "width" : "100%"})
])

# app.layout = html.Div([
#     html.Div([
#         html.Div(id='question'),
#         html.Button('New Question', id='new-question', n_clicks=0)
#     ], style={'display': 'flex', 'justify-content': 'space-between'}),
#     dcc.Graph(id='tsne-plot', style={"height" : "100vh", "width" : "100%"})
# ])

rooms = ["kitchen", "livingroom", "bedroom", "bathroom"]

# Calculate t-SNE transformation for each room's objects and store in a dictionary
tsne_coordinates = {}
for room in rooms:
    room_objects = [obj for obj, room_list in object_room.items() if room in room_list]
    room_embeddings = np.array([embeddings[obj][0] for obj in room_objects])  # Ensure embeddings are 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(room_embeddings) - 1))
    transformed = tsne.fit_transform(room_embeddings)
    
    for obj, coord in zip(room_objects, transformed):
        tsne_coordinates[(obj, room)] = coord


@app.callback(
    [Output('tsne-plot', 'figure'), Output('question', 'children')],
    [Input('new-question', 'n_clicks')],
    [State('tsne-plot', 'figure')]
)
def update_plot(n_clicks, figure):
    # Choose a random question and associated states
    row = df.sample(1)
    question = row['Question'].values[0]
    answer = row['GPT Answer'].values[0]
    states = ast.literal_eval(row['Object-States'].values[0])
    relationships = row['Relationships'].values[0]

    if isinstance(relationships, str):
        relationships = [rel.strip() for rel in relationships.split(',')]
    else:
        relationships = ast.literal_eval(relationships)


    rooms = ["kitchen", "livingroom", "bedroom", "bathroom"]

    fig = make_subplots(rows=2, cols=2, subplot_titles=rooms, horizontal_spacing=0.01, vertical_spacing=0.01)

    highlighted_objects = set(states.keys()) # Objects to highlight (includes those from 'Object-States')

    # Define a regex pattern to match object names (alphanumeric characters and underscores)
    pattern = r'\b\w+\b'

    object_room_dict = defaultdict(list)

    for relationship in relationships:

        if relationship.startswith('[') and relationship.endswith(']'):
            # Convert the string to a list
            relationship_list = ast.literal_eval(relationship)
            for item in relationship_list:
                parts = item.split()
                match_1 = re.findall(pattern, parts[0])
                match_2 = re.findall(pattern, parts[2])

                object_room_dict[str(match_1[0])].append(match_2[0])

                highlighted_objects.add(match_1[0])
                highlighted_objects.add(match_2[0])

        else:
            parts = relationship.split()
            match_1 = re.findall(pattern, parts[0])
            match_2 = re.findall(pattern, parts[2])

            object_room_dict[str(match_1[0])].append(match_2[0])
            highlighted_objects.add(match_1[0])
            highlighted_objects.add(match_2[0])

    highlighted_objects = list(highlighted_objects)

    print("Object room dict is {}".format(object_room_dict))

    for i, room in enumerate(rooms):
        room_objects = [obj for obj, room_list in object_room.items() if room in room_list]

        # Draw arrows for relationships of type 1
        for relationship in relationships:
            parts = relationship.split()
            if parts[0] in room_objects and parts[2] in room_objects:
                coord1 = tsne_coordinates.get((parts[0], room), [0, 0])
                coord2 = tsne_coordinates.get((parts[2], room), [0, 0])
                fig.add_trace(go.Scatter(
                    x=[coord1[0], coord2[0], None],
                    y=[coord1[1], coord2[1], None],
                    mode='lines',
                    line=dict(color='gray', width=4),
                    hoverinfo='none',
                    showlegend=False
                ), row=i//2 + 1, col=i%2 + 1)

        for obj in room_objects:
            color = colors[room]['normal']  # Default color            
            size = 8
            opacity = 0.8
            text = None  # By default, don't show any text

            if obj in states.keys():
                if 'ON' in str(states.get(obj, "")) or 'OPEN' in str(states.get(obj, "")):
                    color = colors[room]['light']
                elif 'OFF' in str(states.get(obj, "")) or 'CLOSED' in str(states.get(obj, "")):
                    color = colors[room]['dark']

            # Check if the object is in the highlighted_objects list
            if obj in highlighted_objects:

                size = 25  # Highlight the object in the specified room
                opacity = 1

                if (obj in object_room_dict.keys()):

                    objrels = object_room_dict[obj]

                    for item in objrels:
                        if (item in rooms) and (room not in item):
                            size = 8
                            opacity = 0.8


            coord = tsne_coordinates.get((obj, room), [0, 0])

            fig.add_trace(go.Scatter(
                x=[coord[0]],
                y=[coord[1]],
                mode='markers',
                marker=dict(
                    color=color,
                    size=size,
                    opacity=opacity,
                    line=dict(
                        color='DarkSlateGrey',
                        width=2
                    )
                ),
                name=obj,
                hovertemplate = f'{obj}: {states.get(obj, "")}<extra></extra>',
                legendgroup=room,
                hoverlabel=dict(namelength=-1),
                showlegend=False
            ), row=i//2 + 1, col=i%2 + 1)

    # Add an invisible marker for each room to show in the legend
    for i, room in enumerate(rooms):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10),
            legendgroup=room,
            name=room,
            showlegend=True,
        ))

    fig.update_layout(title='Objects in rooms (t-SNE plot)', hovermode='closest')

    return fig, ['Question: {} GPT Answer: {}.'.format(question, answer), html.Br(), 'States: ', ', '.join(states), html.Br(), 'Relationships: ', ', '.join(relationships)]


if __name__ == '__main__':
    app.run_server(debug=True)
