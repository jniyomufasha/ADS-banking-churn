from distutils.log import debug
from http import server
from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


#Importing and preparing the data
banking_data = pd.read_csv('data/banking_churn.csv')

def preprocessing_func(df):
    def scaling_func(column, value):
        col_min = banking_data[column].min()
        col_max = banking_data[column].max()
        return (value-col_min)/(col_max - col_min)
    
    x_test = df.copy()
    
    columns_to_scale = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'NumOfProducts']
    for column in columns_to_scale:
        x_test[column] = x_test[column].apply(lambda x: scaling_func(column, x))
    
    x_test['Geography'] = pd.Categorical(x_test['Geography'], categories=banking_data['Geography'].unique(), ordered=True)
    x_test['Gender'] = pd.Categorical(x_test['Gender'], categories=banking_data['Gender'].unique(), ordered=True)
    test_dummies = pd.get_dummies(x_test[['Geography', 'Gender']])
    x_test = pd.concat([x_test, test_dummies], axis = 1)
    x_test.drop(['Geography', 'Gender'], axis = 1, inplace = True)
    return x_test


app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.title = 'Banking Churn'

app.layout = dbc.Container([
    dbc.Row([
        dbc.Card([
            dbc.CardBody(html.H1('BANKING CHURN', style={'text-align': 'center'}))
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P('Enter credit score', style={'display': 'inline'}),
                    dcc.Input(type='number', debounce=True, min=0, step=1, style={'width': '50%', 'float': 'right'}, id='credit')
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.P('Choose country', style={'display': 'inline'}),
                    dcc.Dropdown(
                        options=banking_data['Geography'].unique(),
                        placeholder='Choose country',
                        style={'color': 'black', 'float': 'right', 'width': '71%'},
                        id = 'country'
                    )
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.P('Choose gender', style={'display': 'inline'}),
                    dcc.Dropdown(
                        options=['Male', 'Female'],
                        placeholder='Choose gender',
                        style={'color': 'black', 'float': 'right', 'width': '71%'},
                        id = 'gender'
                    )
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.P('Enter age', style={'display': 'inline'}),
                    dcc.Input(type='number', debounce=True, min=0, step=1, style={'width': '50%', 'float': 'right'}, id = 'age')
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.P('Enter tenure', style={'display': 'inline'}),
                    dcc.Input(type='number', debounce=True, min=0, step=1, style={'width': '50%', 'float': 'right'}, id = 'tenure')
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.P('Enter balance', style={'display': 'inline'}),
                    dcc.Input(type='number', debounce=True, style={'width': '50%', 'float': 'right'}, id = 'balance')
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.P('Enter number of products', style={'display': 'inline'}),
                    dcc.Input(type='number', debounce=True, min=0, step=1, style={'width': '50%', 'float': 'right'}, id = 'prods')
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.P('Has a credit card?', style={'display': 'inline'}),
                    dcc.Dropdown(
                        options=['Yes', 'No'],
                        placeholder='Has a credit card?',
                        style={'color': 'black', 'float': 'right', 'width': '71%'},
                        id = 'card'
                    )
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.P('Is an active member?', style={'display': 'inline'}),
                    dcc.Dropdown(
                        options=['Yes', 'No'],
                        placeholder='Is an active member?',
                        style={'color': 'black', 'float': 'right', 'width': '71%'},
                        id = 'active'
                    )
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.P('Enter estimated salary', style={'display': 'inline'}),
                    dcc.Input(type='number', debounce=True, style={'width': '50%', 'float': 'right'}, id = 'salary')
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.Button('Clear inputs', id='clear', n_clicks=0, style={'width': '50%',})
                ])
            ])
        ], width=7, style={'padding-left': 0}),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("""Based on the customer's banking behaviour, we are predicting whether they are most\
                            likely leaving the bank or staying."""),
                    html.P("""Fill the customer banking parameters on the left to get the customer's predictions."""),
                    html.P("""The logic for the predictions can be found on the link down here"""),
                    html.A('Github Link', href='https://github.com/jniyomufasha/ADS-banking-churn', target='_blank')
                ])
            ]),
            dbc.Card([
                dbc.CardHeader(
                    html.H2('PREDICTION RESULTS', style={'color':'black', 'text-align': 'center'}),
                    style={'background-color': 'white'}
                ),
                dbc.CardBody([
                    html.P('That customer has 80% probability of leaving the bank.')
                ], id = 'output')
            ], className='mt-5')
        ], width=5, style={'padding-right': 0})
    ], className='mt-3')
])

@app.callback(
    [
        Output(component_id='output', component_property='children')
    ],
    [
        Input(component_id='credit', component_property='value'),
        Input(component_id='country', component_property='value'),
        Input(component_id='gender', component_property='value'),
        Input(component_id='age', component_property='value'),
        Input(component_id='tenure', component_property='value'),
        Input(component_id='balance', component_property='value'),
        Input(component_id='prods', component_property='value'),
        Input(component_id='card', component_property='value'),
        Input(component_id='active', component_property='value'),
        Input(component_id='salary', component_property='value'),
    ]
)
def get_result(credit, country, gender, age, tenure, balance, prods, card, active, salary):
    my_dict = {
        'CreditScore': credit,
        'Geography': country,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': prods,
        'HasCrCard': card,
        'IsActiveMember': active,
        'EstimatedSalary': salary
    }
    has_null = False
    for k in my_dict.keys():
        if my_dict[k] is None:
            has_null = True
    if has_null:
        return [html.P('Please provide all features.')]
    def change_yes_no(value):
        if value == 'Yes':
            return 1
        elif value == 'No':
            return 0
    my_arr = pd.DataFrame([my_dict])
    my_arr['HasCrCard'] = my_arr['HasCrCard'].apply(lambda x: change_yes_no(x))
    my_arr['IsActiveMember'] = my_arr['IsActiveMember'].apply(lambda x: change_yes_no(x))
    x_test = preprocessing_func(my_arr)
    model = pickle.load(open('models/model.pkl', 'rb'))
    y_probs = model.predict_proba(x_test)
    y_preds = model.predict(x_test)
    if y_preds == 1:
        churn = 'leaving'
    elif y_preds == 0:
        churn = 'staying in'
    output = f'That client has {max(y_probs[0])*100:.2f}% probability of {churn} the bank'
    return [html.P(output)]

@app.callback(
    [
        Output(component_id='credit', component_property='value'),
        Output(component_id='country', component_property='value'),
        Output(component_id='gender', component_property='value'),
        Output(component_id='age', component_property='value'),
        Output(component_id='tenure', component_property='value'),
        Output(component_id='balance', component_property='value'),
        Output(component_id='prods', component_property='value'),
        Output(component_id='card', component_property='value'),
        Output(component_id='active', component_property='value'),
        Output(component_id='salary', component_property='value'),
    ],
    [
        Input(component_id = 'clear', component_property = 'n_clicks')
    ]
)
def clear_input(n_clicks):
    return [None,None,None,None,None,None,None,None,None,None]

if __name__ == '__main__':
    app.run_server()