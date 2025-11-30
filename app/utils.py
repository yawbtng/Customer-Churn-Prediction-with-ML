import plotly.graph_objects as go

def create_gauge_chart(probability):
    # Determine color based on churn probability
    if probability < 0.3:
        color = "green"
    elif probability < 0.6:
        color = "yellow"
    else:
        color = "red"

    # Create a gauge chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={
                'x': [0, 1],
                'y': [0, 1]
            },
            title={
                'text': "Churn Probability",
                'font': {
                    'size': 24,
                    'color': 'white'
                }
            },
            number={'font': {
                'size': 40,
                'color': 'white'
            }},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': 'white'
                },
                'bar': {
                    'color': color
                },
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 2,
                'bordercolor': 'white',

                'steps': [
                    {
                        'range': [0, 30],
                        'color': "rgba(0, 255, 0, 0.3)"
                    },
                    {
                        'range': [30, 60],
                        'color': "rgba(255, 255, 0, 0.3)"
                    },
                    {
                        'range': [60, 100],
                        'color': "rgba(255, 0, 0, 0.3)"
                    },
                ],

                'threshold': {
                    'line': {
                        'color': 'white',
                        'width': 4
                    },
                    'thickness': 0.75,
                    'value': 100
                }
            }
        )
    )

    # Update chart layout
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        width=400,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def create_model_probability_chart(probabilities):
    models = list(probabilities.keys())
    probs = list(probabilities.values())

    fig = go.Figure(
        data=[
            go.Bar(
                y=models,
                x=probs,
                orientation='h',
                text=[f"{p:.2%}" for p in probs],
                textposition='auto'
            )
        ]
    )

    fig.update_layout(
        title='Churn Probability by Model',
        yaxis_title='Models',
        xaxis_title='Probability',
        xaxis=dict(tickformat='.0%', range=[0, 1]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def calculate_customer_percentiles(customer_values, df):
    """
    Calculate percentiles for customer metrics compared to all customers in the dataset.
    
    Args:
        customer_values: dict with keys: CreditScore, Balance, EstimatedSalary, Tenure, NumOfProducts
        df: DataFrame with all customer data
    
    Returns:
        dict with metric names as keys and percentile values (0-100) as values
    """
    percentiles = {}
    
    # Calculate percentile for each metric
    for metric, value in customer_values.items():
        if metric in df.columns:
            # Calculate what percentile this value falls into
            percentile = (df[metric] <= value).sum() / len(df) * 100
            percentiles[metric] = percentile
    
    return percentiles


def create_customer_percentiles_chart(percentiles):
    """
    Create a horizontal bar chart showing customer percentiles for various metrics.
    
    Args:
        percentiles: dict with metric names as keys and percentile values (0-100) as values
    
    Returns:
        plotly figure object
    """
    # Order metrics as shown in screenshot
    metric_order = ['NumOfProducts', 'Balance', 'EstimatedSalary', 'Tenure', 'CreditScore']
    
    # Filter to only include metrics that exist in percentiles
    metrics = [m for m in metric_order if m in percentiles]
    values = [percentiles[m] for m in metrics]
    
    # Create horizontal bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                y=metrics,
                x=values,
                orientation='h',
                marker=dict(
                    color='rgba(100, 200, 255, 0.8)',
                    line=dict(color='rgba(100, 200, 255, 1.0)', width=1)
                ),
                text=[f"{v:.0f}%" for v in values],
                textposition='outside'
            )
        ]
    )
    
    # Update layout to match screenshot style (dark theme)
    fig.update_layout(
        title={
            'text': 'Customer Percentiles',
            'font': {'size': 20, 'color': 'white'},
            'x': 0.05,
            'y': 0.95
        },
        xaxis=dict(
            title=dict(text='Percentile', font=dict(color='white')),
            tickfont={'color': 'white'},
            range=[0, 100],
            tickmode='linear',
            tick0=0,
            dtick=10,
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis=dict(
            title=dict(text='Metric', font=dict(color='white')),
            tickfont={'color': 'white'},
            showgrid=False
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=350,
        margin=dict(l=100, r=20, t=60, b=40),
        showlegend=False
    )
    
    return fig
