import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def overall_modelxmodel_clustermap(df,
                                   metric,
                                   out_fpath,
                                    cmap="Reds",
                                     figsize=(8,10)):
    g = sns.clustermap(data=df,
                    cmap=cmap,
                    figsize=figsize,
                    #sizes=(50,400),
                   )
    g.fig.suptitle(metric+ " (Output concordance)") 
    
    #ax.t(metric)
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    #ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05))

    g.savefig(out_fpath, bbox_inches="tight")
    
def overall_modelxmodel_dotplot(df,
                                   metric,
                                   out_fpath,
                                     figsize=(8,6)):
    size = f"mean_{metric}"
    hue = f"concordance_{metric}"

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(data=df,
                    x="Model2",
                    y="Model1",
                    size=size,
                    hue=hue,
                    palette="Reds",
                    sizes=(25,400),
                   )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05))
    ax.figure.savefig(out_fpath, bbox_inches="tight")

def visualize_pairwise_win_fraction(elo_scores, title, max_num_models=30):
    #row_beats_col = compute_pairwise_win_fraction(battles, max_num_models)
    fig = px.imshow(elo_scores, color_continuous_scale='RdBu_r', # Blues, tempo
                    text_auto=".2f", title=title,)
    fig.update_layout(xaxis_title="Model B: Loser",
                  yaxis_title="Model A: Winner",
                  xaxis_side="bottom",
                      height=900,
                      width=900,
                  title_y=0.89,
                      title_x=0.55)
    fig.update_xaxes(tickangle=-30)
    fig.update_traces(hovertemplate=
                  "Model A: %{y}<br>Model B: %{x}<br>Fraction of A Wins: %{z}<extra></extra>")

    return fig


    