import streamlit as st
import json
import pandas as pd
import pycaret.clustering #import load_model, predict_model
import plotly.express as px
st.title("Znajdź znajomych")

DATA = 'welcome_survey_simple_v2.csv'
MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

@st.cache_data 
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding="utf-8") as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("pomożemy ci znaleźć osoby, które mają podbne zainteresowania")
    age = st.selectbox("Wiek", ['<18','18-24','25-34','35-44','45-54','55-64','>=64','unknown'])
    edu_level = st.selectbox("Wykształcenie",['Podstawowe','Średnie','Wyższe'])
    fav_animals=st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty','Inne', 'Koty i Psy'])
    fav_place=st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta']) 

person_df = pd.DataFrame([
    {
        'age': age,
        'edu_level':edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender
    }
])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]


st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df=all_df[all_df["Cluster"]==predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

##
##Wykres kołowy
##
st.header("Procentowy podział w grupie")

feature_option = {
    "Wiek": "age",
    "Wykształcenie": "edu_level",
    "Ulubione zwierzęta": "fav_animals",
    "Ulubione miejsce": "fav_place",
    "Płeć": "gender",
}

feature_label = st.selectbox(
    "Wybierz cechę:",
    options=list(feature_option.keys())
)

feature = feature_option[feature_label]

fig = px.pie(
    same_cluster_df,
    names=feature,
    color=feature,    
)
st.plotly_chart(fig, use_container_width=True)

##
##Histogramy cech
##

st.header("Histogramy cech")
feature_option = {
    "Wiek": "age",
    "Wykształcenie": "edu_level",
    "Ulubione zwierzęta": "fav_animals",
    "Ulubione miejsce": "fav_place",
    "Płeć": "gender",
}

feature_label = st.selectbox(
    "Wybierz cechę do analizy (histogram):",
    options=list(feature_option.keys())
)

feature = feature_option[feature_label]

# sortowanie tylko dla wieku — opcjonalne
df_plot = same_cluster_df.sort_values(feature) if feature == "age" else same_cluster_df

fig = px.histogram(
    df_plot,
    x=feature,
    color=feature,
)

fig.update_layout(
    title=f"Rozkład cechy: {feature_label}",
    xaxis_title=feature_label,
    yaxis_title="Liczba osób",
)

st.plotly_chart(fig, use_container_width=True)

##
##Heatmapa zależnosci między cechami
##

st.header("Zależności między cechami")

feature_option = {
    "Wiek": "age",
    "Wykształcenie": "edu_level",
    "Ulubione zwierzęta": "fav_animals",
    "Ulubione miejsce": "fav_place",
    "Płeć": "gender",
}

col1, col2 = st.columns(2)

with col1:
    feature_label_x = st.selectbox(
        "Wybierz cechę (oś X):",
        options=list(feature_option.keys()),
        key="heatmap_x"
    )

with col2:
    feature_label_y = st.selectbox(
        "Wybierz cechę (oś Y):",
        options=list(feature_option.keys()),
        key="heatmap_y"
    )

feature_x = feature_option[feature_label_x]
feature_y = feature_option[feature_label_y]

# Crosstab
cross = pd.crosstab(same_cluster_df[feature_x], same_cluster_df[feature_y])

# Heatmap
fig = px.imshow(
    cross,
    text_auto=True,
    color_continuous_scale="Blues",
)

fig.update_layout(
    title=f"Zależność: {feature_label_x} vs {feature_label_y}",
    xaxis_title=feature_label_y,
    yaxis_title=feature_label_x,
)

st.plotly_chart(fig, use_container_width=True)


