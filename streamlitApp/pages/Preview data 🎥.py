#third party imports
import streamlit as st
from pathlib import Path
import pandas as pd
#from streamlit_stl import stl_from_file
import numpy as np
import re

#custom imports
from fabricad.constants import PATHS



def extract_rank(path_str):
    return float(re.findall(r'\d+(?:\.\d+)?', path_str)[-1])

st.set_page_config(layout="wide")

path_to_samples = PATHS.PREVIEW_DATA

samples = sorted(list(path_to_samples.iterdir()))
samples = [elem for elem in samples if elem.is_dir()]

samplesTotal = len(samples)




st.title('Sample Viewer für synthetische Arbeitsplandaten')
l,m,r = st.columns([1,2,1])

with m:
    totalToShow = st.number_input("Total of samples to display:", min_value=1, max_value=1000, step=5, value=10)

# to get different images/media in the rows and columns, have a systematic 
# way to label your images/media. For mine, I have used row_{i}_col_0/1
# also note that different media types such as an audio or video file you 
# will need to have that whole column as an audio or video column!

#Setup table
cols = st.columns([1,1,2])
# first column of the ith row
cols[0].subheader("ID")
cols[1].subheader("geo - preview")
cols[2].subheader("plan")    

for i in range(0,totalToShow):

    st.divider()

    if i==samplesTotal:
        break

    #load sample data
    sample_path = samples[i]

    id = sample_path.name


    #show_substeps = cols[2].checkbox("Zeige Featureebene", key = f"checkbox_{i}", value = False)

    #if show_substeps:
    interim_states = list(sample_path.joinpath("interim").glob("*.png"))  #key = lambda x: extract_rank(x))
    substeps_states = list(sample_path.joinpath("interim/substeps/").glob("*.png"))  #key = lambda x: extract_rank(x))

    final_state_preview = list((sample_path / "interim").glob("*.png"))[0]

    assert final_state_preview is not None, "No final state preview found!"

    all_state_paths = interim_states + substeps_states
    #st.write(all_state_paths)
    all_states = {}

    for elem in all_state_paths:
        all_states[extract_rank(str(elem))]=elem 


    #else:
    #    interims = [elem for elem in sample_path.joinpath("interim").glob("*.png") if True]


    try:
        plan = pd.read_csv(str(sample_path.joinpath("plan.csv")), sep=";", index_col=False, decimal='.', dtype={'Materialnummer' : str, 'Nr.' : str, ' Qualifikation' : 'uint8', "Kosten[($)]" : float})
        features_metadata = pd.read_csv(str(sample_path.joinpath("interim/substeps/features.csv")), sep=";", index_col=False, decimal='.', dtype={'Subschritt' : str ,'Materialnummer' : str, 'Nr.' : str, "Kosten[($)]" : float})

        plan = plan.drop(0).drop("Qualifikation", axis=1) # drop init row
        

    except Exception as e:
        st.warning(f"Could Not read plan! with exception {e}")
        break
        

    cost = np.round(plan['Kosten[($)]'].sum(),2)
    manuf_time = np.round(plan['Dauer[min]'].sum(),2)

    metaplan = pd.read_csv(str(samples[i].joinpath("plan_metadata.csv")), sep=";", index_col=False, decimal=',')

    rmid = metaplan.Abmaße.loc[0].replace("x", "")


    cols = st.columns([1,1,2])
    # first column of the ith row

    cols[0].subheader("Arbeitsplan:")
    cols[0].markdown(f"Artikel: {metaplan.Kurzbezeichnung.loc[0][:13]}-{id}")
    cols[0].markdown(f"Artikel-Nr: {id}")
    cols[0].markdown(f"(Hz-Nr.): RM EN {rmid}")
    cols[0].markdown(f"Material: {metaplan.Material.loc[0]}")

    cols[0].markdown(f"Abmaße: {metaplan.Abmaße.loc[0]}")

    cols[0].subheader(f"Abschätzungen:")

    cols[0].markdown(f"Herstellungskosten: {cost} EUR")
    cols[0].markdown(f"Herstellungszeit  : {manuf_time} min.")

    show_features_mata = cols[0].checkbox("Featureebene", key = f"Checkbox_{i}")
    show_comp = cols[0].checkbox("Zusammensetzung Kosten", key = f"Checkbox_comp_{i}")





    step_selector = cols[1].slider("Schrittauswahl:", min_value=1, max_value=len(plan), value=len(interim_states), key = f"Slider_{i}")

    color = "lightgreen"
    plan = plan.style.format("{:.2f}", subset=plan.select_dtypes(include=['number']).columns)
    plan = plan.map(
    lambda _: f"background-color: {color}", subset=(plan.index[[step_selector-1]],)
    )
  
    
    cols[2].dataframe(plan, hide_index = False)  

    if show_comp:

        #try to load data
        try:
            path_comp_data_time = sample_path / "interim/substeps" / f"comp_time_step_{step_selector}.csv"
            path_comp_data_cost = sample_path / "interim/substeps" / f"comp_cost_step_{step_selector}.csv"

            comp_data_time = pd.read_csv(path_comp_data_time.as_posix(),sep = ";")
            comp_data_cost = pd.read_csv(path_comp_data_cost.as_posix(),sep = ";")

            cols[2].dataframe(comp_data_time)
            cols[2].dataframe(comp_data_cost)

        except Exception as e:
            #cols[2].write(e)
            pass

    if show_features_mata:
        #right section
        meta_plan = features_metadata[features_metadata['Arbeitsschritt']==step_selector]

        
        if len(meta_plan)>0:
            meta_plan = meta_plan.drop(["Arbeitsschritt", "Kosten[($)]"], axis = 1)
            #middle section
            if len(meta_plan)>1:
                select_meta_step = cols[1].slider("Featureauswahl:", min_value=1, max_value=len(meta_plan), value=1, key = f"Slider_Feature_{i}")
            else:
                select_meta_step=1
            
            try:    
                cols[1].markdown("Vorschau Werkstück:")
                cols[1].image(str(all_states[float(f"{step_selector}.{select_meta_step}")]), use_container_width=True)
            except:
                cols[1].image(final_state_preview.as_posix(), use_container_width=True)
                cols[1].markdown(":green[Hinweis: Keine Vorschau der Zwischenschritte verfügbar.]")


            color = "lightgray"
            meta_plan = meta_plan.style.format("{:.2f}", subset=meta_plan.select_dtypes(include=['number']).columns).map(lambda _: f"background-color: {color}", subset=(meta_plan.index[[select_meta_step-1]],))

            cols[2].dataframe(meta_plan, hide_index=True)
        else:
            cols[2].markdown(":red[Arbeitsschritt hat keine geometrischen Features hinzugefügt!]")
            cols[1].markdown("Vorschau Werkstück:")
            #check if image is available
            try:
                key = all_states[float(step_selector)]
                cols[1].image(str(key), width=5)
            except KeyError:
                cols[1].image(final_state_preview.as_posix(), use_container_width=True)

    else:
        cols[1].markdown("Vorschau Werkstück:")
        try:
            key = all_states[float(step_selector)]
            cols[1].image(str(key), width=5)
        except KeyError:
            cols[1].image(final_state_preview.as_posix(), use_container_width=True)

st.divider()
left, _, _ = st.columns(3)
with left:
    st.markdown(f":gray[show ({i}/{samplesTotal})]")  
    