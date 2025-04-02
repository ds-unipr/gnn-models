from graphs_plots import plot_single_model, plot_selected_models


##### CHANGE model_name, USE TO FORCE A plot_single_model ON OLD MODEL #####

#model_name = 'MLP'
#plot_single_model(model_name=model_name)

#######################################



##### INSERT models, USE TO FORCE A plot_selected_models ON OLD MODELS #####

model_names = ["modelD5_100epochs", "MLP"]
plot_selected_models(model_names=model_names)

#######################################