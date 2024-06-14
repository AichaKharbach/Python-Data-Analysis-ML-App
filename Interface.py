from tkinter import Label, simpledialog, filedialog, messagebox, ttk, Entry
import customtkinter as ctk
import pandas as pd
import tkinter as tk
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pickle


# Variables globales pour stocker les données de test et le modèle
X_test = None
y_test = None
model = None

app = ctk.CTk()
app.title('Application de Python')
app.geometry('1000x600')

# Définir le style 
style = ttk.Style(app)
style.theme_use("default")
# Couleur de fond pour les lignes impaires
style.configure("Treeview", background="#D3D3D3", foreground="black", rowheight=25, fieldbackground="#D3D3D3")
# Couleur de fond pour les lignes paires
style.map('Treeview', background=[('alternate', '#E8E8E8')])
# Police et taille pour les en-têtes de colonnes
style.configure("Treeview.Heading", font=('Calibri', 10, 'bold'))
# Retirer les lignes de bordure
style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])

# Frame principal qui sera affiché après le clic sur 'Démarrer l'analyse'
main_frame = ctk.CTkFrame(app)

# Frame pour les boutons sur la gauche
left_frame = tk.Frame(main_frame, width=200)
left_frame.pack(side='left', fill='y', padx=20, pady=20)

# Frame pour afficher les données sur la droite
right_frame = tk.Frame(main_frame)
right_frame.pack(side='right', fill='both', expand=True, padx=20, pady=20)

# Ajout d'un Treeview à right_frame pour afficher les données sous forme de tableau
tree = ttk.Treeview(right_frame, style="Treeview")
tree.tag_configure('alternate', background='#E8E8E8') 
tree.pack(side='top', fill='both', expand=True)

# Fonction pour démarrer l'application
def start_analysis():
    start_frame.pack_forget()  # Cache le frame de départ
    main_frame.pack(fill='both', expand=True)  # Affiche le frame principal

# Fonction pour fermer l'application
def fermer_application():
    app.quit()  # ou app.destroy() pour détruire la fenêtre

# Fonction pour effacer le contenu du right frame
def effacer_contenu_right_frame():
    for widget in right_frame.winfo_children():
        widget.destroy()

# Fonction pour charger les données
def charger_donnees():
    global df, tree
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xls;*.xlsx"), ("All Files", "*.*")])
    if not file_path:
        return
    try:
        effacer_contenu_right_frame()  # Effacer le contenu de right_frame avant de créer le Treeview
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Format de fichier non pris en charge.")
        
        messagebox.showinfo("Chargement des données", "Les données ont été chargées avec succès!")
        # Créer le Treeview après avoir effacé le contenu de right_frame
        tree = ttk.Treeview(right_frame, style="Treeview")
        tree.tag_configure('alternate', background='#E8E8E8')
        tree.pack(side='top', fill='both', expand=True)

        tree["column"] = list(df.columns)
        tree["show"] = "headings"
        for column in tree["column"]:
            tree.heading(column, text=column)
            tree.column(column, width=100)
        
        for index, row in df.head(10).iterrows():  # Afficher toutes les données
            tree.insert("", "end", values=list(row))
    except Exception as e:
        messagebox.showerror("Erreur de chargement des données", str(e))

# Fonction pour effacer le contenu du Treeview
def effacer_contenu_treeview():
    if tree.winfo_exists():  # Vérifie si tree existe
        for i in tree.get_children():
            tree.delete(i)

# Fonction pour créer le tableau de données manuel
def afficher_champs_creation_donnees():
    effacer_contenu_right_frame()
    global tree, entry_nombre_colonnes, entry_nombre_lignes, entry_col_names, bouton_generer, bouton_effacer

    entry_nombre_colonnes = ctk.CTkEntry(right_frame, placeholder_text="Nombre de colonnes")
    entry_nombre_colonnes.pack(pady=10)

    entry_nombre_lignes = ctk.CTkEntry(right_frame, placeholder_text="Nombre de lignes")
    entry_nombre_lignes.pack(pady=10)

    entry_col_names = ctk.CTkEntry(right_frame, placeholder_text="Noms des colonnes")
    entry_col_names.pack(pady=20)

    bouton_generer = ctk.CTkButton(right_frame, text="Générer Tableau", command=creer_tableau_manuel)
    bouton_generer.pack(pady=10)

    tree = ttk.Treeview(right_frame, style="Treeview")
    tree.tag_configure('alternate', background='#E8E8E8')
    tree.pack(side='top', fill='both', expand=True)

# Fonction pour génerer le tableau 
def creer_tableau_manuel():
    global entry_grid, num_rows, num_cols
    try:
        num_cols = int(entry_nombre_colonnes.get())
        num_rows = int(entry_nombre_lignes.get())
        col_names = [col.strip() for col in entry_col_names.get().split(',')]
    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer un nombre valide.")
        return
    # Vérifier si le nombre de noms de colonnes fourni correspond au nombre de colonnes spécifié
    if len(col_names) != num_cols:
        messagebox.showerror("Erreur", "Le nombre de noms de colonnes ne correspond pas au nombre de colonnes spécifié.")
        return
    # Effacer le contenu du Treeview avant de créer un nouveau tableau
    effacer_contenu_treeview()
    tree["columns"] = col_names
    tree["show"] = "headings"
    for col_name in col_names:
        tree.heading(col_name, text=col_name)
        # Ajustez la largeur de la colonne ici si nécessaire
        tree.column(col_name, width=100)
    # Créer un cadre pour contenir les entrées des valeurs
    entries_frame = ctk.CTkFrame(right_frame)
    entries_frame.pack(fill='x')
    entry_grid = [] 
    for i in range(num_rows):
        row_entries = []
        for j in range(num_cols):
            entry = ctk.CTkEntry(entries_frame, placeholder_text="Valeur")
            entry.grid(row=i, column=j, sticky='nsew', padx=5, pady=5)  
            row_entries.append(entry)
        entry_grid.append(row_entries)
        # Configurez le poids des colonnes pour qu'elles se redimensionnent avec la fenêtre
    for j in range(num_cols):
        entries_frame.grid_columnconfigure(j, weight=1)
    # Boutton pour sauvegarder les valeurs
    col_start = (num_cols // 2) - 1 # Soustrayez 1 pour ajuster le bouton correctement au centre
    save_button = ctk.CTkButton(entries_frame, text="Sauvegarder les valeurs", command=save_values)
    save_button.grid(row=num_rows, column=col_start, columnspan=2, padx=10, pady=10)

# Fonction pour sauvegarder les valeurs entrees par l'utilisateur
def save_values():
    global entry_grid, num_rows, num_cols, tree
    # Rassembler les valeurs de chaque champ d'entrée
    values = []
    for row_entries in entry_grid:
        row_values = [entry.get() for entry in row_entries]
        values.append(row_values)
    # Insérer les nouvelles valeurs dans le Treeview
    for row_values in values:
        tree.insert("", "end", values=row_values)
    # Effacer les champs d'entrée après insertion
    for row_entries in entry_grid:
        for entry in row_entries:
            entry.delete(0, tk.END)

    # Créer un DataFrame avec les valeurs
    df_manual = pd.DataFrame(values, columns=tree['columns'])

    # Demander à l'utilisateur de choisir le nom du fichier
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])

    # Vérifier si l'utilisateur a annulé la sélection du fichier
    if not file_path:
        return

    # Enregistrer le DataFrame au format CSV avec le nom de fichier choisi par l'utilisateur
    df_manual.to_csv(file_path, index=False)

    # Afficher un message de confirmation
    messagebox.showinfo("Enregistrement CSV", f"Les données manuelles ont été enregistrées dans {file_path}")

#Fonction pour normaliser les caracteristiques
def normaliser_caracteristiques(df, exclure):
    colonnes_a_exclure = [exclure] if exclure is not None else []
    colonnes_numeriques = df.select_dtypes(include=[np.number]).drop(columns=colonnes_a_exclure, errors='ignore')
    scaler = StandardScaler()
    df_norm = scaler.fit_transform(colonnes_numeriques)
    df_norm = pd.DataFrame(df_norm, columns=colonnes_numeriques.columns, index=df.index)
    df.loc[:, colonnes_numeriques.columns] = df_norm
    return df

# Fonction pour mettre à jour le Treeview avec le DataFrame modifié
def mettre_a_jour_treeview(df):
    tree.delete(*tree.get_children())
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"
    for column in tree["columns"]:
        tree.heading(column, text=column)
        tree.column(column, width=100)
    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))

# Fonction qui crée les widgets pour la préparation des données
def creer_widgets_preparation_donnees():
    global entry_colonne_a_supprimer, entry_ancienne_valeur, entry_nouvelle_valeur, var_valeurs_manquantes, var_normalisation, preparation_frame
    # S'assurer que le cadre de préparation est nettoyé avant de recréer
    fermer_preparation_frame()
    # Création du cadre pour la préparation des données
    preparation_frame = ctk.CTkFrame(right_frame)
    preparation_frame.pack(fill='x', expand=False, padx=20, pady=10)
    # Section pour la suppression de colonne
    suppression_frame = ctk.CTkFrame(preparation_frame)
    suppression_frame.pack(fill='x', padx=10, pady=5)
    ctk.CTkLabel(suppression_frame, text="Nom de la colonne à supprimer:").pack(side='left')
    entry_colonne_a_supprimer = ctk.CTkEntry(suppression_frame)
    entry_colonne_a_supprimer.pack(side='left', fill='x', expand=True, padx=10)
    # Section pour le remplacement de valeur
    remplacement_frame = ctk.CTkFrame(preparation_frame)
    remplacement_frame.pack(fill='x', padx=10, pady=5)
    ctk.CTkLabel(remplacement_frame, text="Valeur à remplacer:").pack(side='left')
    entry_ancienne_valeur = ctk.CTkEntry(remplacement_frame)
    entry_ancienne_valeur.pack(side='left', fill='x', expand=True, padx=10)
    ctk.CTkLabel(remplacement_frame, text="Nouvelle valeur:").pack(side='left')
    entry_nouvelle_valeur = ctk.CTkEntry(remplacement_frame)
    entry_nouvelle_valeur.pack(side='left', fill='x', expand=True, padx=10)
    # Section pour la gestion des valeurs manquantes
    valeurs_manquantes_frame = ctk.CTkFrame(preparation_frame)
    valeurs_manquantes_frame.pack(fill='x', padx=10, pady=5, expand=True)
    ctk.CTkLabel(valeurs_manquantes_frame, text="Gestion des valeurs manquantes:", anchor='w').pack(fill='x', expand=True)
    var_valeurs_manquantes = tk.IntVar()
    ctk.CTkRadioButton(valeurs_manquantes_frame, text="Remplacer par la moyenne", variable=var_valeurs_manquantes, value=1).pack(side='left', fill='x', expand=True)
    ctk.CTkRadioButton(valeurs_manquantes_frame, text="Supprimer les lignes", variable=var_valeurs_manquantes, value=2).pack(side='left', fill='x', expand=True)
    # Section de normalisation des caractéristiques
    normalisation_frame = ctk.CTkFrame(preparation_frame)
    normalisation_frame.pack(fill='x', padx=10, pady=5, expand=True)
    var_normalisation = tk.IntVar()
    ctk.CTkCheckBox(normalisation_frame, text="Normaliser les caractéristiques numériques", variable=var_normalisation).pack()

    # Boutons pour appliquer les modifications et fermer
    bouton_frame = ctk.CTkFrame(preparation_frame)
    bouton_frame.pack(fill='x', padx=10, pady=5)
    bouton_appliquer = ctk.CTkButton(bouton_frame, text="Appliquer", command=appliquer_preparation, fg_color="#4CAF50", text_color="white")
    bouton_appliquer.pack(side='left', expand=True, padx=5)
    bouton_fermer = ctk.CTkButton(bouton_frame, text="Fermer", command=fermer_preparation_frame, fg_color="#F44336", text_color="white")
    bouton_fermer.pack(side='left', expand=True, padx=5)

entry_colonne_a_supprimer = None  # Ajoutez cette ligne pour déclarer la variable globalement
entry_ancienne_valeur = None  # Ajoutez cette ligne pour déclarer la variable globalement
entry_nouvelle_valeur = None 
var_normalisation = None
var_valeurs_manquantes = None
# Fonction pour appliquer la préparation des données
def appliquer_preparation():
    global df
    try:
        # Suppression des colonnes spécifiées
        colonnes_a_supprimer = entry_colonne_a_supprimer.get().split(',')
        df.drop(columns=[col.strip() for col in colonnes_a_supprimer if col.strip() in df.columns], inplace=True)

        # Remplacement de valeur
        ancienne_valeur = entry_ancienne_valeur.get()
        nouvelle_valeur = entry_nouvelle_valeur.get()
        if ancienne_valeur:
            if nouvelle_valeur:  # Si une nouvelle valeur est fournie, utilisez-la pour le remplacement
                df.replace(ancienne_valeur, nouvelle_valeur, inplace=True)
            else:  # Sinon, remplacez par NaN
                df.replace(ancienne_valeur, np.nan, inplace=True)

        # Gestion des valeurs manquantes
        if var_valeurs_manquantes.get() == 1:
            df_numerical = df.select_dtypes(include=[np.number])
            df[df_numerical.columns] = df_numerical.fillna(df_numerical.mean())
        elif var_valeurs_manquantes.get() == 2:
            df.dropna(inplace=True)

        # Normalisation des caractéristiques
        if var_normalisation.get():
            if 'species' in df.columns:  # Dataset Iris
                df = normaliser_caracteristiques(df, exclure='species')
            elif 'survived' in df.columns:  # Dataset Titanic
                df = normaliser_caracteristiques(df, exclure='survived')
            else:
                # Si aucune des colonnes cibles n'est dans le DataFrame, normaliser toutes les colonnes numériques
                df = normaliser_caracteristiques(df, exclure=None)

        # Mettez à jour le Treeview avec le nouveau DataFrame
        mettre_a_jour_treeview(df)

        messagebox.showinfo("Préparation des données", "Les données ont été préparées avec succès.")
    except ValueError as ve:
        messagebox.showerror("Erreur de préparation des données", f"Erreur de valeur : {ve}")
    except Exception as e:
        messagebox.showerror("Erreur de préparation des données", str(e))

# Fonction pour valider la préparation et quitter
def valider_preparation_et_quitter():
    # Vous pouvez ajouter des vérifications supplémentaires ici avant de quitter
    app.quit()

# Fonction pour fermer la section de préparation
def fermer_preparation_frame():
    pass

# Variables globales
preparation_frame = None
visualisation_frame = None


def masquer_visualisation_frame():
    # Cette fonction masque le cadre de visualisation
    visualisation_frame.pack_forget()

def creer_widgets_visualisation():
    global visualisation_frame

    # S'assurer que le cadre de préparation est nettoyé avant de recréer
    fermer_preparation_frame()

    # Création du cadre pour la visualisation des données
    visualisation_frame = ctk.CTkFrame(right_frame)
    visualisation_frame.pack(fill='both', expand=True, padx=20, pady=10)

    def executer_visualisation():

        selected_type = type_var.get()
        x_column = x_entry.get()

        if not x_column or x_column not in df.columns:
            messagebox.showerror("Erreur", "Colonne X invalide")
            return

        if selected_type == "Scatter Plot":
            y_column = y_entry.get()
            if not y_column or y_column not in df.columns:
                messagebox.showerror("Erreur", "Colonne Y invalide")
                return

            # Scatter Plot
            fig, ax = plt.subplots()
            ax.scatter(df[x_column], df[y_column])
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title('Scatter Plot')

        elif selected_type == "Histogramme":
            # Histogramme
            fig, ax = plt.subplots()
            ax.hist(df[x_column], bins='auto', edgecolor='black')
            ax.set_xlabel(x_column)
            ax.set_ylabel('Fréquence')
            ax.set_title('Histogramme')

        # Afficher le graphique
        canvas = FigureCanvasTkAgg(fig, master=visualisation_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side='top', fill='both', expand=True)
        close_button.pack(side='bottom', pady=10)

    # Effacer le contenu précédent de visualisation_frame
    effacer_contenu_visualisation_frame()

    # Type de visualisation
    type_var = tk.StringVar(value="Scatter Plot")
    ctk.CTkRadioButton(visualisation_frame, text="Scatter Plot", variable=type_var, value="Scatter Plot").pack(anchor='w')
    ctk.CTkRadioButton(visualisation_frame, text="Histogramme", variable=type_var, value="Histogramme").pack(anchor='w')

    # Champ pour l'axe X
    ctk.CTkLabel(visualisation_frame, text="Colonne pour l'axe X:").pack()
    x_entry = ctk.CTkEntry(visualisation_frame)
    x_entry.pack()

    # Champ pour l'axe Y, visible seulement pour Scatter Plot
    y_entry = ctk.CTkEntry(visualisation_frame)
    y_label = ctk.CTkLabel(visualisation_frame, text="Colonne pour l'axe Y:")

    def toggle_y_entry(*args):
        if type_var.get() == "Scatter Plot":
            y_label.pack()
            y_entry.pack()
        else:
            y_label.pack_forget()
            y_entry.pack_forget()

    y_label = ctk.CTkLabel(visualisation_frame, text="Colonne pour l'axe Y:")
    type_var.trace("w", toggle_y_entry)
    toggle_y_entry()  # Appel initial pour définir l'état visible/invisible

    # Bouton pour exécuter la visualisation
    execute_button = ctk.CTkButton(visualisation_frame, text="Visualiser", command=executer_visualisation)
    execute_button.pack(pady=10)

    # Bouton pour fermer la visualisation
    close_button = ctk.CTkButton(visualisation_frame, text="Fermer la visualisation", command=masquer_visualisation_frame)
    close_button.pack(side='bottom', pady=10)

# Cette fonction efface le contenu de visualisation_frame
def effacer_contenu_visualisation_frame():
    for widget in visualisation_frame.winfo_children():
        widget.destroy()

# Cette fonction entraine les modeles
def entrainer_modele():
    global df, model, X_test, y_test, y_pred, label_encoder

    def lancer_entrainement():
        global model, X_test, y_test, y_pred, label_encoder, df
        selected_algo = algo_combobox.get()

        try:
        # Sélection et préparation des données
            if 'species' in df.columns:
                X = df.drop('species', axis=1)
                y = df['species']
                if selected_algo in ["Régression logistique", "KNN"]:
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)
            elif 'survived' in df.columns:
                X = df.drop('survived', axis=1)
                y = df['survived']
                if selected_algo in ["Régression logistique", "KNN", "SVM"]:
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)
            else:
                messagebox.showerror("Erreur", "Dataset non reconnu.")
                return
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if selected_algo == "Régression logistique":
                model = LogisticRegression()
            elif selected_algo == "Régression linéaire":
                model = LinearRegression()
            elif selected_algo == "KNN":
                model = KNeighborsClassifier()
            elif selected_algo == "DT":
                model = DecisionTreeClassifier()
            elif selected_algo == "Random Forests":
                model = RandomForestClassifier()
            elif selected_algo == "SVM":
                model = SVC()
            else:
                messagebox.showerror("Erreur", f"Algorithme {selected_algo} non supporté")
                return

    # Entraînement du modèle
            model.fit(X_train, y_train)

    # Prédictions sur l'ensemble de test
            y_pred = model.predict(X_test)

    # Affichage du résultat de l'entraînement
            result_label = tk.Label(training_frame, text="Modèle entraîné avec succès!", fg="green")
            result_label.pack()

        except Exception as e:
            messagebox.showerror("Erreur d'entraînement", str(e))
    # Effacer le contenu précédent de right_frame
    effacer_contenu_right_frame()
    # Frame pour la sélection de l'algorithme
    selection_frame = tk.Frame(right_frame)
    selection_frame.pack(pady=10)

    # Combobox pour la sélection de l'algorithme
    tk.Label(selection_frame, text="Choisissez un algorithme :").pack(side="left", padx=5)
    algo_combobox = ttk.Combobox(selection_frame, values=["Régression logistique", "Régression linéaire","KNN", "DT", "Random Forests", "SVM"])
    algo_combobox.pack(side="left", padx=5)
    algo_combobox.set("Sélectionnez un algorithme")

    # Bouton pour lancer l'entraînement
    train_button = ctk.CTkButton(selection_frame, text="Entraîner le modèle", command=lancer_entrainement)
    train_button.pack(pady=10)
    
    # Bouton pour valider le modèle
    bouton_valider = ctk.CTkButton(
    right_frame,
    text="Valider le modèle",
    command=valider_modele  # Utiliser une fonction lambda ici
    )
    bouton_valider.pack(pady=10)

    # Frame pour les résultats de l'entraînement
    training_frame = tk.Frame(right_frame)
    training_frame.pack()

# Cette fonction pour valider les moodeles
def valider_modele():
    global model, X_test, y_test, y_pred, label_encoder, right_frame

    # Vérifier si le modèle a été entraîné
    if model is None:
        messagebox.showerror("Erreur", "Aucun modèle n'a été entraîné.")
        return

    # Faire des prédictions sur les données de test
    y_pred = model.predict(X_test)

    # Calculer les métriques de performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Affichage des métriques dans right_frame
    Label(right_frame, text=f"Précision (Accuracy): {accuracy:.2f}").pack()
    Label(right_frame, text=f"Précision (Precision): {precision:.2f}").pack()
    Label(right_frame, text=f"Rappel (Recall): {recall:.2f}").pack()
    Label(right_frame, text=f"Score F1: {f1:.2f}").pack()

# Cette fonction pour visualiser les resultats de modele entrainer
def visualiser_resultats():
    global model, X_test, y_test, y_pred, label_encoder

    # Vérifier si le modèle a été entraîné
    if model is None:
        messagebox.showerror("Erreur", "Aucun modèle n'a été entraîné.")
        return

    # Faire des prédictions sur les données de test
    y_pred = model.predict(X_test)

    # Calculer les métriques de performance
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    effacer_contenu_right_frame()

    # Création de graphiques pour la visualisation
    fig = Figure(figsize=(15, 8))

    # Sous-graphique pour la matrice de confusion
    ax1 = fig.add_subplot(121)
    sns.heatmap(confusion_mat, annot=True, fmt='g', ax=ax1)
    ax1.set_title('Matrice de confusion')

    # Sous-graphique pour le rapport de classification
    ax2 = fig.add_subplot(122)
    sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True, ax=ax2)
    ax2.set_title('Rapport de classification')

    # Créer un widget de canevas pour afficher la figure
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    # Afficher des résultats supplémentaires si nécessaire
    if isinstance(model, LogisticRegression):
        print("Coefficients du modèle:")
        print(model.coef_)
    
    return confusion_mat, classification_rep

# Cette fonction pour exporter les resultats sous forme fichier pdf
def exporter_resultats_pdf():
    global model, y_pred

    # Vérifier si le modèle a été entraîné
    if model is None:
        messagebox.showerror("Erreur", "Aucun modèle n'a été entraîné.")
        return
    
    # Vérifier si les métriques ont été calculées
    try:
        confusion_mat, classification_rep = visualiser_resultats()
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur s'est produite lors du calcul des métriques : {str(e)}")
        return

    # Créer une nouvelle figure pour les graphiques
    fig = Figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    sns.heatmap(confusion_mat, annot=True, fmt='g', ax=ax1)
    ax1.set_title('Matrice de confusion')

    ax2 = fig.add_subplot(122)
    sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True, ax=ax2)
    ax2.set_title('Rapport de classification')

    # Exporter la figure au format PDF
    try:
        fig.savefig("resultats.pdf", format="pdf")
        messagebox.showinfo("Succès", "Les résultats ont été sauvegardés au format PDF avec succès.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur s'est produite lors de la sauvegarde au format PDF: {str(e)}")

# Frame de départ avec les boutons Démarrer et Quitter
start_frame = ctk.CTkFrame(app)
start_frame.pack(fill='both', expand=True)

import tkinter as tk
from PIL import Image, ImageTk
import customtkinter as ctk  # Assuming ctk is a custom tkinter module you are using

def start_analysis():
    # Your start_analysis function code
    pass

app = ctk.CTk()  # Create main application window

# Load the background image
background_image = Image.open("path_to_your_image.jpg")  # Load your image file
background_photo = ImageTk.PhotoImage(background_image)

start_button = ctk.CTkButton(start_frame, text="Démarrer l'analyse", command=start_analysis)
start_button.pack(pady=20, side='top')

quit_button = ctk.CTkButton(start_frame, text="Quitter", command=app.quit)
quit_button.pack(pady=20, side='top')

# Bouton pour créer des données manuellement dans left_frame
bouton_creer_donnees = ctk.CTkButton(left_frame, text="Créer ", command=afficher_champs_creation_donnees)
bouton_creer_donnees.pack(pady=10)

# Bouton pour charger des données 
load_data_button = ctk.CTkButton(left_frame, text="Charger Données", command=charger_donnees)
load_data_button.pack(pady=10)

# Bouton pour visualiser des données 
bouton_visualisation = ctk.CTkButton(left_frame, text="Visualiser Données", command=creer_widgets_visualisation)
bouton_visualisation.pack(pady=10)

# Bouton pour afficher les widgets de préparation des données
bouton_preparation_donnees = ctk.CTkButton(left_frame, text="Préparer Données", command=creer_widgets_preparation_donnees)
bouton_preparation_donnees.pack(pady=10)

# Bouton pour entraîner le modèle
bouton_entrainer = ctk.CTkButton(left_frame, text="Entrainer Modèle", command=entrainer_modele)
bouton_entrainer.pack(pady=10)


# Bouton pour visualiser les résultats du modèle
bouton_visualiser_resultats = ctk.CTkButton(
    left_frame,
    text="Visualiser Résultats",
    command=visualiser_resultats)
bouton_visualiser_resultats.pack(pady=10)

bouton_exporter_resultats_pdf = ctk.CTkButton(
    left_frame,
    text="Exporter Résultats",
    command=exporter_resultats_pdf)
bouton_exporter_resultats_pdf.pack(pady=10)

# Bouton pour fermer l'application
bouton_fermer = ctk.CTkButton(left_frame, text="Quitter", command=fermer_application)
bouton_fermer.pack(pady=10)

app.mainloop()

