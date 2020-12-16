import dash_bootstrap_components as dbc
import dash_html_components as html

modal = html.Div([
    dbc.Button("Descriptif du brief", id="open-centered"),
    dbc.Modal(
        [
            dbc.ModalHeader("La Roue des Emotions"),
            dbc.ModalBody(["Construit d’après les travaux du psychologue américain Robert Plutchik, la roue des émotions est un modèle des émotions humaines et peut facilement servir à définir des personnages, ainsi que leur évolution dans une trame narrative. Est-il possible d'identifier des émotions dans des phrases narratives issues de communications écrites ?",
                html.Br(),
                html.Strong("Contexte du Projet"),
                html.Br(),
                "Depuis quelques années, les dispositifs de communication médiatisée par ordinateur (CMO) sont massivement utilisés, aussi bien dans les activités professionnelles que personnelles. Ces dispositifs permettent à des participants distants physiquement de communiquer. La plupart implique une communication écrite médiatisée par ordinateur (CEMO) : forums de discussion, courrier électronique, messagerie instantanée. Les participants ne s’entendent pas et ne se voient pas mais peuvent communiquer par l’envoi de messages écrits, qui combinent, généralement, certaines caractéristiques des registres écrit et oral (Marcoccia, 2000a ; Marcoccia, Gauducheau, 2007 ; Riva, 2001).",
                html.Br(),
                "Imaginez que vous souhaitez savoir ce qui se passe derrière votre écran d'ordinateur, quels sont vos contacts les plus actifs et quelle est leur personnalité (pas banal comme question !!). Vous allez alors vous lancer dans l’analyse de leur narration et tenter d’extraire quelle émotion se dégage de chacune des phrases.",
                html.Br(),
                "Chez Simplon nous utilisons tous les jours des outils de discussion textuels et nous construisons nos relations sociales et professionnelles autour de ces dispositifs. Pour entretenir des rapports sociaux stables, sereins, de confiance et efficaces, au travers des outils de communication écrites, lorsqu'il n'est pas possible d'avoir la visio (avec caméra), il est nécessaire de détecter des éléments 'clés' dans les channels de discussions / mails qui nous permettront de déceler de la colère, de la frustration, de la tristesse ou encore de la joie de la part d'un collègue ou d'un amis pour adapter nos relations sociales. En tant qu'expert en data science, nous allons vous demander de développer un modèle de machine learning permettant de classer les phrases suivant l'émotion principale qui en ressort.",
                html.Br(),
                "Pour des questions d’ordre privé, nous ne vous demanderons pas de nous communiquer les conversations provenant de votre réseau social favori ou de vos emails mais nous allons plutôt vous proposer deux jeux de données contenant des phrases, ces fichiers ayant déjà été annoté.",
                html.Br(),
                "Vous devrez proposer plusieurs modèles de classification des émotions et proposer une analyse qualitative et quantitative de ces modèles en fonction de critères d'évaluation. Vous pourrez notamment vous appuyer sur les outils de reporting des librairies déjà étudiées. Vous devrez investiguer aux travers de librairies d'apprentissage automatique standards et de traitement automatique du langage naturel comment obtenir les meilleurs performance de prédiction possible en prenant en compte l'aspect multi-class du problème et en explorant l'impact sur la prédiction de divers prétraitement tel que la suppression des stop-words, la lemmatisation et l'utilisation de n-grams, et différente approche pour la vectorisation.",
                html.Br(),
                "Vous devrez travailler dans un premier temps avec le jeu de données issu de Kaggle pour réaliser vos apprentissage et l'évaluation de vos modèles.",
                html.Br(),
                "Dans l'objectif d'enrichir notre prédictions nous souhaitons augmenter notre jeux de donneés. Vous devrez donc travailler dans un deuxième temps avec le jeux de données fournie, issue de data.world afin de :",
            ]),
            dbc.ModalFooter(
                dbc.Button(
                    "Close", id="close-centered", className="ml-auto"
                )
            ),
        ],
        id="modal_Home",
        size="xl",
        centered=True,
    ),
])