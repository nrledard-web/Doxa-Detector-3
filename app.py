import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


# =========================================================
# Configuration générale
# =========================================================
st.set_page_config(
    page_title="DOXA Detector",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# Textes d'interface
# =========================================================
T = {
    "titre": "DOXA Detector",
    "sous_titre": "Analyse structurelle du discours fondée sur le noyau cognitif M = (G + N) − D.",
    "zone_texte": "Texte à analyser",
    "placeholder": "Collez ici un article, un post, un discours ou un extrait suffisamment long.",
    "analyser": "Analyser le texte",
    "charger_exemple": "Charger un exemple",
    "vider": "Vider",
    "onglet_analyse": "Analyse",
    "onglet_methode": "Méthode",
    "onglet_exemples": "Exemples",
    "resume": "Résumé du diagnostic",
    "noyau": "Noyau cognitif",
    "derivees_logiques": "Dérivées logiques",
    "derivees_discursives": "Dérivées discursives",
    "synthese": "Synthèse finale",
    "credibilite": "Crédibilité globale",
    "jauge_mensonge": "Jauge de mensonge",
    "pression_rhetorique": "Pression rhétorique",
    "signal_propagandiste": "Signal propagandiste",
    "mecroyance": "Mécroyance",
    "surconfiance": "Surconfiance",
    "calibration": "Calibration cognitive",
    "revisabilite": "Révisabilité",
    "cloture": "Clôture cognitive",
    "gnose": "Gnose (G)",
    "nous": "Nous (N)",
    "doxa": "Doxa (D)",
    "aucun_texte": "Veuillez coller un texte avant de lancer l’analyse.",
}


# =========================================================
# Utilitaires
# =========================================================
def borner(valeur: float, mini: float, maxi: float) -> float:
    return max(mini, min(maxi, valeur))


def pourcentage(valeur: float, maxi: float = 10.0) -> int:
    if maxi <= 0:
        return 0
    return int(borner((valeur / maxi) * 100, 0, 100))


def nettoyer_texte(texte: str) -> str:
    texte = texte.replace("\xa0", " ")
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte


def decouper_phrases(texte: str) -> List[str]:
    phrases = re.split(r"(?<=[\.!\?])\s+", texte)
    return [p.strip() for p in phrases if p.strip()]


def decouper_mots(texte: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+", texte.lower())


def ratio(numerateur: float, denominateur: float) -> float:
    if denominateur == 0:
        return 0.0
    return numerateur / denominateur


# =========================================================
# Lexiques heuristiques
# =========================================================
CONNECTEURS_NUANCE = {
    "peut-être", "peut", "semble", "semblet", "semblez", "semblement",
    "probablement", "possible", "possiblement", "souvent", "parfois",
    "selon", "toutefois", "cependant", "néanmoins", "en partie", "relativement",
    "sous réserve", "il est possible", "il semble", "à confirmer", "à vérifier",
    "dans certains cas", "on peut supposer", "en apparence", "globalement"
}

ABSOLUS = {
    "toujours", "jamais", "certainement", "évidemment", "forcément", "indiscutablement",
    "incontestable", "incontestablement", "absolument", "preuve", "prouvé", "vérité",
    "vérité absolue", "sans aucun doute", "tout le monde", "personne", "aucun", "tous"
}

TERMES_EMOTIONNELS = {
    "scandale", "honte", "catastrophe", "trahison", "mensonge", "trahi", "effondrement",
    "désastre", "criminel", "ignoble", "dangereux", "révoltant", "terrifiant", "choquant",
    "massacre", "corrompu", "pourri", "manipulation", "explosion", "urgence", "panique"
}

TERMES_PROPAGANDE = {
    "ennemi", "traîtres", "complot", "système", "élite", "propagande", "peuple",
    "oligarchie", "soumission", "lavage", "endoctrinement", "invasion", "censure",
    "résistance", "trahison", "collabos", "marionnettes", "dictature", "vendus"
}

MARQUEURS_FACTUELS = {
    "selon", "rapport", "étude", "source", "sources", "document", "données", "chiffres",
    "pourcentage", "statistique", "enquête", "analyse", "publication", "archive", "article",
    "décret", "loi", "tribunal", "cour", "publication", "journal", "université", "rapporteur"
}

MARQUEURS_EXPERIENCE = {
    "j'ai", "j’ai", "nous avons", "j'observe", "j’observe", "j'ai vu", "j’ai vu",
    "j'ai constaté", "j’ai constaté", "expérience", "terrain", "vécu", "observé",
    "concret", "réel", "pratique", "directement"
}

INJONCTIONS = {
    "il faut", "il faut que", "vous devez", "on doit", "réveillez-vous", "ouvrez les yeux",
    "comprenez", "agissez", "refusez", "rejetez", "combattez", "dénoncez"
}


# =========================================================
# 1. Noyau cognitif
# =========================================================
def calculer_gnose(texte: str) -> float:
    mots = decouper_mots(texte)
    phrases = decouper_phrases(texte)
    nb_mots = len(mots)
    nb_phrases = max(len(phrases), 1)

    nb_chiffres = len(re.findall(r"\b\d+[\d\.,%]*\b", texte))
    nb_dates = len(re.findall(r"\b(19|20)\d{2}\b", texte))
    nb_marqueurs_factuels = sum(texte.lower().count(m) for m in MARQUEURS_FACTUELS)
    nb_connecteurs_logiques = len(re.findall(r"\b(car|donc|ainsi|puisque|par conséquent|en effet|or|cependant|néanmoins)\b", texte.lower()))

    score = (
        min(nb_chiffres * 0.8, 2.5)
        + min(nb_dates * 0.7, 1.5)
        + min(nb_marqueurs_factuels * 0.45, 3.0)
        + min(nb_connecteurs_logiques * 0.3, 2.0)
        + min(ratio(nb_mots, nb_phrases) / 8, 1.5)
    )
    return borner(score, 0.0, 10.0)


def calculer_nous(texte: str) -> float:
    texte_min = texte.lower()
    mots = decouper_mots(texte)
    nb_mots = len(mots)
    vocab_unique = len(set(mots))
    richesse = ratio(vocab_unique, max(nb_mots, 1))

    nb_marqueurs_experience = sum(texte_min.count(m) for m in MARQUEURS_EXPERIENCE)
    nb_nuance = sum(texte_min.count(m) for m in CONNECTEURS_NUANCE)
    nb_oppositions = len(re.findall(r"\b(mais|cependant|toutefois|pourtant|néanmoins|or)\b", texte_min))

    score = (
        min(richesse * 8, 3.5)
        + min(nb_marqueurs_experience * 0.8, 2.5)
        + min(nb_nuance * 0.45, 2.0)
        + min(nb_oppositions * 0.35, 2.0)
    )
    return borner(score, 0.0, 10.0)


def calculer_doxa(texte: str) -> float:
    texte_min = texte.lower()
    nb_absolus = sum(texte_min.count(m) for m in ABSOLUS)
    nb_injonctions = sum(texte_min.count(m) for m in INJONCTIONS)
    nb_emotion = sum(texte_min.count(m) for m in TERMES_EMOTIONNELS)
    nb_maj = len(re.findall(r"\b[A-ZÀ-ÖØ-Ý]{3,}\b", texte))
    nb_exclam = texte.count("!")

    score = (
        min(nb_absolus * 0.8, 4.0)
        + min(nb_injonctions * 0.9, 2.5)
        + min(nb_emotion * 0.35, 2.0)
        + min(nb_maj * 0.25, 0.8)
        + min(nb_exclam * 0.2, 0.7)
    )
    return borner(score, 0.0, 10.0)


def calculer_mecroyance(G: float, N: float, D: float) -> float:
    return (G + N) - D


# =========================================================
# 2. Dérivées logiques
# =========================================================
def calculer_surconfiance(G: float, N: float, D: float) -> float:
    return D - (G + N)


def calculer_calibration(G: float, N: float) -> float:
    return borner((G + N) / 2, 0.0, 10.0)


def calculer_revisabilite(M: float) -> float:
    # Mise à l’échelle de -10..20 vers 0..10
    return borner((M + 10) / 3, 0.0, 10.0)


def calculer_cloture(G: float, N: float, D: float) -> float:
    base = max(G + N, 0.1)
    fermeture = (D / base) * 10
    return borner(fermeture, 0.0, 10.0)


# =========================================================
# 3. Dérivées discursives
# =========================================================
def calculer_pression_rhetorique(texte: str) -> float:
    texte_min = texte.lower()
    nb_emotion = sum(texte_min.count(m) for m in TERMES_EMOTIONNELS)
    nb_absolus = sum(texte_min.count(m) for m in ABSOLUS)
    nb_injonctions = sum(texte_min.count(m) for m in INJONCTIONS)
    nb_exclam = texte.count("!")
    nb_questions = texte.count("?")

    score = (
        min(nb_emotion * 0.45, 3.0)
        + min(nb_absolus * 0.5, 2.5)
        + min(nb_injonctions * 0.8, 2.5)
        + min(nb_exclam * 0.25, 1.0)
        + min(nb_questions * 0.1, 1.0)
    )
    return borner(score, 0.0, 10.0)


def calculer_signal_propagandiste(texte: str) -> float:
    texte_min = texte.lower()
    nb_termes = sum(texte_min.count(m) for m in TERMES_PROPAGANDE)
    nb_binaire = len(re.findall(r"\b(eux|nous|eux-mêmes|traîtres|patriotes|ennemis|collabos)\b", texte_min))
    nb_repetitions = 0

    mots = decouper_mots(texte_min)
    frequences = pd.Series(mots).value_counts() if mots else pd.Series(dtype="int64")
    for mot, freq in frequences.items():
        if len(mot) >= 6 and freq >= 4:
            nb_repetitions += 1

    score = (
        min(nb_termes * 0.55, 4.0)
        + min(nb_binaire * 0.35, 2.5)
        + min(nb_repetitions * 0.45, 2.0)
    )
    return borner(score, 0.0, 10.0)


def calculer_jauge_mensonge(M: float, OVER: float, CLO: float, RP: float, PROP: float) -> float:
    base = 0.0

    # Surconfiance symétrique du noyau
    if OVER > 0:
        base += min(OVER * 1.4, 4.0)

    # Clôture cognitive
    base += min(CLO * 0.25, 2.0)

    # Renfort discursif
    base += min(RP * 0.22, 2.0)
    base += min(PROP * 0.22, 2.0)

    # Si la mécroyance reste positive, on atténue la dérive vers le mensonge
    if M > 0:
        base -= min(M * 0.18, 2.0)

    return borner(base, 0.0, 10.0)


# =========================================================
# 4. Synthèse finale
# =========================================================
def calculer_credibilite(M: float, CAL: float, REV: float, CLO: float, LIE_GAUGE: float, RP: float, PROP: float) -> float:
    score = 0.0
    score += CAL * 0.30
    score += REV * 0.22
    score += borner((M + 10) / 3, 0.0, 10.0) * 0.18
    score += (10 - CLO) * 0.10
    score += (10 - LIE_GAUGE) * 0.10
    score += (10 - RP) * 0.05
    score += (10 - PROP) * 0.05
    return borner(score, 0.0, 10.0)


# =========================================================
# Interprétations textuelles
# =========================================================
def interpreter_mecroyance(M: float) -> str:
    if M < 0:
        return "Structure de clôture cognitive : la certitude excède ici l’assise cognitive."
    if 0 <= M <= 10:
        return "Structure de mécroyance modérée : le discours reste cohérent, mais encore révisable."
    if 10 < M <= 17:
        return "Zone de lucidité croissante : l’équilibre entre savoir, intégration et doute reste favorable."
    if 17 < M < 19:
        return "Zone rare de haute intégration cognitive."
    return "Zone limite théorique de cohérence cognitive très élevée."


def interpreter_credibilite(score: float) -> str:
    if score < 2:
        return "Très faible"
    if score < 4:
        return "Douteuse"
    if score < 6:
        return "Fragile"
    if score < 8:
        return "Correcte"
    return "Robuste"


def interpreter_mensonge(score: float) -> str:
    if score < 2:
        return "Très faible signal de manipulation"
    if score < 4:
        return "Signal faible"
    if score < 6:
        return "Signal modéré"
    if score < 8:
        return "Signal fort"
    return "Signal très fort"


# =========================================================
# Moteur principal
# =========================================================
@dataclass
class ResultatAnalyse:
    G: float
    N: float
    D: float
    M: float
    OVER: float
    CAL: float
    REV: float
    CLO: float
    RP: float
    PROP: float
    LIE_GAUGE: float
    CRED: float
    resume_m: str
    resume_cred: str
    resume_lie: str


def analyser_texte(texte: str) -> ResultatAnalyse:
    G = calculer_gnose(texte)
    N = calculer_nous(texte)
    D = calculer_doxa(texte)
    M = calculer_mecroyance(G, N, D)

    OVER = calculer_surconfiance(G, N, D)
    CAL = calculer_calibration(G, N)
    REV = calculer_revisabilite(M)
    CLO = calculer_cloture(G, N, D)

    RP = calculer_pression_rhetorique(texte)
    PROP = calculer_signal_propagandiste(texte)
    LIE_GAUGE = calculer_jauge_mensonge(M, OVER, CLO, RP, PROP)

    CRED = calculer_credibilite(M, CAL, REV, CLO, LIE_GAUGE, RP, PROP)

    return ResultatAnalyse(
        G=G,
        N=N,
        D=D,
        M=M,
        OVER=OVER,
        CAL=CAL,
        REV=REV,
        CLO=CLO,
        RP=RP,
        PROP=PROP,
        LIE_GAUGE=LIE_GAUGE,
        CRED=CRED,
        resume_m=interpreter_mecroyance(M),
        resume_cred=interpreter_credibilite(CRED),
        resume_lie=interpreter_mensonge(LIE_GAUGE),
    )


# =========================================================
# Affichage Streamlit
# =========================================================
def afficher_jauge(titre: str, valeur: float, maximum: float = 10.0, aide: str | None = None):
    st.markdown(f"**{titre}**")
    st.progress(pourcentage(valeur, maximum))
    st.caption(f"{valeur:.2f} / {maximum:.2f}")
    if aide:
        st.caption(aide)


def afficher_carte_metric(titre: str, valeur: float, aide: str = ""):
    st.metric(titre, f"{valeur:.2f}")
    if aide:
        st.caption(aide)


# =========================================================
# Exemples
# =========================================================
EXEMPLE_1 = """
Selon un rapport publié en 2024 par un organisme indépendant, les dépenses ont augmenté de 12 % en trois ans.
Cette hausse ne suffit pas à elle seule à prouver une dégradation générale du système, mais elle constitue un indicateur sérieux.
Il faut donc comparer ces données avec d'autres sources, notamment les chiffres régionaux et les rapports précédents.
""".strip()

EXEMPLE_2 = """
Tout le monde sait désormais que le système nous ment et que les élites organisent sciemment notre appauvrissement.
Les médias répètent toujours les mêmes mensonges, et personne ne peut encore nier cette vérité absolue.
Réveillez-vous : il faut rejeter immédiatement cette propagande criminelle avant l'effondrement total.
""".strip()


# =========================================================
# Interface principale
# =========================================================
st.title(T["titre"])
st.caption(T["sous_titre"])

onglet1, onglet2, onglet3 = st.tabs([T["onglet_analyse"], T["onglet_methode"], T["onglet_exemples"]])

with onglet1:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        texte = st.text_area(
            T["zone_texte"],
            value=st.session_state.get("texte_analyse", ""),
            placeholder=T["placeholder"],
            height=260,
        )
    with col_b:
        st.write("")
        if st.button(T["charger_exemple"], use_container_width=True):
            st.session_state["texte_analyse"] = EXEMPLE_2
            st.rerun()
        if st.button(T["vider"], use_container_width=True):
            st.session_state["texte_analyse"] = ""
            st.rerun()

    if st.button(T["analyser"], type="primary", use_container_width=True):
        texte_nettoye = nettoyer_texte(texte)
        st.session_state["texte_analyse"] = texte_nettoye
        if not texte_nettoye:
            st.warning(T["aucun_texte"])
        else:
            resultat = analyser_texte(texte_nettoye)
            st.session_state["resultat"] = resultat

    resultat = st.session_state.get("resultat")

    if resultat:
        st.divider()
        st.subheader(T["resume"])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(T["credibilite"], f"{resultat.CRED:.2f} / 10", resultat.resume_cred)
        with c2:
            st.metric(T["jauge_mensonge"], f"{resultat.LIE_GAUGE:.2f} / 10", resultat.resume_lie)
        with c3:
            st.metric(T["mecroyance"], f"{resultat.M:.2f}", resultat.resume_m[:40] + "...")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(T["noyau"])
            afficher_jauge(T["gnose"], resultat.G, 10, "Densité de savoir articulé et d’indices vérifiables.")
            afficher_jauge(T["nous"], resultat.N, 10, "Degré d’intégration, de nuance et d’expérience réfléchie.")
            afficher_jauge(T["doxa"], resultat.D, 10, "Intensité de certitude saturante et de fermeture affirmée.")
            st.info(f"**M = (G + N) − D = {resultat.M:.2f}**")
            st.caption(resultat.resume_m)

        with col2:
            st.subheader(T["synthese"])
            afficher_jauge(T["credibilite"], resultat.CRED, 10, "Synthèse finale de fiabilité structurelle.")
            afficher_jauge(T["jauge_mensonge"], resultat.LIE_GAUGE, 10, "Distance entre erreur sincère et manipulation probable.")
            st.caption(
                "Sur cette échelle de crédibilité, un texte véritablement crédible se situe généralement dans la zone « robuste ».")

        st.divider()
        col3, col4 = st.columns(2)

        with col3:
            st.subheader(T["derivees_logiques"])
            afficher_jauge(T["surconfiance"], max(resultat.OVER, 0.0), 10, "Excès de certitude par rapport à l’assise cognitive.")
            afficher_jauge(T["calibration"], resultat.CAL, 10, "Niveau moyen de fondation cognitive G + N.")
            afficher_jauge(T["revisabilite"], resultat.REV, 10, "Capacité du discours à rester révisable.")
            afficher_jauge(T["cloture"], resultat.CLO, 10, "Rapport entre doxa et fondation cognitive.")

        with col4:
            st.subheader(T["derivees_discursives"])
            afficher_jauge(T["pression_rhetorique"], resultat.RP, 10, "Charge affective, absolue ou injonctive du texte.")
            afficher_jauge(T["signal_propagandiste"], resultat.PROP, 10, "Présence d’oppositions simplifiées et de vocabulaire de mobilisation.")
            afficher_jauge(T["jauge_mensonge"], resultat.LIE_GAUGE, 10, "Synthèse discursive de dérive possible vers la manipulation.")

        st.divider()
        st.subheader("Lecture structurée")
        st.markdown(
            f"""
- **Crédibilité : {resultat.resume_cred}**
- **Diagnostic cognitif :** {resultat.resume_m}
- **Signal de mensonge : {resultat.resume_lie}**
- **Lecture rapide :** plus la doxa domine la somme de la gnose et du nous, plus la clôture cognitive augmente ; plus cette clôture se combine à une forte pression rhétorique, plus la jauge de mensonge monte.
"""
        )

with onglet2:
    st.subheader("Hiérarchie correcte")
    st.code(
        """NOYAU
G
N
D
M

DÉRIVÉES LOGIQUES
OVER
CAL
REV
CLO

DÉRIVÉES DISCURSIVES
LIE_GAUGE
RP
PROP

SYNTHÈSE
CRED"""
    )

    st.markdown(
        """
### Noyau
- **G** : savoir articulé, indices, structure vérifiable.
- **N** : intégration, nuance, expérience et profondeur de compréhension.
- **D** : certitude saturante, rigidité du discours, fermeture.
- **M** : **mécroyance**, calculée par **M = (G + N) − D**.

### Dérivées logiques
- **OVER** : surconfiance symétrique du noyau.
- **CAL** : calibration cognitive.
- **REV** : révisabilité du discours.
- **CLO** : clôture cognitive.

### Dérivées discursives
- **RP** : pression rhétorique.
- **PROP** : signal propagandiste.
- **LIE_GAUGE** : jauge de mensonge issue du croisement cognitif et discursif.

### Synthèse
- **CRED** : crédibilité globale.
"""
    )

with onglet3:
    st.subheader("Exemple plutôt fondé")
    st.code(EXEMPLE_1)
    st.subheader("Exemple fortement saturé")
    st.code(EXEMPLE_2)

    if st.button("Analyser l’exemple fondé", use_container_width=True):
        st.session_state["texte_analyse"] = EXEMPLE_1
        st.session_state["resultat"] = analyser_texte(EXEMPLE_1)
        st.rerun()

    if st.button("Analyser l’exemple saturé", use_container_width=True):
        st.session_state["texte_analyse"] = EXEMPLE_2
        st.session_state["resultat"] = analyser_texte(EXEMPLE_2)
        st.rerun()
