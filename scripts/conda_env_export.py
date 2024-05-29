import platform
import re
import subprocess
from subprocess import CalledProcessError

try:
    import yaml
except ImportError:
    raise ImportError("Ce script à besoin que vous installiez PyYaml, veuillez faire 'pip install pyyaml'")


def run_command(command: str) -> str:
    """
    Exécute une commande bash en supprimant les séquences ANSI des sorties

    Args:
        command (str): Commande bash à exécuter.

    Returns:
        str: La sortie de la commande après nettoyage des séquences ANSI.
    """

    # Exécution de la commande en capturant la sortie standard et les erreurs.
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    except CalledProcessError as exception:
        raise Exception(f"Command '{command}' returned non-zero exit status 1.\n{exception.stderr}")

    # Obtient le résultat de la commande
    stdout = result.stdout

    # Utilisation d'une expression régulière pour supprimer les séquences ANSI de la sortie.
    clean_output = re.sub(r'\x1B[\[\]()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]', '', stdout)
    return clean_output


def export_conda_environment(env_name: str, output_path: str = "environment.yml"):
    """
    Exporte un fichier d'installation de l'environement Conda.

    Args:
        env_name (str): Nom de l'environnement Conda à gérer.
        output_path (str): Chemin de sortie du fichier d'export
    """

    # Si besoin, active conda pour le shell
    conda_init_command = ""
    os_name = platform.system()
    length_result_conda_init = 0

    # Ubuntu need to active shell and init
    if os_name == "Linux":
        conda_init_command = ". ~/miniconda3/etc/profile.d/conda.sh && conda init && "
        length_result_conda_init = len(run_command(conda_init_command[:-4]))  # enlève-le " && "

    # Avec conda initialisé, active l'environement
    env_command = f"{conda_init_command}conda activate {env_name} && conda env export --from-history"
    channels_command = f"{conda_init_command}conda activate {env_name} && conda env export -c conda-forge"
    pip_command = f"{conda_init_command}conda run -n {env_name} pip list --format=freeze"

    # '--from-history' ne fournit pas les canaux, donc nous devons les obtenir séparément
    channels_result = run_command(channels_command)[length_result_conda_init:]
    channels_data = yaml.safe_load(channels_result)
    channels = channels_data['channels']

    # Activer l'environnement Conda
    env_result = run_command(env_command)[length_result_conda_init:]

    # Lecture des données YAML exportées
    env_data = yaml.safe_load(env_result)

    # Modifier le préfixe
    env_data['prefix'] = '~'
    env_data['channels'] = channels

    # Ajouter des dépendances pip
    pip_result = run_command(pip_command)[length_result_conda_init:]

    # Traitement de la sortie de pip freeze
    pip_dependencies = pip_result.split('\n')
    pip_dependencies = filter(lambda x: x != "", pip_dependencies)
    pip_dependencies = list(pip_dependencies)

    # Si aucune dépendance n'existe, crée une liste vide
    if env_data.get('dependencies') is None:
        env_data['dependencies'] = ['pip']

    # Ajouter les dépendances pip
    env_data['dependencies'].append({'pip': pip_dependencies})

    # Enregistrer les modifications dans un nouveau fichier YAML
    try:
        with open(output_path, 'w') as file:
            yaml.dump(env_data, file)
    except PermissionError as exception:
        raise Exception("Vous n'avez pas les permissions d'écriture sur le dossier du projet.\n"
                        f"Veuillez faire sur Ubuntu : 'sudo chown -R USER:USER FOLDER'\n{exception}")


if __name__ == "__main__":
    export_conda_environment('chess_pytorch_ai')
